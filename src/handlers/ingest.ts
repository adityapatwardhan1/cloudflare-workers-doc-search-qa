import { invalidateQueryCaches } from "../lib/cache";
import { chunkText } from "../lib/chunker";
import { errorResponse, jsonResponse } from "../lib/http";
import type { Env, IngestPayload, IngestResponse, TextChunk } from "../types";
import {
  EMBEDDING_BATCH_SIZE,
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MAX_CHARS,
  EMBEDDING_MODEL,
  VECTOR_UPSERT_BATCH_SIZE,
} from "../types";

interface EmbeddingResult {
  data?: number[][];
}

interface ChunkRecord {
  id: string;
  vectorId: string;
  index: number;
  content: string;
  embedding: number[];
}

interface ExistingDocument {
  id: string;
  vectorIds: string[];
}

function buildChunkId(documentId: string, chunkIndex: number): string {
  return `${documentId}:${chunkIndex}`;
}

async function getExistingDocument(env: Env, url: string): Promise<ExistingDocument | null> {
  const existing = await env.DB_D1.prepare(
    "SELECT id FROM documents WHERE url = ?",
  )
    .bind(url)
    .first<{ id: string }>();

  if (!existing) {
    return null;
  }

  const vectorRows = await env.DB_D1.prepare(
    "SELECT vector_id FROM chunks WHERE document_id = ?",
  )
    .bind(existing.id)
    .all<{ vector_id: string }>();

  return {
    id: existing.id,
    vectorIds: (vectorRows.results ?? []).map((row) => row.vector_id),
  };
}

async function acquireIngestLock(env: Env, url: string): Promise<boolean> {
  const now = new Date().toISOString();
  const result = await env.DB_D1.prepare(
    "INSERT INTO ingest_locks (url, locked_at) VALUES (?, ?)",
  )
    .bind(url, now)
    .run();

  return result.success;
}

async function releaseIngestLock(env: Env, url: string): Promise<void> {
  await env.DB_D1.prepare("DELETE FROM ingest_locks WHERE url = ?").bind(url).run();
}

async function deleteVectorsInBatches(env: Env, vectorIds: string[]): Promise<void> {
  for (let i = 0; i < vectorIds.length; i += VECTOR_UPSERT_BATCH_SIZE) {
    const batch = vectorIds.slice(i, i + VECTOR_UPSERT_BATCH_SIZE);
    await env.VECTOR_INDEX.deleteByIds(batch);
  }
}

function validateEmbeddings(
  data: number[][] | undefined,
  expectedCount: number,
): string | null {
  if (!data || !Array.isArray(data)) {
    return "Embedding model returned no data";
  }
  if (data.length !== expectedCount) {
    return `Expected ${expectedCount} embeddings, received ${data.length}`;
  }
  for (let i = 0; i < data.length; i++) {
    const vector = data[i];
    if (!Array.isArray(vector) || vector.length !== EMBEDDING_DIMENSIONS) {
      return `Embedding at index ${i} has invalid dimensions (expected ${EMBEDDING_DIMENSIONS})`;
    }
  }
  return null;
}

function validateChunkSizes(chunks: TextChunk[]): string | null {
  for (const chunk of chunks) {
    if (chunk.content.length > EMBEDDING_MAX_CHARS) {
      return `Chunk at index ${chunk.index} exceeds embedding model input limit (${chunk.content.length} > ${EMBEDDING_MAX_CHARS} chars)`;
    }
  }
  return null;
}

async function embedChunks(env: Env, chunks: TextChunk[]): Promise<ChunkRecord[]> {
  const sizeError = validateChunkSizes(chunks);
  if (sizeError !== null) {
    throw new Error(sizeError);
  }

  const records: ChunkRecord[] = [];

  for (let i = 0; i < chunks.length; i += EMBEDDING_BATCH_SIZE) {
    const batch = chunks.slice(i, i + EMBEDDING_BATCH_SIZE);
    const texts = batch.map((chunk) => chunk.content);

    const result = (await env.AI.run(EMBEDDING_MODEL, {
      text: texts,
    })) as EmbeddingResult;

    const validationError = validateEmbeddings(result.data, batch.length);
    if (validationError !== null) {
      throw new Error(validationError);
    }

    const embeddings = result.data as number[][];
    for (let j = 0; j < batch.length; j++) {
      records.push({
        id: "",
        vectorId: "",
        index: batch[j].index,
        content: batch[j].content,
        embedding: embeddings[j],
      });
    }
  }

  return records;
}

async function persistDocumentContent(
  env: Env,
  documentId: string,
  title: string,
  url: string,
  records: ChunkRecord[],
  isReingest: boolean,
): Promise<void> {
  const now = new Date().toISOString();
  const statements = [];

  if (isReingest) {
    statements.push(
      env.DB_D1.prepare(
        "UPDATE documents SET title = ?, updated_at = ? WHERE id = ?",
      ).bind(title, now, documentId),
    );
    statements.push(
      env.DB_D1.prepare("DELETE FROM chunks WHERE document_id = ?").bind(documentId),
    );
  } else {
    statements.push(
      env.DB_D1.prepare(
        "INSERT INTO documents (id, title, url, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
      ).bind(documentId, title, url, now, now),
    );
  }

  for (const record of records) {
    const chunkId = buildChunkId(documentId, record.index);
    record.id = chunkId;
    record.vectorId = chunkId;
    statements.push(
      env.DB_D1.prepare(
        "INSERT INTO chunks (id, document_id, chunk_index, content, vector_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      ).bind(chunkId, documentId, record.index, record.content, chunkId, now),
    );
  }

  const results = await env.DB_D1.batch(statements);
  for (const result of results) {
    if (!result.success) {
      throw new Error("Failed to persist document content");
    }
  }
}

async function upsertVectors(
  env: Env,
  documentId: string,
  title: string,
  url: string,
  records: ChunkRecord[],
): Promise<string[]> {
  const upsertedIds: string[] = [];

  for (let i = 0; i < records.length; i += VECTOR_UPSERT_BATCH_SIZE) {
    const batch = records.slice(i, i + VECTOR_UPSERT_BATCH_SIZE);
    const vectors: VectorizeVector[] = batch.map((record) => ({
      id: record.vectorId,
      values: record.embedding,
      metadata: {
        documentId,
        title,
        url,
        chunkIndex: record.index,
      },
    }));

    await env.VECTOR_INDEX.upsert(vectors);
    upsertedIds.push(...batch.map((record) => record.vectorId));
  }

  return upsertedIds;
}

async function rollbackIngest(
  env: Env,
  documentId: string,
  vectorIds: string[],
  isReingest: boolean,
  previousVectorIds: string[],
): Promise<boolean> {
  try {
    if (vectorIds.length > 0) {
      await deleteVectorsInBatches(env, vectorIds);
    }

    if (!isReingest) {
      await env.DB_D1.prepare("DELETE FROM chunks WHERE document_id = ?")
        .bind(documentId)
        .run();
      await env.DB_D1.prepare("DELETE FROM documents WHERE id = ?")
        .bind(documentId)
        .run();
    } else if (previousVectorIds.length > 0) {
      console.error(
        `Re-ingest rollback: D1 may be inconsistent for document ${documentId}; old vectors preserved`,
      );
    }

    return true;
  } catch (rollbackError) {
    console.error("Ingest rollback failed:", rollbackError);
    return false;
  }
}

export async function ingestDocument(
  payload: IngestPayload,
  env: Env,
): Promise<Response> {
  const chunks = chunkText(payload.content);
  if (chunks.length === 0) {
    return errorResponse(
      "EMPTY_CONTENT",
      "Document content produced no indexable chunks",
      400,
    );
  }

  const existing = await getExistingDocument(env, payload.url);
  const documentId = existing?.id ?? crypto.randomUUID();
  const previousVectorIds = existing?.vectorIds ?? [];
  const isReingest = existing !== null;

  const lockAcquired = await acquireIngestLock(env, payload.url);
  if (!lockAcquired) {
    return errorResponse(
      "INGEST_IN_PROGRESS",
      "Another ingest for this URL is already in progress",
      409,
    );
  }

  const upsertedVectorIds: string[] = [];

  try {
    const records = await embedChunks(env, chunks);

    for (const record of records) {
      const vectorId = buildChunkId(documentId, record.index);
      record.id = vectorId;
      record.vectorId = vectorId;
    }

    try {
      const vectorIds = await upsertVectors(
        env,
        documentId,
        payload.title,
        payload.url,
        records,
      );
      upsertedVectorIds.push(...vectorIds);
    } catch (writeError) {
      console.error("Vectorize upsert failed:", writeError);
      return errorResponse(
        "VECTORIZE_UPSERT_FAILED",
        "Failed to upsert vectors into Vectorize",
        502,
      );
    }

    try {
      await persistDocumentContent(
        env,
        documentId,
        payload.title,
        payload.url,
        records,
        isReingest,
      );
    } catch (writeError) {
      const rolledBack = await rollbackIngest(
        env,
        documentId,
        upsertedVectorIds,
        isReingest,
        previousVectorIds,
      );
      console.error("D1 write failed:", writeError);
      if (!rolledBack) {
        return errorResponse(
          "ROLLBACK_FAILED",
          "Failed to persist document and cleanup could not be completed",
          500,
        );
      }
      return errorResponse(
        "D1_WRITE_FAILED",
        "Failed to persist document content",
        500,
      );
    }

    const orphanedVectorIds = previousVectorIds.filter(
      (id) => !upsertedVectorIds.includes(id),
    );
    if (orphanedVectorIds.length > 0) {
      try {
        await deleteVectorsInBatches(env, orphanedVectorIds);
      } catch (cleanupError) {
        console.error("Failed to delete orphaned vectors after re-ingest:", cleanupError);
      }
    }

    try {
      await invalidateQueryCaches(env);
    } catch (cacheError) {
      console.error("Failed to invalidate query caches after ingest:", cacheError);
    }

    const response: IngestResponse = {
      documentId,
      chunkCount: records.length,
      vectorCount: records.length,
    };
    return jsonResponse(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown ingest error";
    console.error("Ingest embedding failed:", message);
    return errorResponse("EMBEDDING_FAILED", "Failed to generate embeddings", 502);
  } finally {
    await releaseIngestLock(env, payload.url);
  }
}
