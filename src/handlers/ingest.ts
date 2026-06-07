import { chunkText } from "../lib/chunker";
import type {
  ApiError,
  Env,
  IngestPayload,
  IngestResponse,
  TextChunk,
} from "../types";
import {
  D1_BATCH_SIZE,
  EMBEDDING_BATCH_SIZE,
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  VECTOR_UPSERT_BATCH_SIZE,
} from "../types";

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

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

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...CORS_HEADERS,
    },
  });
}

function errorResponse(
  code: string,
  message: string,
  status: number,
  details?: string,
): Response {
  const body: ApiError = { error: message, code };
  if (details !== undefined) {
    body.details = details;
  }
  return jsonResponse(body, status);
}

function buildChunkId(documentId: string, chunkIndex: number): string {
  return `${documentId}:${chunkIndex}`;
}

async function deleteExistingByUrl(env: Env, url: string): Promise<void> {
  const existing = await env.DB_D1.prepare(
    "SELECT id FROM documents WHERE url = ?",
  )
    .bind(url)
    .first<{ id: string }>();

  if (!existing) {
    return;
  }

  const vectorRows = await env.DB_D1.prepare(
    "SELECT vector_id FROM chunks WHERE document_id = ?",
  )
    .bind(existing.id)
    .all<{ vector_id: string }>();

  const vectorIds = (vectorRows.results ?? []).map((row) => row.vector_id);
  if (vectorIds.length > 0) {
    await deleteVectorsInBatches(env, vectorIds);
  }

  await env.DB_D1.prepare("DELETE FROM chunks WHERE document_id = ?")
    .bind(existing.id)
    .run();
  await env.DB_D1.prepare("DELETE FROM documents WHERE id = ?")
    .bind(existing.id)
    .run();
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

async function embedChunks(env: Env, chunks: TextChunk[]): Promise<ChunkRecord[]> {
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

async function writeDocument(
  env: Env,
  documentId: string,
  title: string,
  url: string,
): Promise<void> {
  const now = new Date().toISOString();
  const result = await env.DB_D1.prepare(
    "INSERT INTO documents (id, title, url, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
  )
    .bind(documentId, title, url, now, now)
    .run();

  if (!result.success) {
    throw new Error("Failed to insert document row");
  }
}

async function writeChunks(
  env: Env,
  documentId: string,
  records: ChunkRecord[],
): Promise<void> {
  const now = new Date().toISOString();

  for (let i = 0; i < records.length; i += D1_BATCH_SIZE) {
    const batch = records.slice(i, i + D1_BATCH_SIZE);
    const statements = batch.map((record) => {
      const chunkId = buildChunkId(documentId, record.index);
      record.id = chunkId;
      record.vectorId = chunkId;
      return env.DB_D1.prepare(
        "INSERT INTO chunks (id, document_id, chunk_index, content, vector_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      ).bind(
        chunkId,
        documentId,
        record.index,
        record.content,
        chunkId,
        now,
      );
    });

    const results = await env.DB_D1.batch(statements);
    for (const result of results) {
      if (!result.success) {
        throw new Error("Failed to insert chunk row");
      }
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
): Promise<void> {
  try {
    if (vectorIds.length > 0) {
      await deleteVectorsInBatches(env, vectorIds);
    }
    await env.DB_D1.prepare("DELETE FROM chunks WHERE document_id = ?")
      .bind(documentId)
      .run();
    await env.DB_D1.prepare("DELETE FROM documents WHERE id = ?")
      .bind(documentId)
      .run();
  } catch (rollbackError) {
    console.error("Ingest rollback failed:", rollbackError);
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

  const documentId = crypto.randomUUID();
  const upsertedVectorIds: string[] = [];

  try {
    await deleteExistingByUrl(env, payload.url);

    const records = await embedChunks(env, chunks);

    for (const record of records) {
      const vectorId = buildChunkId(documentId, record.index);
      record.id = vectorId;
      record.vectorId = vectorId;
    }

    try {
      await writeDocument(env, documentId, payload.title, payload.url);
    } catch (writeError) {
      const message =
        writeError instanceof Error ? writeError.message : "Unknown write error";
      return errorResponse(
        "D1_WRITE_FAILED",
        "Failed to persist document metadata",
        500,
        message,
      );
    }

    try {
      await writeChunks(env, documentId, records);
    } catch (writeError) {
      await rollbackIngest(env, documentId, upsertedVectorIds);
      const message =
        writeError instanceof Error ? writeError.message : "Unknown write error";
      return errorResponse(
        "D1_WRITE_FAILED",
        "Failed to persist document chunks",
        500,
        message,
      );
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
      await rollbackIngest(env, documentId, upsertedVectorIds);
      const message =
        writeError instanceof Error ? writeError.message : "Unknown write error";
      return errorResponse(
        "VECTORIZE_UPSERT_FAILED",
        "Failed to upsert vectors into Vectorize",
        502,
        message,
      );
    }

    const response: IngestResponse = {
      documentId,
      chunkCount: records.length,
      vectorCount: records.length,
    };
    return jsonResponse(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown ingest error";
    return errorResponse("EMBEDDING_FAILED", "Failed to generate embeddings", 502, message);
  }
}
