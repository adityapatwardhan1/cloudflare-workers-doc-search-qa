import type {
  Env,
  KvSearchCacheEntry,
  RetrievedChunk,
  SearchContextResult,
  SearchTimings,
} from "../types";
import {
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  KV_CACHE_TTL_SECONDS,
  KV_SEARCH_PREFIX,
  VECTOR_TOP_K,
} from "../types";

interface EmbeddingResult {
  data?: number[][];
}

interface ChunkRow {
  id: string;
  content: string;
  chunk_index: number;
  vector_id: string;
  title: string;
  url: string;
}

export class SearchError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export async function hashQuery(question: string): Promise<string> {
  const normalized = question.trim().toLowerCase();
  const encoded = new TextEncoder().encode(normalized);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function buildCacheKey(queryHash: string): string {
  return `${KV_SEARCH_PREFIX}${queryHash}`;
}

function isValidSearchCacheEntry(value: unknown): value is KvSearchCacheEntry {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  if (!Array.isArray(record.chunks) || typeof record.timestamp !== "number") {
    return false;
  }
  return record.chunks.every((chunk) => {
    if (typeof chunk !== "object" || chunk === null) {
      return false;
    }
    const item = chunk as Record<string, unknown>;
    return (
      typeof item.chunkId === "string" &&
      typeof item.content === "string" &&
      typeof item.title === "string" &&
      typeof item.url === "string" &&
      typeof item.score === "number" &&
      typeof item.chunkIndex === "number"
    );
  });
}

function emptyTimings(cacheMs: number, totalMs: number): SearchTimings {
  return {
    cacheMs,
    embedMs: 0,
    vectorMs: 0,
    d1Ms: 0,
    totalMs,
  };
}

function validateEmbedding(data: number[][] | undefined): number[] {
  if (!data || !Array.isArray(data) || data.length !== 1) {
    throw new SearchError(
      "EMBEDDING_FAILED",
      "Embedding model returned invalid data for query",
      502,
    );
  }
  const vector = data[0];
  if (!Array.isArray(vector) || vector.length !== EMBEDDING_DIMENSIONS) {
    throw new SearchError(
      "EMBEDDING_FAILED",
      `Query embedding has invalid dimensions (expected ${EMBEDDING_DIMENSIONS})`,
      502,
    );
  }
  return vector;
}

async function embedQuery(env: Env, question: string): Promise<number[]> {
  const result = (await env.AI.run(EMBEDDING_MODEL, {
    text: [question],
  })) as EmbeddingResult;
  return validateEmbedding(result.data);
}

async function queryVectorIndex(
  env: Env,
  vector: number[],
): Promise<VectorizeMatches> {
  try {
    return await env.VECTOR_INDEX.query(vector, {
      topK: VECTOR_TOP_K,
      returnMetadata: "none",
    });
  } catch (error) {
    console.error("Vectorize query failed:", error);
    throw new SearchError(
      "VECTORIZE_QUERY_FAILED",
      "Failed to query Vectorize index",
      502,
    );
  }
}

async function fetchChunksFromD1(
  env: Env,
  vectorIds: string[],
  scoreByVectorId: Map<string, number>,
): Promise<RetrievedChunk[]> {
  if (vectorIds.length === 0) {
    return [];
  }

  const placeholders = vectorIds.map(() => "?").join(", ");
  const sql = `
    SELECT c.id, c.content, c.chunk_index, c.vector_id,
           d.title, d.url
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.vector_id IN (${placeholders})
  `;

  try {
    const result = await env.DB_D1.prepare(sql)
      .bind(...vectorIds)
      .all<ChunkRow>();

    const rows = result.results ?? [];
    const chunks: RetrievedChunk[] = rows.map((row) => ({
      chunkId: row.id,
      content: row.content,
      title: row.title,
      url: row.url,
      score: scoreByVectorId.get(row.vector_id) ?? 0,
      chunkIndex: row.chunk_index,
    }));

    chunks.sort((a, b) => b.score - a.score);
    return chunks;
  } catch (error) {
    console.error("D1 chunk fetch failed:", error);
    throw new SearchError(
      "D1_FETCH_FAILED",
      "Failed to fetch chunk context from D1",
      500,
    );
  }
}

async function storeSearchCache(
  env: Env,
  cacheKey: string,
  chunks: RetrievedChunk[],
): Promise<void> {
  const entry: KvSearchCacheEntry = {
    chunks,
    timestamp: Date.now(),
  };
  await env.KV_CACHE.put(cacheKey, JSON.stringify(entry), {
    expirationTtl: KV_CACHE_TTL_SECONDS,
  });
}

export async function searchContext(
  question: string,
  env: Env,
): Promise<SearchContextResult> {
  const totalStart = performance.now();
  const queryHash = await hashQuery(question);
  const cacheKey = buildCacheKey(queryHash);

  const cacheStart = performance.now();
  const cached = await env.KV_CACHE.get(cacheKey, "json");
  const cacheMs = performance.now() - cacheStart;

  if (isValidSearchCacheEntry(cached)) {
    return {
      chunks: cached.chunks,
      cacheHit: true,
      queryHash,
      timings: emptyTimings(cacheMs, performance.now() - totalStart),
    };
  }

  const embedStart = performance.now();
  const queryVector = await embedQuery(env, question);
  const embedMs = performance.now() - embedStart;

  const vectorStart = performance.now();
  const matches = await queryVectorIndex(env, queryVector);
  const vectorMs = performance.now() - vectorStart;

  const scoreByVectorId = new Map<string, number>();
  const vectorIds: string[] = [];
  for (const match of matches.matches) {
    scoreByVectorId.set(match.id, match.score);
    vectorIds.push(match.id);
  }

  const d1Start = performance.now();
  const chunks = await fetchChunksFromD1(env, vectorIds, scoreByVectorId);
  const d1Ms = performance.now() - d1Start;

  await storeSearchCache(env, cacheKey, chunks);

  return {
    chunks,
    cacheHit: false,
    queryHash,
    timings: {
      cacheMs,
      embedMs,
      vectorMs,
      d1Ms,
      totalMs: performance.now() - totalStart,
    },
  };
}
