export interface Env {
  AI: Ai;
  KV_CACHE: KVNamespace;
  DB_D1: D1Database;
  VECTOR_INDEX: Vectorize;
}

export const EMBEDDING_MODEL = "@cf/baai/bge-large-en-v1.5" as const;
export const GENERATION_MODEL_PRIMARY =
  "@cf/meta/llama-3.3-70b-instruct" as const;
export const GENERATION_MODEL_FALLBACK =
  "@cf/mistralai/mistral-small-3.1-24b-instruct" as const;
export const EMBEDDING_DIMENSIONS = 1024;
export const KV_CACHE_TTL_SECONDS = 3600;
export const KV_SEARCH_PREFIX = "search:" as const;
export const VECTOR_TOP_K = 5;

export const CHUNK_SIZE = 1000;
export const CHUNK_OVERLAP = 200;
export const CHUNK_STRIDE = CHUNK_SIZE - CHUNK_OVERLAP;
export const EMBEDDING_BATCH_SIZE = 8;
export const VECTOR_UPSERT_BATCH_SIZE = 50;
export const D1_BATCH_SIZE = 50;

export const VALIDATION_LIMITS = {
  MAX_BODY_BYTES: 512_000,
  MAX_TITLE_LENGTH: 512,
  MAX_URL_LENGTH: 2048,
  MAX_CONTENT_LENGTH: 500_000,
  MAX_QUESTION_LENGTH: 2_000,
} as const;

export interface TextChunk {
  index: number;
  content: string;
  startOffset: number;
  endOffset: number;
}

export interface IngestPayload {
  title: string;
  content: string;
  url: string;
}

export interface QueryPayload {
  question: string;
}

export interface IngestRequestBody {
  title: string;
  content: string;
  url: string;
}

export interface QueryRequestBody {
  question: string;
}

export interface Citation {
  quote: string;
  title: string;
  url: string;
  chunkId: string;
}

export interface KvCacheEntry {
  answer: string;
  citations: Citation[];
  timestamp: number;
}

export interface RetrievedChunk {
  chunkId: string;
  content: string;
  title: string;
  url: string;
  score: number;
  chunkIndex: number;
}

export interface KvSearchCacheEntry {
  chunks: RetrievedChunk[];
  timestamp: number;
}

export interface SearchTimings {
  cacheMs: number;
  embedMs: number;
  vectorMs: number;
  d1Ms: number;
  totalMs: number;
}

export interface SearchContextResult {
  chunks: RetrievedChunk[];
  cacheHit: boolean;
  queryHash: string;
  timings: SearchTimings;
}

export interface DocumentRow {
  readonly id: string;
  readonly title: string;
  readonly url: string;
  readonly created_at: string;
  readonly updated_at: string;
}

export interface ChunkRow {
  readonly id: string;
  readonly document_id: string;
  readonly chunk_index: number;
  readonly content: string;
  readonly vector_id: string;
  readonly created_at: string;
}

export interface AuditLogRow {
  readonly id: string;
  readonly query_hash: string;
  readonly cache_hit: number;
  readonly latency_cache_ms: number;
  readonly latency_vector_ms: number;
  readonly latency_ai_ms: number;
  readonly total_latency_ms: number;
  readonly created_at: string;
}

export interface ApiError {
  error: string;
  code: string;
  details?: string;
}

export interface HealthResponse {
  status: "ok";
  version: string;
}

export interface IngestResponse {
  documentId: string;
  chunkCount: number;
  vectorCount: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  cacheHit: boolean;
  fallback: boolean;
}

export type SanitizeResult<T> = T | ApiError;
