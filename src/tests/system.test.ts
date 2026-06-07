import { describe, expect, it, vi } from "vitest";
import worker from "../index";
import { chunkText } from "../lib/chunker";
import { composeAnswer } from "../lib/composer";
import { hashQuery } from "../lib/search";
import type {
  ApiError,
  Env,
  IngestResponse,
  QueryResponse,
  RetrievedChunk,
} from "../types";
import { EMBEDDING_DIMENSIONS, KV_ANSWER_PREFIX, KV_SEARCH_PREFIX } from "../types";

async function readJson<T>(response: Response): Promise<T> {
  return response.json() as Promise<T>;
}

const SAMPLE_DOC = {
  title: "Auth Guide",
  content:
    "JWT rotation requires updating the signing key every 90 days. Configure rotation in wrangler.toml under the secrets binding.",
  url: "https://docs.example.com/auth",
};

const SAMPLE_VECTOR = Array.from({ length: EMBEDDING_DIMENSIONS }, (_, i) => i * 0.001);

function createExecutionContext(): ExecutionContext {
  const waitUntilTasks: Promise<unknown>[] = [];
  return {
    waitUntil(promise: Promise<unknown>) {
      waitUntilTasks.push(promise);
    },
    passThroughOnException() {},
    // @ts-expect-error test helper for awaiting background audit writes
    _waitUntilTasks: waitUntilTasks,
  };
}

async function flushWaitUntil(ctx: ExecutionContext): Promise<void> {
  const tasks = (ctx as ExecutionContext & { _waitUntilTasks?: Promise<unknown>[] })
    ._waitUntilTasks;
  if (tasks && tasks.length > 0) {
    await Promise.all(tasks);
  }
}

interface StoredDocument {
  id: string;
  title: string;
  url: string;
}

interface StoredChunk {
  id: string;
  documentId: string;
  chunkIndex: number;
  content: string;
  vectorId: string;
}

interface MockState {
  documents: Map<string, StoredDocument>;
  documentsByUrl: Map<string, string>;
  chunks: Map<string, StoredChunk>;
  ingestLocks: Set<string>;
  upsertedVectorIds: string[];
}

function createMockEnv(options: {
  vectorizeThrows?: boolean;
  embedThrows?: boolean;
  aiResponses?: Record<string, unknown>;
  kvStore?: Map<string, string>;
  auditLogs?: Array<Record<string, unknown>>;
  apiKey?: string;
  state?: MockState;
} = {}): Env {
  const kvData = options.kvStore ?? new Map<string, string>();
  const auditLogs = options.auditLogs ?? [];
  const state: MockState = options.state ?? {
    documents: new Map(),
    documentsByUrl: new Map(),
    chunks: new Map(),
    ingestLocks: new Set(),
    upsertedVectorIds: [],
  };

  const defaultAiResponses: Record<string, unknown> = {
    "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
    "@cf/meta/llama-3.3-70b-instruct": {
      response: JSON.stringify({
        answer: "Configure JWT rotation in wrangler.toml under the secrets binding.",
        citations: [],
      }),
    },
    ...options.aiResponses,
  };

  const db = {
    prepare(sql: string) {
      return {
        bind: (...binds: unknown[]) => ({
          async run() {
            if (sql.includes("INSERT INTO audit_logs")) {
              auditLogs.push({
                id: binds[0],
                query_hash: binds[1],
                cache_hit: binds[2],
                answer_cache_hit: binds[3],
                latency_cache_ms: binds[4],
                latency_vector_ms: binds[5],
                latency_ai_ms: binds[6],
                total_latency_ms: binds[7],
                created_at: binds[8],
              });
              return { success: true };
            }

            if (sql.includes("INSERT INTO ingest_locks")) {
              const url = binds[0] as string;
              if (state.ingestLocks.has(url)) {
                return { success: false };
              }
              state.ingestLocks.add(url);
              return { success: true };
            }

            if (sql.includes("DELETE FROM ingest_locks")) {
              state.ingestLocks.delete(binds[0] as string);
              return { success: true };
            }

            if (sql.includes("INSERT INTO documents")) {
              const [id, title, url] = binds as [string, string, string];
              state.documents.set(id, { id, title, url });
              state.documentsByUrl.set(url, id);
              return { success: true };
            }

            if (sql.includes("UPDATE documents SET title")) {
              const [title, , id] = binds as [string, string, string];
              const doc = state.documents.get(id);
              if (doc) {
                doc.title = title;
              }
              return { success: true };
            }

            if (sql.includes("DELETE FROM chunks WHERE document_id")) {
              for (const [chunkId, chunk] of state.chunks.entries()) {
                if (chunk.documentId === binds[0]) {
                  state.chunks.delete(chunkId);
                }
              }
              return { success: true };
            }

            if (sql.includes("DELETE FROM documents WHERE id")) {
              const docId = binds[0] as string;
              const doc = state.documents.get(docId);
              if (doc) {
                state.documentsByUrl.delete(doc.url);
                state.documents.delete(docId);
              }
              return { success: true };
            }

            if (sql.includes("INSERT INTO chunks")) {
              const [id, documentId, chunkIndex, content, vectorId] = binds as [
                string,
                string,
                number,
                string,
                string,
              ];
              state.chunks.set(id, {
                id,
                documentId,
                chunkIndex,
                content,
                vectorId,
              });
              return { success: true };
            }

            return { success: true };
          },
          async first<T>() {
            if (sql.includes("SELECT id FROM documents WHERE url")) {
              const url = binds[0] as string;
              const docId = state.documentsByUrl.get(url);
              if (!docId) {
                return null as T;
              }
              return { id: docId } as T;
            }
            return null as T;
          },
          async all<T>() {
            if (sql.includes("FROM chunks c")) {
              const vectorIds = binds as string[];
              const rows = [...state.chunks.values()]
                .filter((chunk) => vectorIds.includes(chunk.vectorId))
                .map((chunk) => {
                  const doc = state.documents.get(chunk.documentId);
                  return {
                    id: chunk.id,
                    content: chunk.content,
                    chunk_index: chunk.chunkIndex,
                    vector_id: chunk.vectorId,
                    title: doc?.title ?? "",
                    url: doc?.url ?? "",
                  };
                });
              return { results: rows } as T;
            }

            if (sql.includes("SELECT vector_id FROM chunks")) {
              const documentId = binds[0] as string;
              const rows = [...state.chunks.values()]
                .filter((chunk) => chunk.documentId === documentId)
                .map((chunk) => ({ vector_id: chunk.vectorId }));
              return { results: rows } as T;
            }

            return { results: [] } as T;
          },
        }),
      };
    },
    async batch(statements: Array<{ run: () => Promise<{ success: boolean }> }>) {
      const results = [];
      for (const statement of statements) {
        const result = await statement.run();
        results.push(result);
        if (!result.success) {
          break;
        }
      }
      return results;
    },
  };

  const mockEnv = {
    AI: {
      run: vi.fn(async (model: string, inputs?: { text?: string[] }) => {
        if (options.embedThrows && model === "@cf/baai/bge-large-en-v1.5") {
          throw new Error("Embedding service unavailable");
        }

        const response = defaultAiResponses[model];
        if (!response) {
          throw new Error(`Unexpected AI model: ${model}`);
        }
        if (
          model === "@cf/baai/bge-large-en-v1.5" &&
          inputs?.text &&
          Array.isArray((response as { data: number[][] }).data)
        ) {
          const vectors = inputs.text.map((_, index) =>
            (response as { data: number[][] }).data[index] ?? SAMPLE_VECTOR,
          );
          return { data: vectors };
        }
        return response;
      }),
    },
    KV_CACHE: {
      get: vi.fn(async (key: string, type?: string) => {
        const value = kvData.get(key);
        if (!value) {
          return null;
        }
        if (type === "json") {
          return JSON.parse(value);
        }
        return value;
      }),
      put: vi.fn(async (key: string, value: string) => {
        kvData.set(key, value);
      }),
      delete: vi.fn(async (key: string) => {
        kvData.delete(key);
      }),
      list: vi.fn(async ({ prefix }: { prefix: string }) => {
        const keys = [...kvData.keys()]
          .filter((key) => key.startsWith(prefix))
          .map((name) => ({ name }));
        return { keys, list_complete: true };
      }),
      getWithMetadata: vi.fn(),
    },
    DB_D1: db,
    VECTOR_INDEX: {
      query: vi.fn(async () => {
        if (options.vectorizeThrows) {
          throw new Error("Vectorize internal error");
        }

        const topChunk = [...state.chunks.values()][0];
        if (!topChunk) {
          return { matches: [], count: 0 };
        }

        return {
          matches: [
            {
              id: topChunk.vectorId,
              score: 0.91,
            },
          ],
          count: 1,
        };
      }),
      upsert: vi.fn(async (vectors: Array<{ id: string }>) => {
        for (const vector of vectors) {
          state.upsertedVectorIds.push(vector.id);
        }
        return { mutationId: "mock-mutation" };
      }),
      insert: vi.fn(),
      deleteByIds: vi.fn(),
      getByIds: vi.fn(),
      queryById: vi.fn(),
    },
    API_KEY: options.apiKey,
    _state: state,
  };

  return mockEnv as unknown as Env;
}

function getState(env: Env): MockState {
  return (env as Env & { _state: MockState })._state;
}

async function postJson(
  path: string,
  body: unknown,
  env: Env,
  ctx: ExecutionContext,
  headers: Record<string, string> = {},
): Promise<Response> {
  return worker.fetch(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
    }),
    env,
    ctx,
  );
}

function citationForChunk(chunk: StoredChunk, env: Env): {
  quote: string;
  title: string;
  url: string;
  chunkId: string;
} {
  const state = getState(env);
  const doc = state.documents.get(chunk.documentId);
  return {
    quote: chunk.content,
    title: doc?.title ?? "",
    url: doc?.url ?? "",
    chunkId: chunk.id,
  };
}

describe("system reliability harness", () => {
  it("health endpoint returns ok", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();
    const response = await worker.fetch(
      new Request("http://localhost/health"),
      env,
      ctx,
    );
    const body = await readJson<{ status: string; version: string }>(response);
    expect(response.status).toBe(200);
    expect(body).toEqual({ status: "ok", version: "1.0.0" });
  });

  it("rejects corrupt JSON payloads with MALFORMED_JSON", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();

    const ingestResponse = await worker.fetch(
      new Request("http://localhost/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{not-valid-json",
      }),
      env,
      ctx,
    );
    const ingestBody = await readJson<ApiError>(ingestResponse);
    expect(ingestResponse.status).toBe(400);
    expect(ingestBody.code).toBe("MALFORMED_JSON");

    const queryResponse = await worker.fetch(
      new Request("http://localhost/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{broken",
      }),
      env,
      ctx,
    );
    const queryBody = await readJson<ApiError>(queryResponse);
    expect(queryResponse.status).toBe(400);
    expect(queryBody.code).toBe("MALFORMED_JSON");
  });

  it("rejects invalid ingest and query field shapes", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();

    const ingestResponse = await postJson("/ingest", {}, env, ctx);
    const ingestBody = await readJson<ApiError>(ingestResponse);
    expect(ingestResponse.status).toBe(400);
    expect(ingestBody.code).toBe("INVALID_TITLE");

    const queryResponse = await postJson("/query", { question: "" }, env, ctx);
    const queryBody = await readJson<ApiError>(queryResponse);
    expect(queryResponse.status).toBe(400);
    expect(queryBody.code).toBe("INVALID_QUESTION");
  });

  it("runs ingest then query end-to-end with consistent chunk IDs", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();

    const ingestResponse = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    const ingestBody = await readJson<IngestResponse>(ingestResponse);
    expect(ingestResponse.status).toBe(200);
    expect(ingestBody.chunkCount).toBeGreaterThan(0);
    expect(env.VECTOR_INDEX.upsert).toHaveBeenCalled();

    const state = getState(env);
    const storedChunk = [...state.chunks.values()][0];
    expect(storedChunk.id).toBe(`${ingestBody.documentId}:0`);

    const envWithCitations = createMockEnv({
      state,
      aiResponses: {
        "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
        "@cf/meta/llama-3.3-70b-instruct": {
          response: JSON.stringify({
            answer: "Configure JWT rotation in wrangler.toml under the secrets binding.",
            citations: [citationForChunk(storedChunk, env)],
          }),
        },
      },
    });

    const queryResponse = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      envWithCitations,
      ctx,
    );
    const queryBody = await readJson<QueryResponse>(queryResponse);
    await flushWaitUntil(ctx);

    expect(queryResponse.status).toBe(200);
    expect(queryBody.cacheHit).toBe(false);
    expect(queryBody.fallback).toBe(false);
    expect(queryBody.citations[0].chunkId).toBe(storedChunk.id);
    expect(envWithCitations.VECTOR_INDEX.query).toHaveBeenCalled();
  });

  it("returns VECTORIZE_QUERY_FAILED when Vectorize throws", async () => {
    const env = createMockEnv({ vectorizeThrows: true });
    const ctx = createExecutionContext();

    const response = await postJson(
      "/search",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    const body = await readJson<ApiError>(response);

    expect(response.status).toBe(502);
    expect(body.code).toBe("VECTORIZE_QUERY_FAILED");
    expect(body.details).toBeUndefined();
  });

  it("serves cached query answers without calling Vectorize again", async () => {
    const kvStore = new Map<string, string>();
    const env = createMockEnv({ kvStore });
    const ctx = createExecutionContext();

    const firstResponse = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    expect(firstResponse.status).toBe(200);
    await flushWaitUntil(ctx);

    vi.mocked(env.VECTOR_INDEX.query).mockClear();

    const secondResponse = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    const secondBody = await readJson<QueryResponse>(secondResponse);

    expect(secondResponse.status).toBe(200);
    expect(secondBody.cacheHit).toBe(true);
    expect(env.VECTOR_INDEX.query).not.toHaveBeenCalled();
  });

  it("writes audit log telemetry with separate answer cache hit flag", async () => {
    const auditLogs: Array<Record<string, unknown>> = [];
    const env = createMockEnv({ auditLogs });
    const ctx = createExecutionContext();

    const response = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    expect(response.status).toBe(200);
    await flushWaitUntil(ctx);

    expect(auditLogs.length).toBe(1);
    expect(auditLogs[0].cache_hit).toBe(0);
    expect(auditLogs[0].answer_cache_hit).toBe(0);
    expect(auditLogs[0].query_hash).toMatch(/^[a-f0-9]{64}$/);
  });

  it("chunker produces deterministic overlapping chunks", () => {
    const shortChunks = chunkText("short document");
    expect(shortChunks).toHaveLength(1);

    const longText = "word ".repeat(600).trim();
    const longChunks = chunkText(longText);
    expect(longChunks.length).toBeGreaterThan(1);
    expect(longChunks[1].startOffset).toBeLessThan(longChunks[0].endOffset);
  });

  it("hashQuery is deterministic for equivalent questions", async () => {
    const first = await hashQuery("  How Do I Rotate JWT?  ");
    const second = await hashQuery("how do i rotate jwt?");
    expect(first).toBe(second);
    expect(first).toMatch(/^[a-f0-9]{64}$/);
  });

  it("composer falls back extractively when AI output is invalid", async () => {
    const env = createMockEnv({
      aiResponses: {
        "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
        "@cf/meta/llama-3.3-70b-instruct": { response: "not valid json" },
        "@cf/mistralai/mistral-small-3.1-24b-instruct": { response: "still bad" },
      },
    });

    const chunks: RetrievedChunk[] = [
      {
        chunkId: "doc-1:0",
        content: SAMPLE_DOC.content,
        title: SAMPLE_DOC.title,
        url: SAMPLE_DOC.url,
        score: 0.9,
        chunkIndex: 0,
      },
    ];

    const result = await composeAnswer("How do I rotate JWT?", chunks, env);
    expect(result.fallback).toBe(true);
    expect(result.citations.length).toBe(1);
    expect(result.answer).toContain("AI synthesis is temporarily unavailable");
  });

  it("composer falls back when all citations fail validation", async () => {
    const env = createMockEnv({
      aiResponses: {
        "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
        "@cf/meta/llama-3.3-70b-instruct": {
          response: JSON.stringify({
            answer: "Some answer without valid citations.",
            citations: [
              {
                quote: "fabricated quote",
                title: SAMPLE_DOC.title,
                url: SAMPLE_DOC.url,
                chunkId: "doc-1:0",
              },
            ],
          }),
        },
        "@cf/mistralai/mistral-small-3.1-24b-instruct": {
          response: JSON.stringify({
            answer: "Still no valid citations.",
            citations: [
              {
                quote: "also fabricated",
                title: SAMPLE_DOC.title,
                url: SAMPLE_DOC.url,
                chunkId: "doc-1:0",
              },
            ],
          }),
        },
      },
    });

    const chunks: RetrievedChunk[] = [
      {
        chunkId: "doc-1:0",
        content: SAMPLE_DOC.content,
        title: SAMPLE_DOC.title,
        url: SAMPLE_DOC.url,
        score: 0.9,
        chunkIndex: 0,
      },
    ];

    const result = await composeAnswer("How do I rotate JWT?", chunks, env);
    expect(result.fallback).toBe(true);
    expect(result.answer).toContain("AI synthesis is temporarily unavailable");
  });

  it("does not cache fallback answers", async () => {
    const kvStore = new Map<string, string>();
    const state: MockState = {
      documents: new Map(),
      documentsByUrl: new Map(),
      chunks: new Map(),
      ingestLocks: new Set(),
      upsertedVectorIds: [],
    };
    const env = createMockEnv({
      kvStore,
      state,
      aiResponses: {
        "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
        "@cf/meta/llama-3.3-70b-instruct": { response: "not valid json" },
        "@cf/mistralai/mistral-small-3.1-24b-instruct": { response: "still bad" },
      },
    });
    const ctx = createExecutionContext();

    const ingestResponse = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    expect(ingestResponse.status).toBe(200);

    const response = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    const body = await readJson<QueryResponse>(response);

    expect(response.status).toBe(200);
    expect(body.fallback).toBe(true);

    const queryHash = await hashQuery("How do I configure JWT rotation?");
    expect(kvStore.has(`${KV_ANSWER_PREFIX}${queryHash}`)).toBe(false);
  });

  it("preserves existing document when embedding fails on re-ingest", async () => {
    const state: MockState = {
      documents: new Map(),
      documentsByUrl: new Map(),
      chunks: new Map(),
      ingestLocks: new Set(),
      upsertedVectorIds: [],
    };

    const env = createMockEnv({ state });
    const ctx = createExecutionContext();

    const first = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    expect(first.status).toBe(200);
    const firstBody = await readJson<IngestResponse>(first);
    const chunkCountBefore = state.chunks.size;

    const failingEnv = createMockEnv({ state, embedThrows: true });
    const second = await postJson(
      "/ingest",
      { ...SAMPLE_DOC, title: "Updated Auth Guide" },
      failingEnv,
      ctx,
    );
    const secondBody = await readJson<ApiError>(second);

    expect(second.status).toBe(502);
    expect(secondBody.code).toBe("EMBEDDING_FAILED");
    expect(state.documentsByUrl.get(SAMPLE_DOC.url)).toBe(firstBody.documentId);
    expect(state.chunks.size).toBe(chunkCountBefore);
  });

  it("re-ingests same URL and invalidates KV caches", async () => {
    const kvStore = new Map<string, string>();
    const state: MockState = {
      documents: new Map(),
      documentsByUrl: new Map(),
      chunks: new Map(),
      ingestLocks: new Set(),
      upsertedVectorIds: [],
    };
    const env = createMockEnv({ state, kvStore });
    const ctx = createExecutionContext();

    const first = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    expect(first.status).toBe(200);
    const firstBody = await readJson<IngestResponse>(first);

    const queryHash = await hashQuery("How do I configure JWT rotation?");
    kvStore.set(`${KV_SEARCH_PREFIX}${queryHash}`, JSON.stringify({ chunks: [], timestamp: 1 }));
    kvStore.set(`${KV_ANSWER_PREFIX}${queryHash}`, JSON.stringify({
      answer: "stale",
      citations: [],
      timestamp: 1,
    }));

    const second = await postJson(
      "/ingest",
      { ...SAMPLE_DOC, title: "Updated Auth Guide" },
      env,
      ctx,
    );
    const secondBody = await readJson<IngestResponse>(second);

    expect(second.status).toBe(200);
    expect(secondBody.documentId).toBe(firstBody.documentId);
    expect(state.documents.get(firstBody.documentId)?.title).toBe("Updated Auth Guide");
    expect(kvStore.has(`${KV_SEARCH_PREFIX}${queryHash}`)).toBe(false);
    expect(kvStore.has(`${KV_ANSWER_PREFIX}${queryHash}`)).toBe(false);
  });

  it("rejects oversized bodies without Content-Length", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();
    const oversized = JSON.stringify({ title: "t", url: "https://example.com", content: "x".repeat(600_000) });

    const response = await worker.fetch(
      new Request("http://localhost/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: oversized,
      }),
      env,
      ctx,
    );
    const body = await readJson<ApiError>(response);

    expect(response.status).toBe(400);
    expect(body.code).toBe("BODY_TOO_LARGE");
  });

  it("rejects concurrent ingest for the same URL", async () => {
    const state: MockState = {
      documents: new Map(),
      documentsByUrl: new Map(),
      chunks: new Map(),
      ingestLocks: new Set(["https://docs.example.com/auth"]),
      upsertedVectorIds: [],
    };
    const env = createMockEnv({ state });
    const ctx = createExecutionContext();

    const response = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    const body = await readJson<ApiError>(response);

    expect(response.status).toBe(409);
    expect(body.code).toBe("INGEST_IN_PROGRESS");
  });

  it("requires API key when configured", async () => {
    const env = createMockEnv({ apiKey: "secret-key" });
    const ctx = createExecutionContext();

    const unauthorized = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    expect(unauthorized.status).toBe(401);

    const authorized = await postJson("/ingest", SAMPLE_DOC, env, ctx, {
      Authorization: "Bearer secret-key",
    });
    expect(authorized.status).toBe(200);
  });
});
