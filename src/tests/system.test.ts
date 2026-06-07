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
import { EMBEDDING_DIMENSIONS } from "../types";

async function readJson<T>(response: Response): Promise<T> {
  return response.json() as Promise<T>;
}

const SAMPLE_DOC = {
  title: "Auth Guide",
  content:
    "JWT rotation requires updating the signing key every 90 days. Configure rotation in wrangler.toml under the secrets binding.",
  url: "https://docs.example.com/auth",
};

const SAMPLE_CHUNK_ID = "doc-1:0";
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

function createMockEnv(options: {
  vectorizeThrows?: boolean;
  aiResponses?: Record<string, unknown>;
  kvStore?: Map<string, string>;
  d1Chunks?: RetrievedChunk[];
  auditLogs?: Array<Record<string, unknown>>;
} = {}): Env {
  const kvData = options.kvStore ?? new Map<string, string>();
  const auditLogs = options.auditLogs ?? [];

  const defaultAiResponses: Record<string, unknown> = {
    "@cf/baai/bge-large-en-v1.5": { data: [SAMPLE_VECTOR] },
    "@cf/meta/llama-3.3-70b-instruct": {
      response: JSON.stringify({
        answer: "Configure JWT rotation in wrangler.toml under the secrets binding.",
        citations: [
          {
            quote: SAMPLE_DOC.content,
            title: SAMPLE_DOC.title,
            url: SAMPLE_DOC.url,
            chunkId: SAMPLE_CHUNK_ID,
          },
        ],
      }),
    },
    ...options.aiResponses,
  };

  const d1Chunks =
    options.d1Chunks ??
    [
      {
        chunkId: SAMPLE_CHUNK_ID,
        content: SAMPLE_DOC.content,
        title: SAMPLE_DOC.title,
        url: SAMPLE_DOC.url,
        score: 0.91,
        chunkIndex: 0,
      },
    ];

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
                latency_cache_ms: binds[3],
                latency_vector_ms: binds[4],
                latency_ai_ms: binds[5],
                total_latency_ms: binds[6],
                created_at: binds[7],
              });
            }
            return { success: true };
          },
          async first<T>() {
            if (sql.includes("SELECT id FROM documents WHERE url")) {
              return null as T;
            }
            return null as T;
          },
          async all<T>() {
            if (sql.includes("FROM chunks c")) {
              return {
                results: d1Chunks.map((chunk) => ({
                  id: chunk.chunkId,
                  content: chunk.content,
                  chunk_index: chunk.chunkIndex,
                  vector_id: chunk.chunkId,
                  title: chunk.title,
                  url: chunk.url,
                })),
              } as T;
            }
            if (sql.includes("SELECT vector_id FROM chunks")) {
              return { results: [] } as T;
            }
            return { results: [] } as T;
          },
        }),
      };
    },
    async batch(statements: Array<{ run: () => Promise<{ success: boolean }> }>) {
      const results = [];
      for (const statement of statements) {
        await statement.run();
        results.push({ success: true });
      }
      return results;
    },
  };

  const mockEnv = {
    AI: {
      run: vi.fn(async (model: string, inputs?: { text?: string[] }) => {
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
      delete: vi.fn(),
      list: vi.fn(),
      getWithMetadata: vi.fn(),
    },
    DB_D1: db,
    VECTOR_INDEX: {
      query: vi.fn(async () => {
        if (options.vectorizeThrows) {
          throw new Error("Vectorize internal error");
        }
        return {
          matches: [
            {
              id: SAMPLE_CHUNK_ID,
              score: 0.91,
            },
          ],
          count: 1,
        };
      }),
      upsert: vi.fn(async () => ({ mutationId: "mock-mutation" })),
      insert: vi.fn(),
      deleteByIds: vi.fn(),
      getByIds: vi.fn(),
      queryById: vi.fn(),
    },
  };

  return mockEnv as unknown as Env;
}

async function postJson(
  path: string,
  body: unknown,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  return worker.fetch(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    env,
    ctx,
  );
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

  it("runs ingest then query end-to-end with mocked bindings", async () => {
    const env = createMockEnv();
    const ctx = createExecutionContext();

    const ingestResponse = await postJson("/ingest", SAMPLE_DOC, env, ctx);
    const ingestBody = await readJson<IngestResponse>(ingestResponse);
    expect(ingestResponse.status).toBe(200);
    expect(ingestBody.chunkCount).toBeGreaterThan(0);
    expect(env.VECTOR_INDEX.upsert).toHaveBeenCalled();

    const queryResponse = await postJson(
      "/query",
      { question: "How do I configure JWT rotation?" },
      env,
      ctx,
    );
    const queryBody = await readJson<QueryResponse>(queryResponse);
    await flushWaitUntil(ctx);

    expect(queryResponse.status).toBe(200);
    expect(queryBody.cacheHit).toBe(false);
    expect(queryBody.fallback).toBe(false);
    expect(queryBody.citations.length).toBeGreaterThan(0);
    expect(env.VECTOR_INDEX.query).toHaveBeenCalled();
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

  it("writes audit log telemetry asynchronously on query", async () => {
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
    expect(auditLogs[0].query_hash).toMatch(/^[a-f0-9]{64}$/);
    expect(auditLogs[0].total_latency_ms).toBeGreaterThanOrEqual(0);
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
        chunkId: SAMPLE_CHUNK_ID,
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
});
