import { composeAnswer } from "../lib/composer";
import { errorResponse, jsonResponse } from "../lib/http";
import { hashQuery, searchContext, SearchError } from "../lib/search";
import { scheduleAuditLog } from "../lib/telemetry";
import type {
  Env,
  KvCacheEntry,
  QueryPayload,
  QueryResponse,
} from "../types";
import { KV_ANSWER_PREFIX, KV_CACHE_TTL_SECONDS } from "../types";

function buildAnswerCacheKey(queryHash: string): string {
  return `${KV_ANSWER_PREFIX}${queryHash}`;
}

function isValidAnswerCacheEntry(value: unknown): value is KvCacheEntry {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  if (typeof record.answer !== "string" || typeof record.timestamp !== "number") {
    return false;
  }
  if (!Array.isArray(record.citations)) {
    return false;
  }
  return record.citations.every((citation) => {
    if (typeof citation !== "object" || citation === null) {
      return false;
    }
    const item = citation as Record<string, unknown>;
    return (
      typeof item.quote === "string" &&
      typeof item.title === "string" &&
      typeof item.url === "string" &&
      typeof item.chunkId === "string"
    );
  });
}

async function getCachedAnswer(
  env: Env,
  queryHash: string,
): Promise<KvCacheEntry | null> {
  const cached = await env.KV_CACHE.get(buildAnswerCacheKey(queryHash), "json");
  if (!isValidAnswerCacheEntry(cached)) {
    return null;
  }
  return cached;
}

async function storeAnswerCache(
  env: Env,
  queryHash: string,
  answer: string,
  citations: KvCacheEntry["citations"],
): Promise<void> {
  const entry: KvCacheEntry = {
    answer,
    citations,
    timestamp: Date.now(),
  };
  await env.KV_CACHE.put(buildAnswerCacheKey(queryHash), JSON.stringify(entry), {
    expirationTtl: KV_CACHE_TTL_SECONDS,
  });
}

export async function answerQuery(
  payload: QueryPayload,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const totalStart = performance.now();
  const queryHash = await hashQuery(payload.question);

  const cacheStart = performance.now();
  const cachedAnswer = await getCachedAnswer(env, queryHash);
  const answerCacheMs = performance.now() - cacheStart;

  if (cachedAnswer !== null) {
    scheduleAuditLog(env, ctx, {
      queryHash,
      searchCacheHit: false,
      answerCacheHit: true,
      latencyCacheMs: answerCacheMs,
      latencyVectorMs: 0,
      latencyAiMs: 0,
      totalLatencyMs: performance.now() - totalStart,
    });

    const response: QueryResponse = {
      answer: cachedAnswer.answer,
      citations: cachedAnswer.citations,
      cacheHit: true,
      fallback: false,
    };
    return jsonResponse(response);
  }

  try {
    const searchResult = await searchContext(payload.question, env);

    const composeStart = performance.now();
    const composed = await composeAnswer(
      payload.question,
      searchResult.chunks,
      env,
    );
    const composeMs = performance.now() - composeStart;

    if (!composed.fallback) {
      await storeAnswerCache(
        env,
        queryHash,
        composed.answer,
        composed.citations,
      );
    }

    scheduleAuditLog(env, ctx, {
      queryHash,
      searchCacheHit: searchResult.cacheHit,
      answerCacheHit: false,
      latencyCacheMs: answerCacheMs + searchResult.timings.cacheMs,
      latencyVectorMs:
        searchResult.timings.embedMs +
        searchResult.timings.vectorMs +
        searchResult.timings.d1Ms,
      latencyAiMs: composeMs,
      totalLatencyMs: performance.now() - totalStart,
    });

    const response: QueryResponse = {
      answer: composed.answer,
      citations: composed.citations,
      cacheHit: false,
      fallback: composed.fallback,
    };
    return jsonResponse(response);
  } catch (error) {
    scheduleAuditLog(env, ctx, {
      queryHash,
      searchCacheHit: false,
      answerCacheHit: false,
      latencyCacheMs: answerCacheMs,
      latencyVectorMs: 0,
      latencyAiMs: 0,
      totalLatencyMs: performance.now() - totalStart,
    });

    if (error instanceof SearchError) {
      return errorResponse(error.code, error.message, error.status);
    }
    return errorResponse("QUERY_FAILED", "Unexpected query error", 500);
  }
}
