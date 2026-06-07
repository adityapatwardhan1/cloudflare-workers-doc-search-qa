import type { Env } from "../types";

export interface AuditMetrics {
  queryHash: string;
  searchCacheHit: boolean;
  answerCacheHit: boolean;
  latencyCacheMs: number;
  latencyVectorMs: number;
  latencyAiMs: number;
  totalLatencyMs: number;
}

async function writeAuditLog(env: Env, metrics: AuditMetrics): Promise<void> {
  await env.DB_D1.prepare(
    `INSERT INTO audit_logs (
      id, query_hash, cache_hit, answer_cache_hit,
      latency_cache_ms, latency_vector_ms, latency_ai_ms,
      total_latency_ms, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      crypto.randomUUID(),
      metrics.queryHash,
      metrics.searchCacheHit ? 1 : 0,
      metrics.answerCacheHit ? 1 : 0,
      metrics.latencyCacheMs,
      metrics.latencyVectorMs,
      metrics.latencyAiMs,
      metrics.totalLatencyMs,
      new Date().toISOString(),
    )
    .run();
}

export function scheduleAuditLog(
  env: Env,
  ctx: ExecutionContext,
  metrics: AuditMetrics,
): void {
  ctx.waitUntil(
    writeAuditLog(env, metrics).catch((error) => {
      console.error("Failed to write audit log:", error);
    }),
  );
}
