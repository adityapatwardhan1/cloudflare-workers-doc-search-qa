import type { Env } from "../types";
import { KV_ANSWER_PREFIX, KV_SEARCH_PREFIX } from "../types";

async function deleteKeysWithPrefix(env: Env, prefix: string): Promise<void> {
  let cursor: string | undefined;

  do {
    const list = await env.KV_CACHE.list({ prefix, cursor });
    await Promise.all(list.keys.map((key) => env.KV_CACHE.delete(key.name)));
    cursor = list.list_complete ? undefined : list.cursor;
  } while (cursor);
}

export async function invalidateQueryCaches(env: Env): Promise<void> {
  await Promise.all([
    deleteKeysWithPrefix(env, KV_SEARCH_PREFIX),
    deleteKeysWithPrefix(env, KV_ANSWER_PREFIX),
  ]);
}
