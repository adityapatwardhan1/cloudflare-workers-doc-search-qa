import type { Env } from "../types";
import { errorResponse } from "./http";

function extractApiKey(request: Request): string | null {
  const headerKey = request.headers.get("X-API-Key");
  if (headerKey !== null && headerKey.trim().length > 0) {
    return headerKey.trim();
  }

  const authorization = request.headers.get("Authorization");
  if (authorization === null) {
    return null;
  }

  const match = authorization.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return null;
  }

  const token = match[1].trim();
  return token.length > 0 ? token : null;
}

export function requireAuth(request: Request, env: Env): Response | null {
  if (!env.API_KEY) {
    return null;
  }

  const providedKey = extractApiKey(request);
  if (providedKey === null || providedKey !== env.API_KEY) {
    return errorResponse("UNAUTHORIZED", "Valid API key required", 401);
  }

  return null;
}
