import { ingestDocument } from "./handlers/ingest";
import { answerQuery } from "./handlers/query";
import { requireAuth } from "./lib/auth";
import { errorResponse, jsonResponse, CORS_HEADERS } from "./lib/http";
import { searchContext, SearchError } from "./lib/search";
import type {
  ApiError,
  Env,
  HealthResponse,
  IngestPayload,
  QueryPayload,
  SanitizeResult,
} from "./types";
import { VALIDATION_LIMITS } from "./types";

const WORKER_VERSION = "1.0.0";

export function isApiError(value: unknown): value is ApiError {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return typeof record.error === "string" && typeof record.code === "string";
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function truncateField(value: string, maxLength: number): string | null {
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > maxLength) {
    return null;
  }
  return trimmed;
}

function isValidHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function validateIngestPayload(raw: unknown): SanitizeResult<IngestPayload> {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return {
      error: "Request body must be a JSON object",
      code: "INVALID_BODY_SHAPE",
    };
  }

  const body = raw as Record<string, unknown>;

  if (!isNonEmptyString(body.title)) {
    return { error: "Field 'title' must be a non-empty string", code: "INVALID_TITLE" };
  }
  if (!isNonEmptyString(body.content)) {
    return { error: "Field 'content' must be a non-empty string", code: "INVALID_CONTENT" };
  }
  if (!isNonEmptyString(body.url)) {
    return { error: "Field 'url' must be a non-empty string", code: "INVALID_URL" };
  }

  const title = truncateField(body.title, VALIDATION_LIMITS.MAX_TITLE_LENGTH);
  if (title === null) {
    return {
      error: `Field 'title' must be between 1 and ${VALIDATION_LIMITS.MAX_TITLE_LENGTH} characters`,
      code: "TITLE_LENGTH_EXCEEDED",
    };
  }

  const content = truncateField(body.content, VALIDATION_LIMITS.MAX_CONTENT_LENGTH);
  if (content === null) {
    return {
      error: `Field 'content' must be between 1 and ${VALIDATION_LIMITS.MAX_CONTENT_LENGTH} characters`,
      code: "CONTENT_LENGTH_EXCEEDED",
    };
  }

  const url = truncateField(body.url, VALIDATION_LIMITS.MAX_URL_LENGTH);
  if (url === null) {
    return {
      error: `Field 'url' must be between 1 and ${VALIDATION_LIMITS.MAX_URL_LENGTH} characters`,
      code: "URL_LENGTH_EXCEEDED",
    };
  }

  if (!isValidHttpUrl(url)) {
    return {
      error: "Field 'url' must be a valid http or https URL",
      code: "INVALID_URL_FORMAT",
    };
  }

  return { title, content, url };
}

export function validateQueryPayload(raw: unknown): SanitizeResult<QueryPayload> {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return {
      error: "Request body must be a JSON object",
      code: "INVALID_BODY_SHAPE",
    };
  }

  const body = raw as Record<string, unknown>;

  if (!isNonEmptyString(body.question)) {
    return {
      error: "Field 'question' must be a non-empty string",
      code: "INVALID_QUESTION",
    };
  }

  const question = truncateField(body.question, VALIDATION_LIMITS.MAX_QUESTION_LENGTH);
  if (question === null) {
    return {
      error: `Field 'question' must be between 1 and ${VALIDATION_LIMITS.MAX_QUESTION_LENGTH} characters`,
      code: "QUESTION_LENGTH_EXCEEDED",
    };
  }

  return { question };
}

function concatUint8Arrays(chunks: Uint8Array[]): Uint8Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

async function readJsonBody(request: Request): Promise<unknown | ApiError> {
  const contentLength = request.headers.get("Content-Length");
  if (contentLength !== null) {
    const bytes = Number(contentLength);
    if (!Number.isFinite(bytes) || bytes < 0) {
      return { error: "Invalid Content-Length header", code: "INVALID_CONTENT_LENGTH" };
    }
    if (bytes > VALIDATION_LIMITS.MAX_BODY_BYTES) {
      return {
        error: `Request body exceeds maximum size of ${VALIDATION_LIMITS.MAX_BODY_BYTES} bytes`,
        code: "BODY_TOO_LARGE",
      };
    }
  }

  if (!request.body) {
    return {};
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    totalBytes += value.byteLength;
    if (totalBytes > VALIDATION_LIMITS.MAX_BODY_BYTES) {
      return {
        error: `Request body exceeds maximum size of ${VALIDATION_LIMITS.MAX_BODY_BYTES} bytes`,
        code: "BODY_TOO_LARGE",
      };
    }
    chunks.push(value);
  }

  if (totalBytes === 0) {
    return {};
  }

  const bodyText = new TextDecoder().decode(concatUint8Arrays(chunks));

  try {
    return JSON.parse(bodyText);
  } catch {
    return { error: "Request body must be valid JSON", code: "MALFORMED_JSON" };
  }
}

function handleOptions(): Response {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

function handleHealth(): Response {
  const body: HealthResponse = { status: "ok", version: WORKER_VERSION };
  return jsonResponse(body);
}

async function handleIngest(request: Request, env: Env): Promise<Response> {
  const authError = requireAuth(request, env);
  if (authError !== null) {
    return authError;
  }

  const raw = await readJsonBody(request);
  if (isApiError(raw)) {
    return errorResponse(raw.code, raw.error, 400);
  }

  const payload = validateIngestPayload(raw);
  if (isApiError(payload)) {
    return errorResponse(payload.code, payload.error, 400);
  }

  return ingestDocument(payload, env);
}

async function handleSearch(request: Request, env: Env): Promise<Response> {
  const authError = requireAuth(request, env);
  if (authError !== null) {
    return authError;
  }

  const raw = await readJsonBody(request);
  if (isApiError(raw)) {
    return errorResponse(raw.code, raw.error, 400);
  }

  const payload = validateQueryPayload(raw);
  if (isApiError(payload)) {
    return errorResponse(payload.code, payload.error, 400);
  }

  try {
    const result = await searchContext(payload.question, env);
    return jsonResponse(result);
  } catch (error) {
    if (error instanceof SearchError) {
      return errorResponse(error.code, error.message, error.status);
    }
    return errorResponse("SEARCH_FAILED", "Unexpected search error", 500);
  }
}

async function handleQuery(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const authError = requireAuth(request, env);
  if (authError !== null) {
    return authError;
  }

  const raw = await readJsonBody(request);
  if (isApiError(raw)) {
    return errorResponse(raw.code, raw.error, 400);
  }

  const payload = validateQueryPayload(raw);
  if (isApiError(payload)) {
    return errorResponse(payload.code, payload.error, 400);
  }

  return answerQuery(payload, env, ctx);
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method === "OPTIONS") {
      return handleOptions();
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (request.method === "GET" && path === "/health") {
      return handleHealth();
    }

    if (request.method === "POST" && path === "/ingest") {
      return handleIngest(request, env);
    }

    if (request.method === "POST" && path === "/search") {
      return handleSearch(request, env);
    }

    if (request.method === "POST" && path === "/query") {
      return handleQuery(request, env, ctx);
    }

    return errorResponse("NOT_FOUND", `No route matches ${request.method} ${path}`, 404);
  },
} satisfies ExportedHandler<Env>;
