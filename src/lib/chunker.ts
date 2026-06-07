import type { TextChunk } from "../types";
import { CHUNK_OVERLAP, CHUNK_SIZE } from "../types";

function normalizeContent(content: string): string {
  let normalized = "";
  for (let i = 0; i < content.length; i++) {
    if (content[i] === "\r" && content[i + 1] === "\n") {
      normalized += "\n";
      i++;
    } else {
      normalized += content[i];
    }
  }
  return normalized;
}

function findSoftBreak(text: string, windowStart: number, windowEnd: number): number {
  const scanStart = Math.max(windowStart, windowEnd - CHUNK_OVERLAP);
  const window = text.slice(scanStart, windowEnd);

  const paragraphBreak = window.lastIndexOf("\n\n");
  if (paragraphBreak !== -1) {
    return scanStart + paragraphBreak + 2;
  }

  const sentenceBreak = window.lastIndexOf(". ");
  if (sentenceBreak !== -1) {
    return scanStart + sentenceBreak + 2;
  }

  return windowEnd;
}

function isWhitespaceOnly(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char !== " " && char !== "\n" && char !== "\t" && char !== "\r") {
      return false;
    }
  }
  return true;
}

export function chunkText(content: string): TextChunk[] {
  const normalized = normalizeContent(content);
  if (normalized.length === 0) {
    return [];
  }

  const chunks: TextChunk[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < normalized.length) {
    const maxEnd = Math.min(start + CHUNK_SIZE, normalized.length);
    let end = maxEnd;

    if (end < normalized.length) {
      end = findSoftBreak(normalized, start, maxEnd);
      if (end <= start) {
        end = maxEnd;
      }
    }

    const slice = normalized.slice(start, end);
    if (!isWhitespaceOnly(slice)) {
      chunks.push({
        index: chunkIndex,
        content: slice,
        startOffset: start,
        endOffset: end,
      });
      chunkIndex++;
    }

    if (end >= normalized.length) {
      break;
    }

    const chunkLength = end - start;
    const advance = Math.max(1, chunkLength - CHUNK_OVERLAP);
    start += advance;
  }

  return chunks;
}
