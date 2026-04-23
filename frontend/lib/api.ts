import { AnimePayload } from '@/types';

// Public backend URL (via Cloudflare tunnel) — used for non-streaming calls
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// SSE streaming goes through the local Next.js proxy route to avoid
// Cloudflare's response buffering which would break token-by-token streaming.
const STREAM_URL = '/api/chat/stream';

// ── SSE streaming ─────────────────────────────────────────────────────────────
// Uses a flat while-loop (not async generator) for reliable browser compatibility.
// Native EventSource only supports GET; our backend uses POST /chat/stream.

// ── Stream chat ───────────────────────────────────────────────────────────────

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onCards: (cards: AnimePayload[]) => void;
  onDone: () => void;
  onError: (msg: string) => void;
}

export async function streamChat(
  message: string,
  threadId: string,
  callbacks: StreamCallbacks,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(STREAM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, thread_id: threadId }),
    });
  } catch (err) {
    callbacks.onError(err instanceof Error ? err.message : 'Network error');
    return;
  }

  if (!response.ok) {
    callbacks.onError(`HTTP ${response.status}`);
    return;
  }

  if (!response.body) {
    callbacks.onError('No response body');
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent: string | undefined;
  let currentData: string | undefined;

  // Returns true when the stream is complete ([DONE] received or error)
  const handleEvent = (event: string | undefined, data: string): boolean => {
    if (event === 'error') { callbacks.onError(data); return true; }
    if (event === 'answer') { callbacks.onToken(data); return false; }
    if (event === 'cards') {
      try { callbacks.onCards(JSON.parse(data)); } catch { /* ignore */ }
      return false;
    }
    if (data === '[DONE]') { return true; } // caller will invoke onDone
    if (data) callbacks.onToken(data);
    return false;
  };

  try {
    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, '');
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6);
        } else if (line === '') {
          if (currentData !== undefined) {
            const finished = handleEvent(currentEvent, currentData);
            currentEvent = undefined;
            currentData = undefined;
            if (finished) { break outer; }
          }
        }
      }
    }
  } catch (err) {
    callbacks.onError(err instanceof Error ? err.message : 'Stream error');
    return;
  } finally {
    reader.cancel().catch(() => { /* ignore */ });
  }

  callbacks.onDone();
}

// ── Health check ──────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  }
}
