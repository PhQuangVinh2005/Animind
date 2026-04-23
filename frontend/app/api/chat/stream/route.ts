import { NextRequest } from 'next/server';

/**
 * Proxy route: forwards SSE stream from the local FastAPI backend.
 *
 * Why this exists:
 *   - Frontend and backend are co-located on the same server.
 *   - Sending SSE through the Cloudflare tunnel buffers the entire response
 *     before delivering it to the browser, breaking real-time streaming.
 *   - Proxying through the same-origin Next.js server bypasses Cloudflare
 *     entirely and avoids CORS issues.
 *
 * Flow: Browser → localhost:3000/api/chat/stream → localhost:8000/chat/stream
 */

// Prevent Next.js from statically optimising or caching this route.
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const body = await req.json();

  const backendRes = await fetch(`${BACKEND_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    // Prevent Next.js from caching/buffering the response body.
    cache: 'no-store',
  });

  if (!backendRes.ok) {
    const text = await backendRes.text();
    return new Response(text, { status: backendRes.status });
  }

  // Pipe the backend ReadableStream directly to the browser response.
  // cache: no-store above disables Next.js body buffering.
  return new Response(backendRes.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
