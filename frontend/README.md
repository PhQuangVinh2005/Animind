# AniMind Frontend

Next.js 14 chat interface for the AniMind RAG chatbot.

---

## Stack

- **Next.js 14** (App Router, TypeScript strict)
- **Tailwind CSS** + `@tailwindcss/typography`
- **react-markdown** + **remark-gfm** for assistant responses
- **Native EventSource** for SSE streaming (no Vercel AI SDK)

---

## How It Works

1. Browser sends message to `/api/chat/stream` (Next.js API route)
2. Next.js proxies the request to `backend:8000/chat/stream` over a direct TCP connection
3. Tokens stream back through the proxy to the browser — bypasses Cloudflare buffering
4. Non-streaming calls (`/health`, sessions) go directly to `NEXT_PUBLIC_API_URL`

---

## Environment

```bash
# frontend/.env.local (not committed)
NEXT_PUBLIC_API_URL=https://api.vinhkaguya.me  # public API URL (non-streaming)
BACKEND_URL=http://localhost:8000              # server-side SSE proxy target
```

> **Docker:** `NEXT_PUBLIC_API_URL` is baked at build time via the `ARG` in the Dockerfile. `BACKEND_URL` is set at runtime in `docker-compose.yml`.

---

## Development

```bash
cd frontend
npm install
cp .env.local.example .env.local  # configure URLs
npm run dev   # http://localhost:3000
```

## Docker

```bash
# From project root — builds and starts frontend + NGINX
bash scripts/start-frontend.sh --build
```

The Dockerfile uses a 3-stage build:
1. **deps** — `npm ci` (cached layer)
2. **builder** — `npm run build` (standalone output)
3. **runner** — minimal Alpine image with non-root user

---

## Key Components

| Component | Path | Purpose |
|---|---|---|
| `ChatWindow` | `components/ChatWindow.tsx` | Main chat container, SSE reader, message state |
| `MessageBubble` | `components/MessageBubble.tsx` | Renders user/assistant messages with markdown |
| `AnimeCard` | `components/AnimeCard.tsx` | Anime result card with image, score, metadata |
| `Sidebar` | `components/Sidebar.tsx` | Session list, create/switch/delete threads |
| `SSE proxy` | `app/api/chat/stream/route.ts` | Server-side proxy for streaming |
| `Sessions` | `lib/sessions.ts` | localStorage persistence for messages per thread |

---

## Session Management

- **Session ID:** UUID v4, generated on first page load, stored in `localStorage`
- **Message persistence:** Completed messages saved per-thread in `localStorage` (~2KB per conversation)
- **Thread isolation:** Each `session_id` maps to a LangGraph `thread_id` on the backend
