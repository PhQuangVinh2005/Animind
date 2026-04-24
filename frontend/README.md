# AniMind Frontend

Next.js 14 chat interface for the AniMind RAG chatbot. Self-hosted on the homeserver at `:3000`, exposed via Cloudflare Tunnel at `chat.vinhkaguya.me`.

> See [../../DEVELOPMENT.md](../../DEVELOPMENT.md) for the full developer guide.

## Stack

- **Next.js 14** (App Router)
- **TypeScript** (strict mode)
- **Tailwind CSS** + `@tailwindcss/typography`
- **react-markdown** + **remark-gfm** — markdown rendering for assistant responses

## Local Dev

```bash
# Prerequisites: backend running on :8000, frontend/.env.local configured

npm install
npm run dev     # http://localhost:3000
```

## Production (self-hosted)

```bash
npm run build
npm run start   # :3000, managed by animind-frontend.service systemd unit
```

## Environment

```bash
# frontend/.env.local (not committed)
NEXT_PUBLIC_API_URL=http://localhost:8000   # public API URL (non-streaming)
BACKEND_URL=http://localhost:8000           # server-side proxy target (SSE)
```

## Key Files

| Path | Purpose |
|------|---------|
| `app/page.tsx` | Main page — session state, sidebar + chat layout |
| `app/api/chat/stream/route.ts` | SSE proxy — forwards streaming to FastAPI, bypasses Cloudflare |
| `components/ChatWindow.tsx` | Message list, input bar, streaming orchestration |
| `components/MessageBubble.tsx` | User (plain) / assistant (markdown) message rendering |
| `components/Sidebar.tsx` | Session list, new chat button |
| `components/AnimeCard.tsx` | Anime result card (cover, score, genres) |
| `lib/api.ts` | `streamChat()` — flat SSE reader with `onToken/onCards/onDone/onError` |
| `lib/sessions.ts` | localStorage helpers: sessions + per-thread message persistence |
| `types/index.ts` | `ChatMessage`, `AnimePayload`, `StoredSession` |

## Code Quality

```bash
npm run lint          # ESLint (next lint)
npx tsc --noEmit      # TypeScript strict check
npm audit --audit-level=high  # Dependency security
```
