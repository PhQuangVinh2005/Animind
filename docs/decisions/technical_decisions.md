# Technical Decisions

## D1: AniList > MyAnimeList
- **Decision:** Use AniList GraphQL API instead of MAL
- **Reason:** 90 req/min (vs MAL 1 req/s), GraphQL flexible queries, no OAuth needed for public media data
- **Trade-off:** Slightly smaller dataset than MAL, but sufficient for demo

## D2: Qdrant > Milvus
- **Decision:** Use Qdrant for vector storage
- **Reason:** Simpler single-node setup, modern REST API, native metadata filtering, active development
- **Trade-off:** Milvus has more enterprise features, but overkill for this project

## D3: Qwen3-Reranker-0.6B via vLLM
- **Decision:** Local reranker instead of cross-encoder or API-based
- **Reason:** ~2GB VRAM (fits easily on 16GB GPU), OpenAI-compatible `/v1/rerank` endpoint, zero cost
- **Trade-off:** Requires vLLM setup, but Docker makes it trivial

## D4: Cloudflare Tunnel > ngrok
- **Decision:** Use Cloudflare Tunnel to expose backend
- **Reason:** Free, stable, custom domain, no session limits, no port forwarding needed
- **Trade-off:** Requires owning a domain managed by Cloudflare

## D5: Vercel for Frontend
- **Decision:** Deploy Next.js frontend on Vercel
- **Reason:** Zero config, free tier, global CDN, native Next.js support
- **Trade-off:** Serverless limitations, but frontend-only so no issue

## D6: RAGAS for Evaluation
- **Decision:** Use RAGAS framework for retrieval evaluation
- **Reason:** Industry standard, multiple metrics (Context Precision/Recall, Faithfulness, Answer Relevancy), LLM-as-judge
- **Trade-off:** Requires GPT-4o calls for judging (~$2 for 50 samples)

## D7: GPT-4o-mini for Development
- **Decision:** Use GPT-4o-mini during development, GPT-4o for demo only
- **Reason:** 30x cheaper, sufficient quality for testing
- **Trade-off:** Slightly lower quality responses during dev

## D8: Cut Hybrid Search
- **Decision:** Remove hybrid search (dense + BM25), keep only dense + reranker
- **Reason:** Reranker compensates for retrieval quality, simplifies pipeline
- **Trade-off:** May miss exact keyword matches, but reranker mitigates this

## D9: SSE > WebSocket
- **Decision:** Server-Sent Events for streaming responses
- **Reason:** Simpler than WebSocket, sufficient for one-way streaming, Vercel AI SDK handles it
- **Trade-off:** No bidirectional communication, but not needed for chat

## D10: SQLite Checkpointer
- **Decision:** Use SQLite for LangGraph state persistence
- **Reason:** Zero config, single-file, sufficient for single-user demo
- **Trade-off:** Not suitable for multi-user production, but that's a non-goal

## D11: Pin Docker Image Versions
- **Decision:** Pin `qdrant:v1.17.1` and `vllm-openai:v0.19.1` instead of `latest`
- **Reason:** Reproducible builds, avoid breaking changes on pull
- **Trade-off:** Must manually bump versions for updates

## D12: vLLM v0.19.1 CLI Changes
- **Decision:** Use `--runner pooling` instead of `--task score`, list-form command in compose
- **Reason:** vLLM v0.19.1 breaking changes:
  1. Entrypoint changed to `["vllm", "serve"]` — `serve` must NOT be in command
  2. `--task` flag removed, replaced with `--runner pooling` for scoring/reranking
  3. `--model` deprecated as flag, model is now a positional argument
- **Trade-off:** Docs/tutorials online still reference old `--task score` syntax
- **Gotcha:** YAML `>` block scalar with JSON `--hf_overrides` causes escaping issues — use list-form command instead

## D13: Named Docker Volumes > Bind Mounts
- **Decision:** Use named volumes (`qdrant_data`, `huggingface_cache`) instead of bind mounts (`./qdrant_data`)
- **Reason:** Docker-managed, portable, avoids permission issues, HF cache persists model weights across restarts
- **Trade-off:** Slightly harder to inspect directly (use `docker volume inspect`)

## D14: ShopAIKey as OpenAI-Compatible Provider
- **Decision:** Route all OpenAI API calls through ShopAIKey (`https://api.shopaikey.com/v1`)
- **Reason:** Cost optimization; ShopAIKey provides OpenAI-compatible endpoints at reduced cost
- **Required:** Must pass browser-like `User-Agent` header or requests are blocked by Cloudflare WAF
- **Client config:**
  ```python
  AsyncOpenAI(
      api_key=settings.shopaikey_api_key,
      base_url="https://api.shopaikey.com/v1",
      default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
  )
  ```
- **Gotcha:** API occasionally returns Cloudflare 522 (origin timeout). Retry after a few minutes.
- **Trade-off:** Dependency on third-party proxy; switch `SHOPAIKEY_BASE_URL` to `https://api.openai.com/v1` to go direct

## D15: Strategy 5 Chunking (Hybrid Metadata Filter + Single Chunk)
- **Decision:** One vector per anime with structured chunk text + rich Qdrant payload
- **Reason:** Anime records are self-contained semantic units (~300-1500 chars). Chunking would break context. Payload metadata enables pre-filtering before vector search.
- **Chunk format:**
  ```
  Title1. Title2. TitleNative. Synonym1. Synonym2.
  Genre1, Genre2, Genre3.
  Tag1 (rank≥70), Tag2, Tag3.
  Synopsis text (HTML-stripped, ≤1000 chars).
  Studio: X. Year: Y. Season: Spring. Format: TV. Episodes: 25. Score: 8.5. Source: Manga.
  ```
- **Payload includes:** `full_data` (entire raw JSON) for reranker and LLM context, plus filterable fields (`year`, `score`, `genres`, `tags`, `format`, `status`)
- **Trade-off:** Slightly larger Qdrant storage due to `full_data` in payload; negligible at 1250 records

## D16: AniList GraphQL Field Expansion (39 fields)
- **Decision:** Fetch 39 fields per anime (expanded from initial ~20)
- **Added fields:** `synonyms`, `source`, `country_of_origin`, `is_adult`, `rankings`, `streaming_links`, `banner_image`, `trailer_info`, `mean_score`, `id_mal`, `studios` (with URLs), `next_airing_episode`, `trending`, `favourites`, `season_year`
- **Reason:** Richer payload enables better metadata filtering, frontend display, and reranker context
- **Trade-off:** Larger `anime.json` (~6.2MB), but one-time cost

## D17: Interrupt-Safe Fetch Checkpointing
- **Decision:** Save page-by-page checkpoint in `data/raw/.fetch_checkpoint.json`
- **Reason:** AniList rate-limits at 30 req/min; a crash mid-fetch would require restarting from scratch
- **Implementation:** Save after every 5 pages. `--reset` flag clears checkpoint for fresh start.
- **Trade-off:** Slightly more complex fetch logic, but protects against data loss

## D18: LLM-Powered Auto-Filter Extraction
- **Decision:** Call GPT-4o-mini to parse metadata filters (year, genre, format, score) from the raw user query before retrieval
- **Reason:** Users phrase queries naturally ("best romance movies from 2020"); manual parsing is brittle; `response_format=json_object` forces well-formed output
- **Implementation:** `extract_filters()` in `chain.py`. Falls back gracefully (returns empty `FilterParams`) on any failure.
- **Trade-off:** One extra LLM call per request (~50ms, ~30 tokens). Disabled by passing `auto_filter=False` or `filter_kwargs=...` directly.
- **FilterParams fields:** `genres`, `year`, `year_min`, `year_max`, `format_`, `score_min`, `is_adult`

## D19: Few-Shot Query Rewriting
- **Decision:** Use a system prompt with 5 input/output examples to rewrite queries before embedding
- **Reason:** Vague conversational queries ("something exciting") embed poorly. Few-shot examples teach the LLM to expand into keyword-rich retrieval queries without explanation. Zero-shot rewriting produced identity rewrites (no-op).
- **Implementation:** `rewrite_query()` in `chain.py` — returns original query on failure or if rewrite is identical.
- **Trade-off:** One extra LLM call per request. Skip with `rewrite=False`.

## D20: Structured System Prompt with Citation Format
- **Decision:** System prompt enforces `[1]`, `[2]` inline citation format and ends with `Sources: [1] Title, [2] Title`
- **Reason:** Grounded answers need traceable citations. Without explicit format instructions, GPT-4o-mini skips citations inconsistently.
- **Prompt structure for caching:** System prompt is static (>1024 tokens) → OpenAI prefix caching activates on second request. User message contains dynamic context passages + question.
- **Few-shot examples in system prompt:** Two complete Q&A pairs demonstrate the expected response structure (direct answer → cited bullets → sources line).
- **Trade-off:** Longer system prompt = more input tokens per call, but offset by caching on repeated requests.

## D21: Contextual Query Reformulation for Multi-Turn
- **Problem:** Follow-up questions ("what about its score?") lose context when used as retrieval queries. The conversation history is stored in LangGraph state but `rag_node` only read the last message.
- **Decision:** Add `_contextualize_query()` step at the start of `rag_node`. Uses GPT-4o-mini with 3 few-shot examples to rewrite follow-ups into self-contained retrieval queries before `extract_filters()` and `rewrite_query()`.
- **Pipeline:** `last_message → contextualize → extract_filters → rewrite → retrieve → rerank → synthesize`
- **History window:** Last 4 turn-pairs (8 messages); older turns dropped to control token cost.
- **Fallback:** Returns original query if history is empty (first turn) or on LLM failure.
- **Trade-off:** One extra LLM call per qa turn (~50ms). Skipped automatically on first message.

## D22: Multi-User Session Orchestration
- **Session identity:** Frontend generates UUID v4 on first page load, stored in localStorage. Sent as `session_id` in every request body.
- **LangGraph mapping:** `session_id` → `thread_id` in `{"configurable": {"thread_id": session_id}}`. Checkpointer isolates state per thread automatically.
- **Singleton graph:** ONE compiled StateGraph built at FastAPI startup (`app.state.agent`), reused across all requests. Thread-safe.
- **Persistence:** `AsyncSqliteSaver` managed via FastAPI lifespan. Survives server restarts.
- **FastAPI lifespan pattern:**
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      async with AsyncSqliteSaver.from_conn_string(settings.agent_db_path) as saver:
          app.state.agent = build_graph(oai_client, checkpointer=saver)
          yield
  ```
- **Scale ceiling:** SQLite write lock under >50 concurrent users → migrate to AsyncPostgresSaver.

## D23: Cloudflare Tunnel for Backend Exposure
- **Decision:** Use Cloudflare Tunnel (`cloudflared`) to expose the homeserver backend at `api.vinhkaguya.me`
- **Reason:** Home router has no static IP and no open inbound ports. Cloudflare Tunnel creates an **outbound** encrypted connection from the homeserver to Cloudflare's edge — no firewall rules, no port forwarding, no DynDNS needed.
- **Architecture:**
  ```
  Browser → https://api.vinhkaguya.me (Cloudflare CDN)
                 ↓  encrypted tunnel (outbound from server)
            cloudflared daemon (kaguyaserver)
                 ↓  http://localhost:8000
            uvicorn FastAPI
  ```
- **Domain:** `vinhkaguya.me` (Namecheap, nameservers delegated to Cloudflare)
- **DNS:** CNAME `api.vinhkaguya.me → <tunnel-uuid>.cfargotunnel.com` (added by `cloudflared tunnel route dns`)
- **Tunnel ID:** `80898b88-f6d7-4092-b694-01035e9c2861`
- **Config:** `~/.cloudflared/config.yml` routes `api.vinhkaguya.me → http://localhost:8000`
- **Credentials:** `~/.cloudflared/80898b88-...json` (keep secret)
- **Setup script:** `scripts/setup_tunnel.sh` automates: install → login → create → route → systemd
- **Trade-off:** Requires a domain managed by Cloudflare. Free tier has no bandwidth limit for tunnels.
- **Bug fixed:** Tunnel ID extraction regex matched name "animind" not UUID — fixed to UUID-only pattern.

## D24: Systemd for Process Management
- **Decision:** Run uvicorn and cloudflared as systemd services (`animind-backend.service`, `cloudflared-animind.service`)
- **Reason:** `Restart=on-failure` + `WantedBy=multi-user.target` ensures auto-recovery on crashes and auto-start on reboots
- **Bug fixed:** Service had `WorkingDirectory` pointing to project root instead of `backend/` — uvicorn could not find `app.main`. Also `EnvironmentFile` pointed to wrong path.
- **Log access:** `journalctl -u animind-backend -f` / `journalctl -u cloudflared-animind -f`
- **Note:** `start.sh`/`stop.sh` are dev scripts (Docker + manual uvicorn). Production always uses systemd.

## D25: Pure ASGI Middleware over Starlette BaseHTTPMiddleware
- **Decision:** Implement all middlewares (RequestID, SecurityHeaders, RequestLogging) as raw ASGI callables
- **Reason:** `BaseHTTPMiddleware` buffers responses and strips headers added after `call_next()`. Pure ASGI intercepts `send()` directly so headers always propagate, including on SSE streaming.
- **Middleware registration order (LIFO):** Last added = outermost. Order: RequestID → SecurityHeaders → CORS → RequestLogging.
- **Gotcha:** curl requires `-i` flag with `grep -i` — uvicorn lowercases all response header names.

## D26: LangChain ChatOpenAI in synthesizer_node for SSE
- **Decision:** Replace raw `AsyncOpenAI` in `synthesizer_node` with LangChain `ChatOpenAI` runnable
- **Reason:** LangGraph `astream_events(version="v2")` only emits `on_chat_model_stream` for LangChain runnables. Raw `AsyncOpenAI` calls are invisible to the event bus — SSE endpoint gets no token deltas.
- **SSE filter:** `POST /chat/stream` filters `on_chat_model_stream` events where `langgraph_node == "synthesizer"`.
- **Trade-off:** Other LLM calls (filter extraction, rewriting, contextualization) stay on raw `AsyncOpenAI` for direct control.

## D27: Self-Hosted Frontend (Option B) over Vercel
- **Decision:** Run Next.js frontend on the homeserver at `:3000` instead of deploying to Vercel
- **Reason:** Simplicity \u2014 no third-party deployment platform, everything managed through the same Cloudflare Tunnel and systemd. Domain infrastructure already in place.
- **Implementation:**
  - `next build && next start` runs as `animind-frontend.service` (systemd, auto-restart)
  - `chat.vinhkaguya.me` added as a second ingress in `~/.cloudflared/config.yml` \u2192 `:3000`
  - `api.vinhkaguya.me` remains as the backend ingress \u2192 `:8000`
  - Both served through the same Cloudflare Tunnel daemon (`cloudflared-animind.service`)
- **Trade-off vs Vercel:**
  - \u2705 No third-party dependency, no build platform to manage
  - \u2705 Simpler mental model \u2014 one server, everything managed via systemd
  - \u274c If homeserver goes down, both frontend AND backend are unavailable (Vercel would keep frontend up)
  - \u274c No automatic deploys on git push (need to ssh + rebuild manually)
- **Streaming:** Native `EventSource` browser API instead of Vercel AI SDK (simpler, no extra dependency)
- **Note:** `vinhkaguya.me` (root domain) continues to show GitHub Pages portfolio \u2014 untouched.

## D28: Next.js SSE Proxy to Bypass Cloudflare Buffering
- **Problem:** Cloudflare Tunnel buffers entire HTTP responses before delivering to the browser, breaking real-time SSE streaming. Token-by-token updates arrived all at once after the full response completed.
- **Decision:** Add a server-side Next.js API route (`/api/chat/stream`) that proxies SSE from the FastAPI backend.
- **Why this works:** The browser talks to `localhost:3000` (or `chat.vinhkaguya.me` same-origin) — no Cloudflare in the streaming path. The Next.js route handler connects to `localhost:8000` over a direct TCP connection on the same machine.
- **Implementation:**
  ```ts
  // app/api/chat/stream/route.ts
  export const dynamic = 'force-dynamic';
  export async function POST(req: NextRequest) {
      const backendRes = await fetch(`${BACKEND_URL}/chat/stream`, {
          method: 'POST', cache: 'no-store', body: req.body, ...
      });
      return new Response(backendRes.body, {
          headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', ... }
      });
  }
  ```
- **Key flags:** `cache: 'no-store'` on the fetch + `dynamic = 'force-dynamic'` on the route prevent Next.js from buffering the response body.
- **Non-streaming calls** (health, sessions) still go directly to `api.vinhkaguya.me` via `NEXT_PUBLIC_API_URL` — no change needed there.
- **Trade-off:** Adds one network hop on the same machine (negligible latency). Requires `BACKEND_URL` server-side env var in addition to `NEXT_PUBLIC_API_URL`.

## D29: Frontend Message Persistence in localStorage
- **Problem:** React state for messages was wiped on every `threadId` change — switching sessions showed an empty chat even if that session had previous messages.
- **Decision:** Persist completed messages per thread in `localStorage` under the key `animind_msgs_<thread_id>`.
- **Implementation (`lib/sessions.ts`):**
  - `saveMessages(threadId, messages)` \u2014 called inside `setMessages` functional updater after `onDone`/`onError`, guaranteeing the latest state is saved.
  - `loadMessages(threadId)` \u2014 called on every `threadId` change to restore history.
  - Only `streaming: false` messages are persisted (in-flight tokens are never saved).
  - `removeSession()` also removes `animind_msgs_<id>` (no orphaned data).
- **Scale ceiling:** `localStorage` quota is ~5MB. At ~2KB per conversation, this supports ~2500 sessions before any issue. Gracefully fails silently on quota exceeded.
- **Trade-off:** Messages are device-local; a different browser/device will not restore history. Acceptable for a single-user local-development setup.

## D30: Flat SSE Reader (While-Loop) over Async Generator
- **Problem:** The original `async function* readSSE()` generator caused silent failures in some browser environments when combined with React 18 batched state updates. The generator protocol overhead also introduced subtle timing issues when streaming state was split across multiple chunks.
- **Decision:** Replace the async generator with a flat `while (true)` loop that processes chunks directly and fires callbacks (`onToken`, `onCards`, `onDone`, `onError`) inline.
- **Key implementation details:**
  - `currentEvent` and `currentData` persist across chunk boundaries (SSE events often span multiple reader chunks).
  - `\r\n` line endings handled alongside `\n` (Windows-style backends).
  - `[DONE]` sentinel breaks the outer loop cleanly.
  - `reader.cancel()` in `finally` ensures the ReadableStream is always released.
- **Trade-off:** Slightly more imperative code vs. the elegant generator pattern. Worth it for reliability and debuggability.

