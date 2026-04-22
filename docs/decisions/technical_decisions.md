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
