# Technical Decisions

## D1: AniList > MyAnimeList
- **Decision:** Use AniList GraphQL API instead of MAL
- **Reason:** 90 req/min (vs MAL 1 req/s), GraphQL flexible queries, no OAuth needed
- **Trade-off:** Slightly smaller dataset than MAL, but sufficient for demo

## D2: Qdrant > Milvus
- **Decision:** Use Qdrant for vector storage
- **Reason:** Simpler single-node setup, modern REST API, native metadata filtering
- **Trade-off:** Milvus has more enterprise features, but overkill for this project

## D3: Qwen3-Reranker-0.6B via vLLM
- **Decision:** Local reranker instead of cross-encoder or API-based
- **Reason:** ~2GB VRAM (fits easily on 16GB GPU), OpenAI-compatible `/v1/rerank` endpoint, zero cost
- **Trade-off:** Requires vLLM setup, but docker makes it trivial

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
