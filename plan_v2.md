# 🎌 AniMind v2 — Kế hoạch Chi tiết (7 ngày)

## Thay đổi so với v1

| Hạng mục | v1 | v2 |
|---|---|---|
| Reranker | Cắt bỏ | ✅ Qwen3-Reranker-0.6B trên vLLM (~2GB VRAM) |
| Retrieval Eval | Không có | ✅ RAGAS (Context Precision/Recall/Relevancy) |
| Deployment | Chỉ local | ✅ Self-hosted (frontend + backend) trên homeserver qua Cloudflare Tunnel |
| Hybrid Search | Giữ | ❌ Cắt — reranker đã bù chất lượng |
| MAL API | Không dùng | Không dùng (AniList 90 req/min >> MAL 1 req/s) |
| LangGraph | OSS local | OSS local — MIT, không giới hạn |

---

## 🏗️ Kiến trúc tổng thể

```
  Browser → https://chat.vinhkaguya.me          Browser → https://api.vinhkaguya.me
                    │                                              │
         ┌──────────▼──────────────────────────────────────────────▼──────────┐
         │                    Cloudflare CDN + Tunnel                         │
         │          (vinhkaguya.me nameservers → Cloudflare)                  │
         └──────────┬──────────────────────────────────────────────┬──────────┘
                    │ chat.vinhkaguya.me → :3000                   │ api.vinhkaguya.me → :8000
  ╔═════════════════▼══════════════════════════════════════════════▼═══════════╗
  ║                         HOMESERVER (Debian)                                ║
  ║           RTX 5060Ti 16GB · R7 5700X · 32GB RAM                           ║
  ║                                                                            ║
  ║  ┌─────────────────────────┐   ┌─────────────────────────────────────┐    ║
  ║  │  Next.js Frontend (:3000)│   │      FastAPI Backend (:8000)         │    ║
  ║  │  Chat UI + SSE client   │   │  ┌───────────────────────────────┐   │    ║
  ║  └─────────────────────────┘   │  │       LangGraph Agent         │   │    ║
  ║                                │  │  [START] → Router             │   │    ║
  ║                                │  │  ├── RAG → Reranker → Answer  │   │    ║
  ║                                │  │  └── Tool Node → Answer       │   │    ║
  ║                                │  │                    → [END]    │   │    ║
  ║                                │  └───────────────────────────────┘   │    ║
  ║                                │  Tools: search_anime · get_details   │    ║
  ║                                └──────────┬───────────────────────────┘    ║
  ║                                           │                                ║
  ║  ┌──────────────────────┐  ┌─────────────▼────────┐  ┌──────────────────┐ ║
  ║  │  Qdrant (Docker)     │  │  vLLM Reranker       │  │  AniList GraphQL │ ║
  ║  │  :6333 · ~1250 vecs  │  │  :8001 · Qwen3-0.6B  │  │  (live API)      │ ║
  ║  └──────────────────────┘  └──────────────────────┘  └──────────────────┘ ║
  ╚════════════════════════════════════════════════════════════════════════════╝

  VRAM Budget: Reranker ~2GB / 16GB available = OK
  RAM Budget:  Qdrant ~200MB + FastAPI ~500MB + vLLM ~2GB + Next.js ~200MB / 32GB = OK
```

---

## 📅 Timeline chi tiết

### Ngày 1 — Hạ tầng + Data Ingestion (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-2 | Docker setup | Qdrant container, vLLM container (Qwen3-Reranker-0.6B) |
| 2-4 | Fetch AniList | Script `fetch_anilist.py`: pagination 50/page, ~400 requests, ~5 phút |
| 4-6 | Chunking + Embedding | Mỗi anime = 1 doc (title + synopsis + metadata). `text-embedding-3-small` → upsert Qdrant |
| 6-7 | Verify | Test similarity search + test reranker endpoint `/v1/rerank` |
| 7-8 | Buffer | Fix issues |

**Deliverables:**
- `scripts/fetch_anilist.py` — fetch + save raw JSON
- `scripts/ingest.py` — chunk, embed, upsert
- `docker-compose.yml` — Qdrant + vLLM
- Qdrant collection `anime` có ~20k records
- vLLM reranker chạy tại `:8001`

**vLLM Reranker setup:**
```yaml
# docker-compose.yml snippet (vLLM v0.19.1)
# NOTE: Image ENTRYPOINT=["vllm","serve"], so command = args to "vllm serve"
# NOTE: --task removed in v0.19, replaced with --runner pooling
reranker:
  image: vllm/vllm-openai:v0.19.1
  command:
    - "Qwen/Qwen3-Reranker-0.6B"
    - "--host"
    - "0.0.0.0"
    - "--port"
    - "8001"
    - "--runner"
    - "pooling"
    - "--gpu-memory-utilization"
    - "0.15"
    - "--max-model-len"
    - "512"
    - "--hf_overrides"
    - '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

---

### Ngày 2 — RAG Pipeline + Reranker Integration (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-3 | RAG chain cơ bản | LangChain: QdrantVectorStore → ChatOpenAI (GPT-4o-mini cho dev) |
| 3-5 | Reranker integration | Retrieve top-20 → POST `/v1/rerank` → lấy top-5 |
| 5-6 | Prompt engineering | System prompt, context injection, cite nguồn |
| 6-7 | Query Rewriting | 1 LLM call rewrite trước khi retrieve |
| 7-8 | Metadata filtering | Qdrant payload filter (year, genre, score, type) |

**RAG Pipeline Flow:**
```
User Query
  → Query Rewrite (GPT-4o-mini)
  → Qdrant Retrieve top-20 (+ metadata filter nếu có)
  → Qwen3 Reranker (vLLM) → top-5
  → Context Augmentation
  → GPT-4o Generate Answer
```

**Deliverables:**
- `rag/retriever.py` — Qdrant retriever + metadata filter
- `rag/reranker.py` — vLLM reranker client
- `rag/chain.py` — full RAG chain
- Test 10+ câu hỏi đa dạng

---

### Ngày 3 — LangGraph Agent + Tools (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-3 | State + Graph design | StateGraph, nodes, conditional edges |
| 3-5 | 2 Tools | `search_anime(query, filters)`, `get_anime_details(id)` |
| 5-7 | Router logic | Intent classification: qa → RAG node, search/detail → Tool node |
| 7-8 | Test + Debug | Test multi-turn, test tool-calling |

**State definition:**
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str          # "qa" | "search" | "detail"
    retrieved_docs: list
    tool_output: dict
    final_answer: str
```

**Graph:**
```
[START] → router_node
  ├── "qa" → rag_node → rerank_node → synthesizer → [END]
  ├── "search" → tool_node (search_anime) → synthesizer → [END]
  └── "detail" → tool_node (get_details) → synthesizer → [END]
```

**Deliverables:**
- `agent/state.py`, `agent/nodes.py`, `agent/tools.py`, `agent/graph.py`
- Agent xử lý đúng 3 intents
- Multi-turn conversation hoạt động

---

### Ngày 4 — FastAPI + Streaming + Tunnel (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-3 | FastAPI endpoints | `POST /chat`, `GET /chat/stream` (SSE), `GET /health` |
| 3-4 | LangGraph streaming | `.astream_events()` → SSE |
| 4-5 | Checkpointer | SQLite checkpointer, thread_id per session |
| 5-6 | CORS + error handling | CORS cho Vercel domain, proper error responses |
| 6-8 | Cloudflare Tunnel | Expose `:8000` → `api.yourdomain.com` |

**Cloudflare Tunnel setup:**
```bash
# 1. Cài cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/

# 2. Login + tạo tunnel
cloudflared tunnel login
cloudflared tunnel create animind

# 3. Config
# ~/.cloudflared/config.yml
tunnel: <TUNNEL_ID>
credentials-file: /home/user/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404

# 4. Route DNS + chạy
cloudflared tunnel route dns animind api.yourdomain.com
cloudflared tunnel run animind
```

**Deliverables:**
- FastAPI chạy `:8000`, test bằng curl OK
- SSE streaming hoạt động
- Cloudflare Tunnel: `api.yourdomain.com` → homeserver
- `cloudflared` chạy như systemd service

---

### Ngày 5 — Next.js Frontend + Self-Host trên Homeserver (8h)

> **Deployment thay đổi:** Không dùng Vercel. Frontend chạy trực tiếp trên homeserver tại `:3000`,
> expose qua Cloudflare Tunnel → `chat.vinhkaguya.me`. Mọi thứ (frontend + backend) trên cùng một máy.

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-3 | Chat UI | ChatWindow, MessageBubble, StreamingText (SSE native EventSource) |
| 3-5 | Anime Cards | AnimeCard component, poster image, score, genres |
| 5-6 | Dark theme + polish | Anime aesthetic, responsive basics |
| 6-7 | Connect backend | `NEXT_PUBLIC_API_URL=https://api.vinhkaguya.me` |
| 7-8 | Self-host + tunnel | `next build && next start`, thêm route vào cloudflared config, systemd service |

**Cloudflare Tunnel config sau Day 5:**
```yaml
tunnel: 80898b88-f6d7-4092-b694-01035e9c2861
credentials-file: ~/.cloudflared/80898b88-....json

ingress:
  - hostname: chat.vinhkaguya.me
    service: http://localhost:3000
  - hostname: api.vinhkaguya.me
    service: http://localhost:8000
  - service: http_status:404
```

**Systemd service thêm:**
- `animind-frontend.service` — `next start` trên `:3000`, WorkingDirectory=frontend/

**Components (ưu tiên):**

| Priority | Component | Bắt buộc |
|---|---|---|
| P0 | ChatWindow + MessageBubble | ✅ |
| P0 | StreamingText (native EventSource) | ✅ |
| P1 | AnimeCard | ✅ |
| P2 | Suggested questions | Nice-to-have |
| P3 | SearchBar + autocomplete | Cắt |

**Deliverables:**
- Frontend live tại `https://chat.vinhkaguya.me`
- Chat + streaming hoạt động end-to-end
- Dark theme, anime cards hiển thị đúng
- `animind-frontend.service` chạy như systemd service

---

### Ngày 6 — Retrieval Evaluation (RAGAS) + Polish (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-2 | Tạo test set | 50 câu hỏi + ground truth answers (manual + LLM-assisted) |
| 2-4 | RAGAS evaluation | Context Precision, Context Recall, Faithfulness, Answer Relevancy |
| 4-5 | So sánh | Eval có reranker vs không reranker → chứng minh reranker cải thiện |
| 5-6 | Ablation | Eval có query rewrite vs không → đo impact |
| 6-8 | Bug fixes + UI polish | Fix issues từ end-to-end testing |

**RAGAS Metrics:**

| Metric | Đo gì | Target |
|---|---|---|
| **Context Precision** | Retrieved docs có relevant không? | ≥ 0.7 |
| **Context Recall** | Có thiếu thông tin quan trọng không? | ≥ 0.7 |
| **Faithfulness** | Answer có bịa thông tin không? | ≥ 0.8 |
| **Answer Relevancy** | Answer có trả lời đúng câu hỏi không? | ≥ 0.8 |

**Evaluation script snippet:**
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    faithfulness, answer_relevancy
)

# So sánh: với reranker vs không reranker
results_with_reranker = evaluate(
    dataset=test_dataset,
    metrics=[context_precision, context_recall,
             faithfulness, answer_relevancy],
)
results_without_reranker = evaluate(...)

# Output comparison table
```

**Deliverables:**
- `eval/test_set.json` — 50 QA pairs
- `eval/evaluate.py` — RAGAS evaluation script
- `eval/results/` — comparison tables + charts
- Báo cáo: reranker impact, query rewrite impact

---

### Ngày 7 — Testing, Docker Compose, Documentation (8h)

| Giờ | Công việc | Chi tiết |
|---|---|---|
| 1-3 | End-to-end testing | 20 câu hỏi đa dạng, edge cases, multi-turn |
| 3-4 | Docker Compose final | Backend + Qdrant + vLLM reranker + cloudflared |
| 4-6 | README + báo cáo | Kiến trúc, setup guide, evaluation results, screenshots |
| 6-7 | Demo script | 6 câu hỏi showcase mọi tính năng |
| 7-8 | Final check | Vercel live, tunnel stable, mọi thứ chạy OK |

**Docker Compose cuối cùng:**
```yaml
# See docker-compose.yml for full production config with health checks,
# resource limits, named volumes, and log rotation.
services:
  qdrant:
    image: qdrant/qdrant:v1.17.1
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]

  reranker:
    image: vllm/vllm-openai:v0.19.1
    # ENTRYPOINT=["vllm","serve"], --task→--runner pooling in v0.19
    command:
      - "Qwen/Qwen3-Reranker-0.6B"
      - "--port"
      - "8001"
      - "--runner"
      - "pooling"
      - "--gpu-memory-utilization"
      - "0.15"
      - "--hf_overrides"
      - '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant, reranker]

  tunnel:
    image: cloudflare/cloudflared:latest
    command: tunnel run animind
    volumes: ["./cloudflared:/etc/cloudflared"]
    depends_on: [backend]

volumes:
  qdrant_data:
  huggingface_cache:
```

**Deliverables:**
- Project hoàn chỉnh, chạy bằng `docker compose up`
- README với setup guide
- Frontend live trên Vercel
- Backend accessible qua Cloudflare Tunnel
- Evaluation report với RAGAS metrics

---

## 📁 Cấu trúc Project cuối cùng

```
animind/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── agent/
│   │   │   ├── graph.py         # LangGraph StateGraph
│   │   │   ├── nodes.py         # router, rag, tool, synthesizer
│   │   │   ├── tools.py         # search_anime, get_anime_details
│   │   │   └── state.py         # AgentState TypedDict
│   │   ├── rag/
│   │   │   ├── retriever.py     # Qdrant + metadata filter
│   │   │   ├── reranker.py      # vLLM Qwen3 reranker client
│   │   │   └── chain.py         # Full RAG pipeline
│   │   └── api/
│   │       └── routes.py        # REST + SSE endpoints
│   ├── scripts/
│   │   ├── fetch_anilist.py     # AniList GraphQL fetcher
│   │   └── ingest.py            # Chunk + embed + upsert
│   ├── eval/
│   │   ├── test_set.json        # 50 QA pairs with ground truth
│   │   ├── evaluate.py          # RAGAS evaluation
│   │   └── results/             # Metrics, charts
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/page.tsx
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── AnimeCard.tsx
│   │   └── StreamingText.tsx
│   ├── lib/api.ts
│   └── package.json
├── cloudflared/                  # Tunnel config
├── docker-compose.yml
└── README.md
```

---

## 💰 Cost Estimate

| Hạng mục | Chi phí |
|---|---|
| Embedding 20k records (text-embedding-3-small) | ~$0.08 |
| GPT-4o-mini dev/testing (~300 queries) | ~$0.50 |
| GPT-4o demo (~20 queries) | ~$0.50 |
| RAGAS evaluation (GPT-4o judge, 50×4 metrics) | ~$2.00 |
| Qdrant Docker | $0 |
| vLLM Qwen3-Reranker | $0 (local) |
| Vercel Frontend | $0 (free tier) |
| Cloudflare Tunnel | $0 (free) |
| **Tổng** | **~$3–5** |

---

## ⚠️ Rủi ro & Mitigation

| Rủi ro | Khả năng | Mitigation |
|---|---|---|
| Ngày 3 (LangGraph) bị trễ | Trung bình | Dùng `create_react_agent()` trước, custom graph sau |
| vLLM reranker OOM | Thấp | 0.6B chỉ ~2GB, RTX 5060Ti có 16GB |
| Cloudflare Tunnel unstable | Thấp | Chạy `cloudflared` as systemd service, auto-restart |
| RAGAS eval chậm | Trung bình | Giới hạn 50 samples, dùng GPT-4o-mini cho judge |
| Vercel → Tunnel latency | Thấp | Acceptable cho demo, không phải production |
| Ngày 5-6 bị trễ | Trung bình | Frontend minimal (chat only), eval giảm xuống 30 samples |

---

## 🎯 Demo Script

1. **RAG cơ bản:** *"Nội dung chính của Fullmetal Alchemist Brotherhood là gì?"*
2. **Semantic search:** *"Anime nào có chủ đề về mất mát và chuộc lỗi?"*
3. **Tool-calling:** *"Cho tôi thông tin chi tiết về Spy x Family"*
4. **Metadata filter:** *"Anime shounen score trên 8.5 năm 2023?"*
5. **Multi-turn:** *"Nhân vật chính của anime đó là ai?"*
6. **Query rewriting:** *"anime buồn về gia đình"*
7. **Evaluation:** Show RAGAS results — reranker vs no reranker comparison

---

## 📊 Decision Log (cập nhật)

| Quyết định | Lý do |
|---|---|
| AniList > MAL | 90 req/min, GraphQL, no auth (MAL chỉ 1 req/s, cần OAuth) |
| Qwen3-Reranker-0.6B > cross-encoder | Nhẹ (~2GB), vLLM serve dễ, OpenAI-compatible API |
| Cloudflare Tunnel > ngrok | Free, stable, custom domain, no session limits |
| Vercel > self-host frontend | Zero config, free, CDN global |
| RAGAS > custom eval | Industry standard, đa metric, LLM-as-judge |
| GPT-4o-mini cho dev > GPT-4o | Rẻ 30x, đủ tốt cho development/testing |
| Cắt Hybrid Search | Reranker đã bù chất lượng retrieval |
