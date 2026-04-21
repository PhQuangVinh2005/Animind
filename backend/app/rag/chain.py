"""RAG chain — full retrieval-augmented generation pipeline."""

# TODO Day 2: Full RAG chain
# Pipeline:
#   User Query
#     → Query Rewrite (GPT-4o-mini)
#     → Qdrant Retrieve top-20 (+ metadata filter)
#     → Qwen3 Reranker (vLLM) → top-5
#     → Context Augmentation
#     → GPT-4o Generate Answer
