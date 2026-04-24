"""AgentState — LangGraph state definition for AniMind agent."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all LangGraph nodes.

    Field lifecycle per turn:
        messages        — append-only (add_messages reducer)
        intent          — set by router_node, read by all downstream nodes
        retrieval_query — set by rag_node (rewritten query), read by rerank_node
        retrieved_docs  — set by rag_node (top-20, serialised dicts), read by rerank_node
        reranked_docs   — set by rerank_node (top-5), read by synthesizer (qa path)
        top_rerank_score — best reranker score from last rerank pass (relevance gate)
        retry_count     — number of retrieval retries (0 = first attempt, max 1)
        tool_output     — set by tool_node (search/detail), read by synthesizer
        final_answer    — set by synthesizer, read by API layer
    """

    # ── Conversation history ──────────────────────────────────────────────────
    messages: Annotated[list, add_messages]   # appends; never overwrites

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str   # "qa" | "search" | "detail"

    # ── RAG path (qa intent) ─────────────────────────────────────────────────
    retrieval_query: str        # rewritten query used for retrieve + rerank
    retrieved_docs: list[dict]  # top-20 from rag_node (serialised)
    reranked_docs: list[dict]   # top-5 from rerank_node (serialised)
    top_rerank_score: float     # best relevance_score from reranker (gate signal)
    retry_count: int            # 0 = first attempt, 1 = retried (max 1)

    # ── Tool path (search / detail intents) ──────────────────────────────────
    tool_output: dict           # {"type": "search"|"detail", "data": {...}}

    # ── Final output ─────────────────────────────────────────────────────────
    final_answer: str           # synthesised answer from synthesizer_node
