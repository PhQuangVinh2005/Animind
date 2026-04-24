"""LangGraph graph — StateGraph definition and compilation.

Graph topology:
    [START] → router_node
      ├── "qa"     → rag_node → rerank_node → relevance_gate
      │                  ↑                          │
      │                  └── (retry, max 1) ────────┤
      │                                             ↓
      │                                        synthesizer → [END]
      ├── "search" → tool_node              → synthesizer → [END]
      └── "detail" → tool_node              → synthesizer → [END]

Self-correcting retrieval:
    After reranking, relevance_gate checks the top reranker score:
      - score >= 0.4 OR already retried → proceed to synthesizer
      - score <  0.4 AND first attempt  → retry rag_node without filters

Persistence:
    InMemorySaver by default (multi-turn within a single process).
    For production (Day 4 FastAPI), inject AsyncSqliteSaver via lifespan:

        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            agent = build_graph(oai_client, checkpointer=saver)

Usage:
    from app.openai_client import make_openai_client
    from app.agent.graph import build_graph

    oai = make_openai_client()
    agent = build_graph(oai)

    config = {"configurable": {"thread_id": "session-abc"}}
    result = await agent.ainvoke(
        {"messages": [("user", "best action anime 2023")]},
        config=config,
    )
    print(result["final_answer"])
"""

from __future__ import annotations

from openai import AsyncOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import make_nodes
from app.agent.state import AgentState

# ── Relevance threshold (must match nodes.py _RELEVANCE_THRESHOLD) ───────────
_RELEVANCE_THRESHOLD = 0.4


# ── Routing functions ────────────────────────────────────────────────────────

def _route_by_intent(state: AgentState) -> str:
    """Read intent set by router_node and return the next node name."""
    intent = state.get("intent", "qa")  # type: ignore[call-overload]
    if intent in ("search", "detail"):
        return "tool"
    return "rag"   # "qa" and unknown → RAG path


def _route_relevance_gate(state: AgentState) -> str:
    """After relevance_gate: proceed to synthesizer or retry via rag_node.

    Flow trace for score < 0.4:
      Pass 1: gate sees retry_count=0, returns {retry_count: 1}
              → router sees retry_count=1, score < 0.4 → "rag" (retry)
      Pass 2: gate sees retry_count=1, returns {retry_count: 2} (pass-through)
              → router sees retry_count=2 → "synthesizer" (done)
    """
    top_score: float = state.get("top_rerank_score", 0.0)  # type: ignore[call-overload]
    retry_count: int = state.get("retry_count", 0)  # type: ignore[call-overload]

    # Score is good enough → synthesize
    if top_score >= _RELEVANCE_THRESHOLD:
        return "synthesizer"

    # Exactly 1 = gate just requested a retry → loop back to rag
    if retry_count == 1:
        return "rag"

    # retry_count >= 2 → retry exhausted, proceed with what we have
    return "synthesizer"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    oai_client: AsyncOpenAI,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the AniMind StateGraph.

    Args:
        oai_client:   Pre-configured AsyncOpenAI client (ShopAIKey).
        checkpointer: LangGraph checkpointer for persistence.
                      Defaults to InMemorySaver (multi-turn within process).
                      For production, pass an initialised AsyncSqliteSaver.

    Returns:
        Compiled LangGraph application.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    nodes = make_nodes(oai_client)
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router",         nodes["router"])
    graph.add_node("rag",            nodes["rag"])
    graph.add_node("reranker",       nodes["reranker"])
    graph.add_node("relevance_gate", nodes["relevance_gate"])
    graph.add_node("tool",           nodes["tool"])
    graph.add_node("synthesizer",    nodes["synthesizer"])

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "rag":  "rag",
            "tool": "tool",
        },
    )

    # QA path: retrieve → rerank → relevance gate → (synthesize | retry)
    graph.add_edge("rag",      "reranker")
    graph.add_edge("reranker", "relevance_gate")

    graph.add_conditional_edges(
        "relevance_gate",
        _route_relevance_gate,
        {
            "synthesizer": "synthesizer",
            "rag":         "rag",           # self-correcting retry loop
        },
    )

    # Tool path (search + detail): tool → synthesize
    graph.add_edge("tool", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)
