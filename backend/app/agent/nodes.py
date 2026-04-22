"""LangGraph nodes — router, rag, reranker, tool, synthesizer.

All nodes are created via make_nodes(oai_client) factory to inject
the AsyncOpenAI dependency without global state.

Node flow:
    router_node  → classifies intent ("qa" | "search" | "detail")
    rag_node     → [qa only] extract_filters + rewrite_query + retrieve top-20
    rerank_node  → [qa only] rerank top-20 → top-5
    tool_node    → [search/detail] calls search_anime or get_anime_details
    synthesizer  → generates final answer from reranked_docs or tool_output
"""

from __future__ import annotations

import json
import re
import html as html_lib
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from openai import AsyncOpenAI

from app.agent.state import AgentState
from app.agent.tools import (
    doc_to_dict,
    get_anime_details,
    payload_to_detail,
    search_anime,
)
from app.config import settings
from app.rag.chain import (
    _SYSTEM_PROMPT,
    _build_context,
    extract_filters,
    rewrite_query,
)
from app.rag.retriever import RetrievedDoc, retrieve
from app.rag.reranker import rerank


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _last_user_text(state: AgentState) -> str:
    """Return the text of the most recent HumanMessage in state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _dict_to_doc(d: dict) -> RetrievedDoc:
    """Reconstruct RetrievedDoc from serialised state dict."""
    return RetrievedDoc(
        qdrant_id=d["qdrant_id"],
        score=d["score"],
        payload=d["payload"],
    )


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Contextual query reformulation
# ══════════════════════════════════════════════════════════════════════════════

_CONTEXTUALIZE_SYSTEM = """\
You are a query reformulation assistant for an anime chatbot.
Given a conversation history and the user's latest follow-up question,
rewrite the question as a fully self-contained search query that
does NOT rely on the conversation context to be understood.

Rules:
- If the question already is self-contained, return it unchanged.
- Preserve any specific titles, genres, years, or names mentioned in the history
  that the follow-up implicitly refers to.
- Output ONLY the rewritten query — no quotes, no explanation.

Examples:
History: [user: "tell me about Vinland Saga", ai: "Vinland Saga S2 score 8.8/10..."]
Follow-up: "what about its score and episodes?"
Output: Vinland Saga score and episode count

History: [user: "recommend action anime from 2023", ai: "Blue Lock, Jujutsu Kaisen..."]
Follow-up: "which one has the highest rating?"
Output: highest rated action anime 2023 Blue Lock Jujutsu Kaisen rating

History: [user: "who is Gojo Satoru?", ai: "Special Grade sorcerer..."]
Follow-up: "how strong is he compared to Sukuna?"
Output: Gojo Satoru vs Ryomen Sukuna strength comparison Jujutsu Kaisen
"""


def _format_history_for_context(messages: list, max_turns: int = 4) -> str:
    """Format last N turn-pairs from messages as plain text for the LLM prompt.

    Used by _contextualize_query() to resolve follow-up question references.
    Excludes the latest message (which is the current question being processed).
    """
    turn_msgs = []
    for msg in messages[:-1]:  # exclude the latest user message
        if isinstance(msg, HumanMessage):
            turn_msgs.append(f"user: {(msg.content or '')[:200]}")
        elif isinstance(msg, AIMessage):
            turn_msgs.append(f"ai: {(msg.content or '')[:300]}")
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = (msg.get("content") or "")[:200]
            if role in ("user", "assistant"):
                turn_msgs.append(f"{role}: {content}")
    # Keep only last max_turns * 2 lines (pairs)
    recent = turn_msgs[-(max_turns * 2):]
    return "\n".join(recent) if recent else ""


def _trim_and_format_history(
    messages: list,
    max_turns: int = 5,
) -> list[dict]:
    """Convert LangGraph messages → OpenAI-format dicts, keeping last N turn-pairs.

    Used by synthesizer_node to inject conversation history into the LLM call.
    This gives the LLM full conversational context when generating answers,
    so it can handle comparisons, follow-ups, and references across turns.

    Args:
        messages:  Full state["messages"] list (HumanMessage / AIMessage / dict).
        max_turns: Number of complete turn-pairs (user+assistant) to keep.
                   Default 5 = 10 messages max = ~1500 tokens estimate.

    Returns:
        List of {"role": ..., "content": ...} dicts ready for the OpenAI API,
        excluding the very last user message (that will be the current question).
    """
    oai_msgs: list[dict] = []
    for msg in messages[:-1]:  # exclude current user message
        if isinstance(msg, HumanMessage):
            oai_msgs.append({"role": "user", "content": msg.content or ""})
        elif isinstance(msg, AIMessage):
            oai_msgs.append({"role": "assistant", "content": msg.content or ""})
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            if role in ("user", "assistant"):
                oai_msgs.append({"role": role, "content": msg.get("content") or ""})
    # Keep last max_turns * 2 messages (pairs)
    return oai_msgs[-(max_turns * 2):]


async def _contextualize_query(
    current_query: str,
    messages: list,
    oai_client: AsyncOpenAI,
) -> str:
    """Rewrite follow-up questions into self-contained retrieval queries.

    Returns the rewritten query, or the original if history is empty or on error.
    """
    history = _format_history_for_context(messages)
    if not history:
        return current_query  # first turn — no history, nothing to contextualize

    prompt = (
        f"Conversation history:\n{history}\n\n"
        f"Follow-up question: {current_query}"
    )
    try:
        resp = await oai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _CONTEXTUALIZE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        result = (resp.choices[0].message.content or "").strip()
        if result and result.lower() != current_query.lower():
            logger.debug(
                "contextualize: {orig!r} → {new!r}",
                orig=current_query[:60],
                new=result[:60],
            )
            return result
    except Exception as exc:
        logger.warning("contextualize_query failed ({exc}), using original.", exc=exc)
    return current_query


# ══════════════════════════════════════════════════════════════════════════════
# Context builders (for synthesizer)
# ══════════════════════════════════════════════════════════════════════════════

def _build_context_from_dicts(docs: list[dict]) -> str:
    """Build numbered citation context from serialised RetrievedDoc dicts."""
    return _build_context([_dict_to_doc(d) for d in docs])


def _build_context_from_tool(tool_output: dict) -> str:
    """Format tool_output into a numbered context block for the LLM."""
    tool_type = tool_output.get("type")
    data = tool_output.get("data", {})

    if tool_type == "search":
        results: list[dict] = data.get("results", [])
        if not results:
            return "No anime found matching your criteria."
        sections = []
        for i, r in enumerate(results, 1):
            payload = r.get("payload", {})
            score_raw = payload.get("score")
            score_str = f"{score_raw / 10:.1f}/10" if score_raw else "N/A"
            year = payload.get("year") or ""
            fmt = payload.get("format") or ""
            genres = ", ".join(payload.get("genres") or [])
            meta = ", ".join(p for p in [str(year), fmt] if p)
            header = f"[{i}] **{r.get('title', 'Unknown')}**"
            if meta:
                header += f" ({meta})"
            header += f" | Score: {score_str}"
            if genres:
                header += f" | Genres: {genres}"
            full = payload.get("full_data") or {}
            desc_raw = full.get("description") or payload.get("description") or ""
            desc = _clean_html(desc_raw)[:400]
            sections.append(f"{header}\n{desc}" if desc else header)
        return "\n\n".join(sections)

    elif tool_type == "detail":
        if not data.get("found"):
            return "That anime was not found in the database."
        anime = data.get("anime") or {}
        score_raw = anime.get("score")
        score_str = f"{score_raw / 10:.1f}/10" if score_raw else "N/A"
        lines = [
            f"[1] **{anime.get('title', 'Unknown')}** — Full Details",
            f"Format: {anime.get('format', 'N/A')} | "
            f"Year: {anime.get('year', 'N/A')} | "
            f"Episodes: {anime.get('episodes', 'N/A')} | "
            f"Score: {score_str}",
            f"Genres: {', '.join(anime.get('genres') or [])}",
            f"Studios: {', '.join(anime.get('studios') or [])}",
            f"Status: {anime.get('status', 'N/A')} | "
            f"Source: {anime.get('source', 'N/A')}",
            f"AniList: {anime.get('site_url', 'N/A')}",
            "",
            anime.get("description", "No synopsis available.")[:1200],
        ]
        if anime.get("tags"):
            lines.append(f"Tags: {', '.join(anime['tags'][:10])}")
        return "\n".join(lines)

    return "No data available."


# ══════════════════════════════════════════════════════════════════════════════
# Router prompt
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER_SYSTEM = """\
You are an intent classifier for an anime/manga chatbot.
Classify the user message into exactly one of:
  "qa"     — factual question, plot/character discussion, thematic analysis,
             or asking about a specific anime's properties
  "search" — looking for recommendations or a list matching criteria
  "detail" — requesting comprehensive info about ONE specific title

Return ONLY valid JSON: {"intent": "qa" | "search" | "detail"}

Examples:
User: "what genres does Sword Art Online have?" → {"intent": "qa"}
User: "recommend action anime from 2022" → {"intent": "search"}
User: "find romance anime similar to Toradora" → {"intent": "search"}
User: "tell me everything about Vinland Saga" → {"intent": "detail"}
User: "how many episodes does Hunter x Hunter have?" → {"intent": "qa"}
User: "best sci-fi anime of all time" → {"intent": "search"}
User: "give me full details on Chainsaw Man" → {"intent": "detail"}
User: "what is the plot of Steins;Gate?" → {"intent": "qa"}
User: "who is the strongest in One Piece?" → {"intent": "qa"}
User: "find isekai anime with magic from 2020" → {"intent": "search"}
"""

_TITLE_EXTRACTOR_SYSTEM = """\
Extract the anime/manga title from the user's query.
Return ONLY the title string — no quotes, no explanation.
If no specific title is mentioned, return the original query.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Node factory
# ══════════════════════════════════════════════════════════════════════════════

def make_nodes(oai_client: AsyncOpenAI) -> dict[str, Callable]:
    """Return a dict of LangGraph node callables bound to oai_client."""

    # ── router_node ───────────────────────────────────────────────────────────
    async def router_node(state: AgentState) -> dict:
        """Classify user intent: 'qa' | 'search' | 'detail'."""
        query = _last_user_text(state)
        try:
            resp = await oai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM},
                    {"role": "user",   "content": query},
                ],
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = json.loads(resp.choices[0].message.content or "{}")
            intent = raw.get("intent", "qa")
            if intent not in ("qa", "search", "detail"):
                intent = "qa"
        except Exception as exc:
            logger.warning("router_node failed ({exc}), defaulting to 'qa'", exc=exc)
            intent = "qa"

        logger.info("router_node: intent={i!r} for query={q!r}", i=intent, q=query[:60])
        return {"intent": intent}

    # ── rag_node ──────────────────────────────────────────────────────────────
    async def rag_node(state: AgentState) -> dict:
        """[qa path] Contextualize → auto-filter → retrieve top-20 from Qdrant."""
        query = _last_user_text(state)

        # Step 0: Contextualize follow-up questions using conversation history
        # e.g. "what about its score?" + history → "Vinland Saga score"
        contextualized = await _contextualize_query(
            query, state["messages"], oai_client
        )

        # Auto-extract metadata filters from the contextualized query
        extracted = await extract_filters(contextualized, oai_client)
        filter_kwargs = extracted.to_dict() if not extracted.is_empty() else None

        # Rewrite the contextualized query for retrieval
        rewritten = await rewrite_query(contextualized, oai_client)

        # Retrieve top-20
        candidates = await retrieve(
            rewritten,
            oai_client,
            top_k=settings.retriever_top_k,
            filter_kwargs=filter_kwargs,
        )

        serialised = [doc_to_dict(d) for d in candidates]
        logger.debug(
            "rag_node: query={q!r} → contextualized={c!r} → rewritten={r!r} | {n} docs",
            q=query[:50],
            c=contextualized[:50],
            r=rewritten[:50],
            n=len(serialised),
        )
        return {
            "retrieval_query": rewritten,
            "retrieved_docs": serialised,
        }

    # ── rerank_node ───────────────────────────────────────────────────────────
    async def rerank_node(state: AgentState) -> dict:
        """[qa path] Rerank retrieved_docs → top-5 using Qwen3-Reranker."""
        docs_dicts: list[dict] = state.get("retrieved_docs", [])  # type: ignore[assignment]
        if not docs_dicts:
            logger.warning("rerank_node: no retrieved_docs in state, skipping.")
            return {"reranked_docs": []}

        query = state.get("retrieval_query") or _last_user_text(state)  # type: ignore[call-overload]
        doc_texts = [d["chunk_text"] for d in docs_dicts]

        ranked = await rerank(query, doc_texts, top_k=settings.reranker_top_k)
        reranked = [docs_dicts[r["index"]] for r in ranked]

        logger.debug(
            "rerank_node: {total} → top-{k}, best={s:.3f}",
            total=len(docs_dicts),
            k=len(reranked),
            s=ranked[0]["relevance_score"] if ranked else 0,
        )
        return {"reranked_docs": reranked}

    # ── tool_node ─────────────────────────────────────────────────────────────
    async def tool_node(state: AgentState) -> dict:
        """[search/detail path] Dispatch to search_anime or get_anime_details."""
        intent = state.get("intent", "search")  # type: ignore[call-overload]
        query = _last_user_text(state)

        if intent == "search":
            extracted = await extract_filters(query, oai_client)
            result = await search_anime(query, oai_client, **extracted.to_dict())
            return {"tool_output": {"type": "search", "data": result}}

        elif intent == "detail":
            # Extract the specific title from the query
            try:
                title_resp = await oai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": _TITLE_EXTRACTOR_SYSTEM},
                        {"role": "user",   "content": query},
                    ],
                    max_tokens=50,
                    temperature=0.0,
                )
                identifier = (title_resp.choices[0].message.content or query).strip()
            except Exception:
                identifier = query

            result = await get_anime_details(identifier, oai_client)
            return {"tool_output": {"type": "detail", "data": result}}

        # Fallback
        return {"tool_output": {"type": "search", "data": {"results": [], "count": 0}}}

    # ── synthesizer_node ──────────────────────────────────────────────────────
    async def synthesizer_node(state: AgentState) -> dict:
        """Generate final answer from reranked_docs (qa) or tool_output (search/detail)."""
        intent = state.get("intent", "qa")  # type: ignore[call-overload]
        query = _last_user_text(state)

        # Build context depending on path
        if intent == "qa":
            reranked: list[dict] = state.get("reranked_docs", [])  # type: ignore[assignment]
            if not reranked:
                answer = (
                    "I couldn't find relevant information for your question. "
                    "Try rephrasing or asking about a specific anime title."
                )
                return {
                    "messages": [AIMessage(content=answer)],
                    "final_answer": answer,
                }
            context = _build_context_from_dicts(reranked)
        else:
            tool_out: dict = state.get("tool_output", {})  # type: ignore[assignment]
            context = _build_context_from_tool(tool_out)

        user_message = (
            f"Context passages:\n\n{context}\n\n"
            f"---\n\n"
            f"User question: {query}"
        )

        # Build message list:
        #   [SYSTEM]              static role + rules + few-shot   ← prefix-cached
        #   [USER/ASSISTANT] ...  last 5 conversation turns         ← dynamic
        #   [USER]                RAG context + current question    ← dynamic
        #
        # Injecting history lets the LLM handle comparisons, follow-ups,
        # and cross-turn references (e.g. "which of those two has better score?").
        history_msgs = _trim_and_format_history(state["messages"], max_turns=5)

        messages_payload: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history_msgs,                                  # last 5 turn-pairs
            {"role": "user",   "content": user_message},    # current Q + RAG context
        ]

        resp = await oai_client.chat.completions.create(
            model=settings.openai_model,
            messages=messages_payload,
            temperature=0.3,
            max_tokens=900,
        )
        answer = resp.choices[0].message.content or ""
        logger.info(
            "synthesizer: intent={i}, answer_len={n}",
            i=intent,
            n=len(answer),
        )
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
        }

    return {
        "router":      router_node,
        "rag":         rag_node,
        "reranker":    rerank_node,
        "tool":        tool_node,
        "synthesizer": synthesizer_node,
    }
