"""RAG chain — full retrieval-augmented generation pipeline.

Pipeline:
    User query
      → extract_filters()   — GPT-4o-mini parses year/genre/format from query
      → rewrite_query()     — GPT-4o-mini few-shot rewrite for retrieval
      → retrieve() top-20   — Qdrant vector search + optional metadata filter
      → rerank() top-5      — Qwen3-Reranker-0.6B (local vLLM)
      → _build_context()    — numbered context block with citation anchors
      → generate answer     — GPT-4o-mini with structured system prompt

Public API:
    # Non-streaming
    answer = await rag_answer(query, oai_client)

    # Streaming (async generator)
    async for chunk in await rag_answer(query, oai_client, stream=True):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, cast

from loguru import logger
from openai import AsyncOpenAI

from app.config import settings
from app.rag.reranker import rerank
from app.rag.retriever import RetrievedDoc, retrieve


# ══════════════════════════════════════════════════════════════════════════════
# Hour 5-6 — Prompt Engineering
# System prompt is intentionally long (>1024 tokens) so OpenAI's automatic
# prefix caching kicks in for repeated calls with the same system prompt.
# Static content (role + rules + format) stays first; dynamic content (context
# + user message) appended per-request — prefix is always a cache hit.
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are AniMind, an expert anime and manga assistant backed by a curated \
database of 1 250+ AniList entries. Your goal is to give accurate, grounded, \
and helpful answers about anime and manga.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY from the numbered context passages provided below.
   Do NOT use external knowledge or invent details not present in the context.
2. Cite sources using inline bracket notation: [1], [2], [3] …
   Each fact, score, or recommendation must link to the passage it came from.
3. If the context does not contain enough information, say exactly:
   "I don't have enough information about that in my database. Try rephrasing \
or ask about a different title."
4. Respond in the SAME LANGUAGE the user used (Vietnamese → Vietnamese, \
English → English, etc.).
5. Never fabricate scores, episode counts, studios, or air dates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT (STRICT — follow exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Opening line**: Direct answer in 1–2 sentences.
2. **Details**: Use a **markdown bullet list** (one bullet per anime title).
   Each bullet MUST follow this template:
   - **Title** [N] — Year, Format. Brief description. Score: X.X/10.
3. **NEVER** write multiple titles in the same paragraph.
   Each title MUST be its own bullet point on a separate line.
4. If recommending multiple titles, rank by score (highest first).
5. End with: Sources: [1] Title A, [2] Title B, …

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Context]
[1] Fullmetal Alchemist: Brotherhood (2009, TV) | Score: 9.1/10 | Genres: Action, Adventure, Drama
Two brothers use alchemy to search for the Philosopher's Stone after a failed resurrection attempt costs them dearly.

[2] Attack on Titan (2013, TV) | Score: 8.9/10 | Genres: Action, Drama, Fantasy
Humanity battles giant humanoid titans behind massive walls in a desperate fight for survival.

[User] What is the highest-rated action anime in the database?

[Assistant]
The highest-rated action anime in the database is **Fullmetal Alchemist: Brotherhood** with a score of 9.1/10 [1].

- **Fullmetal Alchemist: Brotherhood** [1] — 2009, TV. Two alchemist brothers quest for the Philosopher's Stone after a failed human transmutation. Score: 9.1/10.
- **Attack on Titan** [2] — 2013, TV. Humanity fights giant titans behind massive walls in a desperate battle for survival. Score: 8.9/10.

Sources: [1] Fullmetal Alchemist: Brotherhood, [2] Attack on Titan

---

[Context]
[1] Toradora! (2008, TV) | Score: 8.0/10 | Genres: Comedy, Drama, Romance, Slice of Life
A boy with a scary face and a tiny, fierce girl team up to help each other confess to their best friends — only to fall for each other instead.

[2] Kaguya-sama: Love Is War (2019, TV) | Score: 8.4/10 | Genres: Comedy, Romance
Two student council geniuses wage psychological warfare to make the other confess first.

[3] Your Lie in April (2014, TV) | Score: 8.5/10 | Genres: Drama, Music, Romance
A piano prodigy who lost the ability to hear his own playing rediscovers music through a free-spirited violinist.

[4] Horimiya (2021, TV) | Score: 8.1/10 | Genres: Comedy, Romance, Slice of Life
A popular girl and a quiet loner discover each other's hidden sides and grow closer.

[User] Recommend some good romance anime

[Assistant]
Here are some highly rated romance anime from the database:

- **Your Lie in April** [3] — 2014, TV. A piano prodigy rediscovers music through a free-spirited violinist. A deeply emotional story about love, loss, and music. Score: 8.5/10.
- **Kaguya-sama: Love Is War** [2] — 2019, TV. Two brilliant student council members engage in hilarious mind games to make the other confess first. Score: 8.4/10.
- **Horimiya** [4] — 2021, TV. A popular girl and a quiet loner discover each other's hidden sides. Sweet and wholesome. Score: 8.1/10.
- **Toradora!** [1] — 2008, TV. A classic rom-com where two unlikely allies help each other confess — and end up falling for each other. Score: 8.0/10.

Sources: [1] Toradora!, [2] Kaguya-sama: Love Is War, [3] Your Lie in April, [4] Horimiya
"""


# ══════════════════════════════════════════════════════════════════════════════
# Hour 7-8 — Metadata Filtering (auto-extraction from natural language)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterParams:
    """Structured metadata filter extracted from a user query."""
    genres: list[str] = field(default_factory=list)
    year: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    format_: str | None = None    # "TV" | "MOVIE" | "OVA" | "ONA" | "SPECIAL"
    score_min: int | None = None  # 0–100 raw (e.g. 75 = 7.5/10)
    is_adult: bool | None = None

    def to_dict(self) -> dict:
        """Convert to kwargs accepted by retriever.build_filter()."""
        d: dict = {}
        if self.genres:
            d["genres"] = self.genres
        if self.year is not None:
            d["year"] = self.year
        if self.year_min is not None:
            d["year_min"] = self.year_min
        if self.year_max is not None:
            d["year_max"] = self.year_max
        if self.format_ is not None:
            d["format_"] = self.format_
        if self.score_min is not None:
            d["score_min"] = self.score_min
        if self.is_adult is not None:
            d["is_adult"] = self.is_adult
        return d

    def is_empty(self) -> bool:
        return not self.to_dict()


_FILTER_EXTRACTION_PROMPT = """\
You are a metadata extractor for an anime database.
Extract structured filter parameters from the user's query.

Available genres: Action, Adventure, Comedy, Drama, Fantasy, Horror, \
Mystery, Psychological, Romance, Sci-Fi, Slice of Life, Sports, \
Supernatural, Thriller, Mecha, Music, Ecchi, Harem, Isekai

Available formats: TV, MOVIE, OVA, ONA, SPECIAL

Rules:
- year: exact year mentioned (e.g. 2023). If a range, use year_min/year_max.
- score_min: minimum score as integer 0-100 (e.g. "above 8/10" → 80).
- Only extract what is explicitly mentioned. Leave fields null if uncertain.
- Return ONLY valid JSON. No explanation.

Examples:
Query: "best action anime from 2023"
{"genres": ["Action"], "year": 2023}

Query: "highly rated romance movies"
{"genres": ["Romance"], "format": "MOVIE", "score_min": 75}

Query: "sci-fi anime between 2018 and 2022"
{"genres": ["Sci-Fi"], "year_min": 2018, "year_max": 2022}

Query: "recommend me something good"
{}

Query: "top rated anime overall"
{"score_min": 80}

Now extract from this query (return JSON only):
"""

_FORMAT_ALIASES: dict[str, str] = {
    "movie": "MOVIE", "film": "MOVIE", "movies": "MOVIE",
    "ova": "OVA", "ovas": "OVA",
    "ona": "ONA", "web": "ONA",
    "special": "SPECIAL", "specials": "SPECIAL",
    "tv": "TV", "series": "TV", "show": "TV", "shows": "TV",
}


async def extract_filters(query: str, oai_client: AsyncOpenAI) -> FilterParams:
    """Use GPT-4o-mini to extract structured metadata filters from a query.

    Returns an empty FilterParams if extraction fails or query has no filters.
    """
    try:
        resp = await oai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "user", "content": f"{_FILTER_EXTRACTION_PROMPT}{query}"},
            ],
            max_tokens=120,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Filter extraction failed ({exc}), skipping filters.", exc=exc)
        return FilterParams()

    # Normalise format field
    fmt_raw: str | None = raw.get("format")
    fmt_norm: str | None = None
    if fmt_raw:
        fmt_norm = _FORMAT_ALIASES.get(fmt_raw.lower(), fmt_raw.upper())

    params = FilterParams(
        genres=raw.get("genres") or [],
        year=raw.get("year"),
        year_min=raw.get("year_min"),
        year_max=raw.get("year_max"),
        format_=fmt_norm,
        score_min=raw.get("score_min"),
    )
    if not params.is_empty():
        logger.debug("Extracted filters: {f}", f=params.to_dict())
    return params


# ══════════════════════════════════════════════════════════════════════════════
# Hour 6-7 — Query Rewriting (few-shot)
# ══════════════════════════════════════════════════════════════════════════════

_REWRITE_SYSTEM = """\
You are a search query optimizer for an anime/manga vector database.
Rewrite the user's question into a concise, keyword-rich retrieval query.

Rules:
- Preserve specific titles, character names, genres, years if mentioned.
- Expand vague terms (e.g. "something exciting" → "high action intense battles").
- Remove filler words ("can you", "please", "I want to find").
- Output ONLY the rewritten query — no quotes, no explanation.

Examples:
User: "What are some good romance anime from the 2010s?"
Rewritten: romance anime 2010s love story emotional

User: "I want something similar to Attack on Titan"
Rewritten: dark intense action survival military humanity threat

User: "best anime movie ever made"
Rewritten: highest rated anime movie masterpiece

User: "recommend me isekai with overpowered main character"
Rewritten: isekai overpowered protagonist reincarnation fantasy

User: "anime like Spirited Away but darker"
Rewritten: dark fantasy supernatural surreal coming-of-age spirit world
"""


async def rewrite_query(query: str, oai_client: AsyncOpenAI) -> str:
    """Few-shot query rewriting for improved retrieval recall.

    Returns the rewritten query, or the original on failure.
    """
    try:
        resp = await oai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        if rewritten and rewritten.lower() != query.lower():
            logger.debug("Rewrite: {orig!r} → {new!r}", orig=query, new=rewritten)
            return rewritten
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed ({exc}), using original.", exc=exc)
    return query


# ══════════════════════════════════════════════════════════════════════════════
# Context assembly — numbered citations [1]..[N]
# ══════════════════════════════════════════════════════════════════════════════

def _build_context(docs: list[RetrievedDoc]) -> str:
    """Format reranked docs as a numbered context block with citation anchors.

    Each passage is prefixed with [N] so the LLM can cite them inline.
    """
    sections: list[str] = []

    for i, doc in enumerate(docs, 1):
        title = doc.title
        payload = doc.payload

        # Score
        score_raw = payload.get("score") or payload.get("average_score")
        score_str = f"{score_raw / 10:.1f}/10" if score_raw else "N/A"

        # Year + format
        year = payload.get("year") or payload.get("season_year")
        fmt = payload.get("format", "")
        meta_parts = [str(year) if year else "", fmt]
        meta_str = ", ".join(p for p in meta_parts if p)

        # Genres
        genres = ", ".join(payload.get("genres") or [])

        # Synopsis — strip HTML, truncate
        full = payload.get("full_data") or {}
        desc_raw = full.get("description") or payload.get("description") or ""
        desc = re.sub(r"<[^>]+>", " ", desc_raw)
        desc = html_lib.unescape(desc)
        desc = re.sub(r"\s+", " ", desc).strip()[:500]

        # Build passage
        header_parts = [f"[{i}] **{title}**"]
        if meta_str:
            header_parts.append(f"({meta_str})")
        header_parts.append(f"| Score: {score_str}")
        if genres:
            header_parts.append(f"| Genres: {genres}")

        passage = " ".join(header_parts)
        if desc:
            passage += f"\n{desc}"

        sections.append(passage)

    return "\n\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
# Main RAG function
# ══════════════════════════════════════════════════════════════════════════════

async def rag_answer(
    query: str,
    oai_client: AsyncOpenAI,
    *,
    filter_kwargs: dict | None = None,
    stream: bool = False,
    rewrite: bool = True,
    auto_filter: bool = True,
) -> str | AsyncIterator[str]:
    """Full RAG pipeline: extract filters → rewrite → retrieve → rerank → generate.

    Args:
        query:          Raw user query string.
        oai_client:     AsyncOpenAI client (ShopAIKey, pre-configured).
        filter_kwargs:  Explicit metadata filters (skips auto-extraction if set).
                        Example: {"genres": ["Action"], "year": 2023}
        stream:         If True, returns an AsyncIterator[str] of text chunks.
        rewrite:        If True (default), rewrites query before retrieval.
        auto_filter:    If True (default) and filter_kwargs is None, auto-extract
                        metadata filters from the query using GPT-4o-mini.

    Returns:
        str if stream=False, AsyncIterator[str] if stream=True.
    """
    # ── Step 1: Auto-extract metadata filters ─────────────────────────────────
    if filter_kwargs is None and auto_filter:
        extracted = await extract_filters(query, oai_client)
        filter_kwargs = extracted.to_dict() if not extracted.is_empty() else None

    # ── Step 2: Rewrite query for retrieval ───────────────────────────────────
    retrieval_query = await rewrite_query(query, oai_client) if rewrite else query

    # ── Step 3: Retrieve top-20 candidates ────────────────────────────────────
    candidates: list[RetrievedDoc] = await retrieve(
        retrieval_query,
        oai_client,
        top_k=settings.retriever_top_k,   # 20
        filter_kwargs=filter_kwargs,
    )

    if not candidates:
        no_result = (
            "I couldn't find any relevant anime for your query. "
            "Try rephrasing, or remove filters (e.g. a very specific year/genre combo)."
        )
        if stream:
            async def _empty() -> AsyncIterator[str]:
                yield no_result
            return _empty()
        return no_result

    # ── Step 4: Rerank top-20 → top-5 ────────────────────────────────────────
    doc_texts = [doc.chunk_text for doc in candidates]
    ranked = await rerank(
        query=retrieval_query,
        documents=doc_texts,
        top_k=settings.reranker_top_k,   # 5
    )
    top_docs = [candidates[r["index"]] for r in ranked]

    logger.info(
        "RAG: retrieve={total} → rerank={k} | top: {title!r} (score={s:.3f})",
        total=len(candidates),
        k=len(top_docs),
        title=top_docs[0].title if top_docs else "N/A",
        s=ranked[0]["relevance_score"] if ranked else 0,
    )

    # ── Step 5: Assemble context + build messages ──────────────────────────────
    # Message structure designed for OpenAI prefix caching:
    #   [SYSTEM]  static role + rules + few-shot  ← cached after first call
    #   [USER]    context passages + question      ← dynamic per request
    context = _build_context(top_docs)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\n"
        f"User question: {query}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    # ── Step 6: Generate answer ────────────────────────────────────────────────
    if stream:
        async def _stream_answer() -> AsyncIterator[str]:
            response = await oai_client.chat.completions.create(
                model=settings.openai_model,
                messages=cast(Any, messages),  # dict form is valid at runtime
                temperature=0.3,
                max_tokens=900,
                stream=True,
            )
            # Narrow union: stream=True guarantees AsyncStream, not ChatCompletion
            async for chunk in cast(Any, response):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _stream_answer()

    resp = await oai_client.chat.completions.create(
        model=settings.openai_model,
        messages=cast(Any, messages),  # dict form is valid at runtime
        temperature=0.3,
        max_tokens=900,
    )
    answer = resp.choices[0].message.content or ""
    logger.debug("Answer generated ({n} chars)", n=len(answer))
    return answer
