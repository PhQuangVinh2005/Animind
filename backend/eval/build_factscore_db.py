"""Build FActScore knowledge source from Qdrant anime collection.

Exports ALL anime entries (scroll API, no query vector needed) to:
  eval/factscore_db/anime_kb.jsonl  — JSONL knowledge source
  eval/factscore_db/anime_kb.db     — SQLite DB with FTS5 for BM25 retrieval

Text format per entry (structured, Format C):
  title: Fullmetal Alchemist: Brotherhood
  english_title: Fullmetal Alchemist: Brotherhood
  native_title: 鋼の錬金術師 FULLMETAL ALCHEMIST
  year: 2009
  format: TV
  score: 90
  episodes: 64
  genres: Action, Adventure, Drama, Fantasy
  studios: bones
  status: FINISHED
  description: In order for something to be obtained...

Usage (run once, animind env):
    cd backend
    conda run -n animind python eval/build_factscore_db.py
"""

from __future__ import annotations

from typing import Any

import html as html_lib
import json
import re
import sqlite3
import sys
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))
from app.config import settings  # noqa: E402

EVAL_DIR = Path(__file__).parent
DB_DIR = EVAL_DIR / "factscore_db"
JSONL_PATH = DB_DIR / "anime_kb.jsonl"
SQLITE_PATH = DB_DIR / "anime_kb.db"


# ── Format structured text per anime ─────────────────────────────────────────

def _format_anime_text(payload: dict) -> str:
    """Produce structured text (Format C) for one anime payload."""
    title_info = payload.get("title") or {}
    preferred = (
        title_info.get("preferred") if isinstance(title_info, dict)
        else str(title_info)
    ) or ""
    english  = title_info.get("english", "")  if isinstance(title_info, dict) else ""
    native   = title_info.get("native", "")   if isinstance(title_info, dict) else ""

    year     = payload.get("year") or payload.get("season_year") or ""
    fmt      = payload.get("format") or ""
    score    = payload.get("score")           # raw int 0–100
    episodes = payload.get("episodes") or ""
    genres   = ", ".join(payload.get("genres") or [])
    studios  = ", ".join(payload.get("studios") or [])
    status   = payload.get("status") or ""

    full     = payload.get("full_data") or {}
    desc_raw = full.get("description") or payload.get("description") or ""
    desc = re.sub(r"<[^>]+>", " ", desc_raw)
    desc = html_lib.unescape(desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    lines: list[str] = []
    if preferred:
        lines.append(f"title: {preferred}")
    if english and english != preferred:
        lines.append(f"english_title: {english}")
    if native:
        lines.append(f"native_title: {native}")
    if year:
        lines.append(f"year: {year}")
    if fmt:
        lines.append(f"format: {fmt}")
    if score is not None:
        score_str = f"{score}/100 ({score / 10:.1f}/10)"
        lines.append(f"score: {score_str}")
    if episodes:
        lines.append(f"episodes: {episodes}")
    if genres:
        lines.append(f"genres: {genres}")
    if studios:
        lines.append(f"studios: {studios}")
    if status:
        lines.append(f"status: {status}")
    if desc:
        lines.append(f"description: {desc}")
    return "\n".join(lines)


def _get_display_title(payload: dict) -> str:
    """Return the best English-friendly title for KB lookup."""
    title_info = payload.get("title") or {}
    if isinstance(title_info, dict):
        return (
            title_info.get("english")
            or title_info.get("preferred")
            or title_info.get("native")
            or ""
        )
    return str(title_info)


# ── Scroll all Qdrant points ──────────────────────────────────────────────────

def _scroll_all(collection: str) -> list[Any]:
    """Scroll ALL points from Qdrant (no query vector, no limit)."""
    from urllib.parse import urlparse
    parsed = urlparse(settings.qdrant_url)
    client = QdrantClient(host=parsed.hostname or "localhost", port=parsed.port or 6333)
    records, next_offset = client.scroll(
        collection_name=collection,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    all_records = list(records)
    while next_offset is not None:
        records, next_offset = client.scroll(
            collection_name=collection,
            offset=next_offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        all_records.extend(records)
    return all_records


# ── Build SQLite FTS5 DB ──────────────────────────────────────────────────────

def _build_sqlite(entries: list[dict], db_path: Path) -> None:
    """Create SQLite DB with FTS5 virtual table for BM25 retrieval."""
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE documents (
            rowid  INTEGER PRIMARY KEY AUTOINCREMENT,
            title  TEXT NOT NULL,
            text   TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts
        USING fts5(title, text, content=documents, content_rowid=rowid);
    """)

    for entry in entries:
        conn.execute(
            "INSERT INTO documents (title, text) VALUES (?, ?)",
            (entry["title"], entry["text"]),
        )

    # Populate FTS5 index
    conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    logger.info("SQLite FTS5 DB built → {}", db_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    collection = settings.qdrant_collection

    logger.info("Scrolling all points from Qdrant collection '{}'", collection)
    records = _scroll_all(collection)
    logger.info("Retrieved {} anime entries", len(records))

    entries: list[dict] = []
    for rec in records:
        payload = rec.payload or {}
        title = _get_display_title(payload)
        text  = _format_anime_text(payload)
        if title and text:
            entries.append({"title": title, "text": text})

    # Write JSONL
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("JSONL written → {} ({} entries)", JSONL_PATH, len(entries))

    # Build SQLite
    _build_sqlite(entries, SQLITE_PATH)
    logger.info("Done. Knowledge source ready at {}", DB_DIR)


if __name__ == "__main__":
    main()
