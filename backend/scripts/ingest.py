"""Ingest anime data into Qdrant vector store.

Usage:
    python scripts/ingest.py

Reads:
    data/raw/anime.json

Process:
    1. Load raw JSON
    2. Build documents (title + synopsis + metadata per anime)
    3. Embed with text-embedding-3-small
    4. Upsert into Qdrant collection "anime"
"""

# TODO Day 1: Implement ingestion pipeline
# - Load data/raw/anime.json
# - Create document text: f"{title}\n\n{description}"
# - Metadata payload: genres, score, year, format, episodes, status
# - Embed via OpenAI text-embedding-3-small (batch)
# - Upsert to Qdrant collection with payload
