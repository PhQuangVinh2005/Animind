"""Fetch anime/manga data from AniList GraphQL API.

Usage:
    python scripts/fetch_anilist.py

Output:
    data/raw/anime.json — raw API response (~20k records)
"""

# TODO Day 1: Implement AniList GraphQL pagination
# - Query: id, title (romaji, english, native), description, genres,
#           averageScore, episodes, format, season, seasonYear, status
# - Pagination: 50 records/page, ~400 requests for 20k records
# - Rate limit: 90 req/min (AniList) — add 0.7s delay between requests
# - Save raw JSON to data/raw/anime.json
