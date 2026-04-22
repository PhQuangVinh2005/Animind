"""Fetch anime/manga data from AniList GraphQL API.

Usage:
    cd backend
    python scripts/fetch_anilist.py            # normal run / resume
    python scripts/fetch_anilist.py --reset    # start fresh, ignore checkpoint

Output:
    data/raw/anime.json             — final cleaned records (~20k)
    data/raw/.fetch_checkpoint.json — resume checkpoint (auto-deleted on finish)

Interrupt-safe:
    Ctrl+C writes all records fetched so far to anime.json and saves checkpoint.
    Re-running resumes from the next page automatically.

AniList auth (https://docs.anilist.co/guide/auth/):
    Public anime data requires NO authentication.
    Set ANILIST_ACCESS_TOKEN in .env to enable user-specific queries.

Rate limiting (https://docs.anilist.co/guide/rate-limiting):
    Current degraded limit: 30 req/min. We read X-RateLimit-Remaining
    per response and throttle proactively.
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "anime.json"
CHECKPOINT_PATH = DATA_DIR / ".fetch_checkpoint.json"

# ── Tuning ────────────────────────────────────────────────────────────────────

PAGE_SIZE = 50
REQUEST_DELAY = 2.1           # 30 req/min = 2.0s min gap; +0.1s buffer
RATE_LIMIT_THRESHOLD = 5      # proactive throttle when remaining ≤ this
THROTTLE_SLEEP = 12.0         # seconds to sleep when near rate limit
RETRY_SLEEP = 65.0            # sleep after 429
MAX_RETRIES = 3
SAVE_EVERY_N_PAGES = 5        # checkpoint flush frequency

# ── GraphQL Query — all RAG-useful fields ────────────────────────────────────

ANIME_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC) {

      # ── Identity ──────────────────────────────────────────────────────────
      id
      idMal
      title {
        romaji
        english
        native
        userPreferred
      }
      synonyms

      # ── Content & Classification ──────────────────────────────────────────
      description(asHtml: false)
      genres
      tags {
        name
        rank
        isGeneralSpoiler
        isMediaSpoiler
        isAdult
        category
      }
      source
      countryOfOrigin
      isAdult

      # ── Airing & Format ───────────────────────────────────────────────────
      format
      status
      episodes
      duration
      season
      seasonYear
      startDate { year month day }
      endDate   { year month day }
      nextAiringEpisode {
        episode
        airingAt
        timeUntilAiring
      }

      # ── Scores & Popularity ───────────────────────────────────────────────
      averageScore
      meanScore
      popularity
      favourites
      trending

      # ── Rankings ──────────────────────────────────────────────────────────
      rankings {
        rank
        type
        format
        year
        season
        allTime
        context
      }

      # ── Production ────────────────────────────────────────────────────────
      studios(isMain: true) {
        nodes { name siteUrl }
      }

      # ── External & Streaming ──────────────────────────────────────────────
      externalLinks {
        url
        site
        type
        language
      }

      # ── Assets ────────────────────────────────────────────────────────────
      coverImage { extraLarge large medium color }
      bannerImage
      trailer  { id site thumbnail }

      # ── Site ──────────────────────────────────────────────────────────────
      siteUrl
      updatedAt
    }
  }
}
"""


# ── Auth ──────────────────────────────────────────────────────────────────────

def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.anilist_access_token:
        headers["Authorization"] = f"Bearer {settings.anilist_access_token}"
        logger.debug("AniList: authenticated (Bearer token)")
    else:
        logger.debug("AniList: unauthenticated (public data only)")
    return headers


# ── Record cleaner ────────────────────────────────────────────────────────────

def _clean_record(media: dict) -> dict:
    """Flatten and clean all RAG-useful fields from a raw AniList media record."""

    # ── Titles ────────────────────────────────────────────────────────────────
    title = media.get("title", {}) or {}

    # ── Dates ─────────────────────────────────────────────────────────────────
    start = media.get("startDate", {}) or {}
    end   = media.get("endDate",   {}) or {}

    # ── Studios ───────────────────────────────────────────────────────────────
    studios_raw = (media.get("studios", {}) or {}).get("nodes", [])
    studios = [
        {"name": s["name"], "site_url": s.get("siteUrl")}
        for s in studios_raw if s.get("name")
    ]

    # ── Tags: top-15 non-spoiler, with category ───────────────────────────────
    raw_tags = media.get("tags") or []
    tags = [
        {
            "name": t["name"],
            "rank": t.get("rank"),
            "category": t.get("category"),
        }
        for t in sorted(raw_tags, key=lambda x: -(x.get("rank") or 0))
        if not t.get("isMediaSpoiler") and not t.get("isGeneralSpoiler")
    ][:15]

    # ── Rankings ──────────────────────────────────────────────────────────────
    raw_rankings = media.get("rankings") or []
    rankings = [
        {
            "rank": r.get("rank"),
            "type": r.get("type"),           # RATED | POPULAR
            "format": r.get("format"),
            "year": r.get("year"),           # None = all time
            "season": r.get("season"),
            "all_time": r.get("allTime"),
            "context": r.get("context"),     # "highest rated all time" etc.
        }
        for r in raw_rankings
    ]

    # ── External links: streaming only (+ keep social/info) ──────────────────
    raw_links = media.get("externalLinks") or []
    streaming_links = [
        {"site": lk.get("site"), "url": lk.get("url"), "language": lk.get("language")}
        for lk in raw_links
        if lk.get("type") == "STREAMING"
    ]

    # ── Cover images ──────────────────────────────────────────────────────────
    cover = media.get("coverImage", {}) or {}

    # ── Next airing ───────────────────────────────────────────────────────────
    airing = media.get("nextAiringEpisode") or {}

    # ── Trailer ───────────────────────────────────────────────────────────────
    trailer = media.get("trailer") or {}

    return {
        # Identity
        "id":               media["id"],
        "id_mal":           media.get("idMal"),           # MyAnimeList ID
        "title_romaji":     title.get("romaji"),
        "title_english":    title.get("english"),
        "title_native":     title.get("native"),
        "title_preferred":  title.get("userPreferred"),
        "synonyms":         media.get("synonyms") or [],  # alternative titles

        # Content
        "description":      (media.get("description") or "").strip(),
        "genres":           media.get("genres") or [],
        "tags":             tags,                         # [{name, rank, category}]
        "source":           media.get("source"),          # MANGA, LIGHT_NOVEL, ORIGINAL …
        "country_of_origin": media.get("countryOfOrigin"),  # JP, KR, CN, TW …
        "is_adult":         media.get("isAdult", False),

        # Format & Airing
        "format":           media.get("format"),          # TV, MOVIE, OVA, ONA …
        "status":           media.get("status"),          # FINISHED, RELEASING, …
        "episodes":         media.get("episodes"),
        "duration":         media.get("duration"),        # minutes per episode
        "season":           media.get("season"),
        "season_year":      media.get("seasonYear"),
        "start_year":       start.get("year"),
        "start_month":      start.get("month"),
        "start_day":        start.get("day"),
        "end_year":         end.get("year"),
        "end_month":        end.get("month"),
        "end_day":          end.get("day"),
        "next_airing_episode": airing.get("episode"),
        "next_airing_at":  airing.get("airingAt"),       # Unix timestamp

        # Scores
        "average_score":    media.get("averageScore"),    # 0-100
        "mean_score":       media.get("meanScore"),
        "popularity":       media.get("popularity"),      # # of users with in list
        "favourites":       media.get("favourites"),
        "trending":         media.get("trending"),        # current trending score

        # Rankings
        "rankings":         rankings,

        # Production
        "studios":          [s["name"] for s in studios],
        "studio_urls":      [s["site_url"] for s in studios if s.get("site_url")],

        # Streaming
        "streaming_links":  streaming_links,              # [{site, url, language}]

        # Assets
        "cover_image":      cover.get("extraLarge") or cover.get("large"),
        "cover_color":      cover.get("color"),           # dominant hex color
        "banner_image":     media.get("bannerImage"),
        "trailer_site":     trailer.get("site"),          # youtube / dailymotion
        "trailer_id":       trailer.get("id"),            # video ID on that site
        "trailer_thumbnail": trailer.get("thumbnail"),

        # Links
        "site_url":         media.get("siteUrl"),
        "updated_at":       media.get("updatedAt"),       # Unix timestamp
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[list[dict], int]:
    if not CHECKPOINT_PATH.exists():
        return [], 1
    try:
        ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        last_page = ckpt.get("last_page", 0)
        records = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else []
        next_page = last_page + 1
        logger.info("Resuming from page {next} ({n} records saved).", next=next_page, n=len(records))
        return records, next_page
    except Exception as exc:
        logger.warning("Bad checkpoint ({exc}), starting fresh.", exc=exc)
        return [], 1


def _save_checkpoint(records: list[dict], last_page: int, total_pages: int | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUTPUT_PATH)
    CHECKPOINT_PATH.write_text(
        json.dumps({"last_page": last_page, "total_pages": total_pages, "total_records": len(records)}),
        encoding="utf-8",
    )


def _finish(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUTPUT_PATH)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ── Page fetcher ──────────────────────────────────────────────────────────────

async def _fetch_page(client: httpx.AsyncClient, page: int) -> tuple[dict, dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(
                settings.anilist_api_url,
                json={"query": ANIME_QUERY, "variables": {"page": page, "perPage": PAGE_SIZE}},
            )

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", RETRY_SLEEP))
                sleep_time = retry_after + 5.0
                logger.warning("429 page {page} (attempt {a}/{m}). Sleeping {s:.0f}s …",
                               page=page, a=attempt, m=MAX_RETRIES, s=sleep_time)
                await asyncio.sleep(sleep_time)
                continue

            resp.raise_for_status()
            payload = resp.json()

            if "errors" in payload:
                errors = payload["errors"]
                is_rate = any(
                    e.get("status") == 429 or "Too Many" in str(e.get("message", ""))
                    for e in errors
                )
                if is_rate:
                    logger.warning("GraphQL rate-limit page {page} (attempt {a}/{m}). Sleeping {s:.0f}s …",
                                   page=page, a=attempt, m=MAX_RETRIES, s=RETRY_SLEEP)
                    await asyncio.sleep(RETRY_SLEEP)
                    continue
                raise RuntimeError(f"GraphQL errors page {page}: {errors}")

            return payload["data"]["Page"], dict(resp.headers)

        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning("Page {page} error (attempt {a}/{m}): {exc}. Retry in {s:.0f}s …",
                           page=page, a=attempt, m=MAX_RETRIES, exc=exc, s=RETRY_SLEEP)
            await asyncio.sleep(RETRY_SLEEP)

    raise RuntimeError(f"Failed page {page} after {MAX_RETRIES} attempts")


# ── Main fetch loop ───────────────────────────────────────────────────────────

async def fetch_all(reset: bool = False) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        records: list[dict] = []
        start_page = 1
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        logger.info("--reset: starting from page 1.")
    else:
        records, start_page = _load_checkpoint()

    total_pages: int | None = None
    interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n\n⚠  Interrupted! Saving {len(records):,} records …", file=sys.stderr)
        _save_checkpoint(records, start_page - 1, total_pages)
        print(f"✓  Saved to {OUTPUT_PATH}", file=sys.stderr)
        print(f"   Resume with:  python scripts/fetch_anilist.py", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    page = start_page

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=_build_headers()) as client:
        while True:
            t0 = time.monotonic()

            logger.info("Fetching page {page}{of} …",
                        page=page, of=f"/{total_pages}" if total_pages else "")

            page_data, resp_headers = await _fetch_page(client, page)
            page_info = page_data["pageInfo"]

            remaining = int(resp_headers.get("x-ratelimit-remaining", 99))
            limit      = int(resp_headers.get("x-ratelimit-limit", 30))

            if total_pages is None:
                total_pages = page_info["lastPage"]
                logger.info("Total pages: {total} (~{records:,} records) | Rate limit: {limit}/min",
                            total=total_pages, records=page_info["total"], limit=limit)

            for media in page_data["media"]:
                records.append(_clean_record(media))

            logger.info("Page {page}/{total} ✓ — {n:,} records | RateLimit: {r}/{lim}",
                        page=page, total=total_pages, n=len(records), r=remaining, lim=limit)

            if page % SAVE_EVERY_N_PAGES == 0:
                _save_checkpoint(records, page, total_pages)
                logger.debug("Checkpoint saved (page {page}).", page=page)

            if not page_info["hasNextPage"]:
                break

            page += 1

            if remaining <= RATE_LIMIT_THRESHOLD:
                logger.warning("Near rate limit ({r} remaining) — throttling {s:.0f}s …",
                               r=remaining, s=THROTTLE_SLEEP)
                await asyncio.sleep(THROTTLE_SLEEP)
            else:
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, REQUEST_DELAY - elapsed))

    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AniList anime data.")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore checkpoint and start from page 1.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AniList fetch — interrupt-safe with checkpoint resume")
    logger.info("Fields: identity, content, format, scores, rankings, streaming, assets")
    logger.info("Output : {}", OUTPUT_PATH)
    if CHECKPOINT_PATH.exists() and not args.reset:
        ckpt = json.loads(CHECKPOINT_PATH.read_text())
        logger.info("Resuming from page {} ({:,} records saved)",
                    ckpt["last_page"] + 1, ckpt["total_records"])
    logger.info("=" * 60)

    t_start = time.monotonic()
    records = asyncio.run(fetch_all(reset=args.reset))
    _finish(records)

    elapsed = time.monotonic() - t_start
    logger.success(
        "Done. {n:,} records → {path}  ({elapsed:.1f}s, {rate:.1f} rec/s)",
        n=len(records), path=OUTPUT_PATH, elapsed=elapsed,
        rate=len(records) / max(elapsed, 1),
    )


if __name__ == "__main__":
    main()
