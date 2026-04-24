#!/usr/bin/env python3
"""Interactive .env setup for AniMind.

Usage:
    python setup_env.py

Prompts for:
    - ShopAIKey API key (used for both GPT chat and embeddings)
    - AniList Client ID + Client Secret (+ optional access token)

Writes values to backend/.env (creates from .env.example if missing).
Never overwrites a key that already has a real value unless you confirm.
"""

import re
import shutil
import sys
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (one level up from scripts/)
ENV_FILE = ROOT / "backend" / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

# ── ANSI colours ─────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"

def header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 56}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 56}{RESET}\n")

def info(text: str)    -> None: print(f"  {CYAN}ℹ{RESET}  {text}")
def ok(text: str)      -> None: print(f"  {GREEN}✓{RESET}  {text}")
def warn(text: str)    -> None: print(f"  {YELLOW}⚠{RESET}  {text}")
def err(text: str)     -> None: print(f"  {RED}✗{RESET}  {text}")

def prompt_secret(label: str, hint: str = "") -> str:
    """Prompt for a secret value (hidden input). Empty = skip."""
    display = f"{BOLD}{label}{RESET}"
    if hint:
        display += f" {YELLOW}({hint}){RESET}"
    display += ": "
    return getpass(display).strip()

def prompt_plain(label: str, default: str = "", hint: str = "") -> str:
    """Prompt for a plain (visible) value."""
    display = f"{BOLD}{label}{RESET}"
    if hint:
        display += f" {YELLOW}({hint}){RESET}"
    if default:
        display += f" [{default}]"
    display += ": "
    val = input(display).strip()
    return val if val else default

# ── .env file helpers ─────────────────────────────────────────────────────────

def load_env(path: Path) -> dict[str, str]:
    """Parse an .env file into a dict. Ignores comments and blank lines."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def update_env(path: Path, updates: dict[str, str]) -> None:
    """Write/update keys in an .env file, preserving existing lines and comments."""
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    handled: set[str] = set()

    # Update in-place where key already exists
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                handled.add(key)
                continue
        new_lines.append(line)

    # Append any keys that weren't already in the file
    for key, val in updates.items():
        if key not in handled:
            new_lines.append(f"{key}={val}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def is_placeholder(val: str) -> bool:
    """Return True if the value looks like an unfilled placeholder."""
    if not val:
        return True
    placeholders = {"your-key-here", "your-client-id", "your-client-secret",
                    "sk-your-key-here", ""}
    return val in placeholders or val.startswith("your-")


# ── Credential collectors ─────────────────────────────────────────────────────

def collect_shopaikey(current: dict[str, str]) -> dict[str, str]:
    """Prompt for ShopAIKey credentials (used for both chat and embeddings)."""
    header("ShopAIKey — OpenAI-compatible API")
    info("Used for: GPT-4o-mini chat + text-embedding-3-small embeddings")
    info("Register: https://shopaikey.com  →  Dashboard → API Keys")
    info("Base URL : https://api.shopaikey.com/v1")
    print()

    updates: dict[str, str] = {}

    # API key
    existing_key = current.get("SHOPAIKEY_API_KEY", "")
    if not is_placeholder(existing_key):
        warn(f"SHOPAIKEY_API_KEY already set ({existing_key[:8]}…). Press Enter to keep.")
    val = prompt_secret("SHOPAIKEY_API_KEY", hint="sk-…")
    if val:
        updates["SHOPAIKEY_API_KEY"] = val
        ok("SHOPAIKEY_API_KEY saved.")
    else:
        if not is_placeholder(existing_key):
            ok("SHOPAIKEY_API_KEY kept (unchanged).")
        else:
            warn("SHOPAIKEY_API_KEY skipped — set it later in backend/.env")

    # Base URL (fixed — but allow override)
    default_url = current.get("SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")
    url = prompt_plain("SHOPAIKEY_BASE_URL", default=default_url, hint="press Enter to keep default")
    updates["SHOPAIKEY_BASE_URL"] = url
    ok(f"SHOPAIKEY_BASE_URL = {url}")

    # Models (show current, allow override)
    model = prompt_plain(
        "OPENAI_MODEL",
        default=current.get("OPENAI_MODEL", "gpt-4o-mini"),
        hint="chat model name",
    )
    updates["OPENAI_MODEL"] = model

    embed = prompt_plain(
        "OPENAI_EMBEDDING_MODEL",
        default=current.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        hint="embedding model name",
    )
    updates["OPENAI_EMBEDDING_MODEL"] = embed

    return updates


def collect_anilist(current: dict[str, str]) -> dict[str, str]:
    """Prompt for AniList OAuth2 credentials."""
    header("AniList OAuth2 Credentials")
    info("Register app: https://anilist.co/settings/developer")
    info("Public anime data does NOT require auth — but Client ID/Secret")
    info("are needed for user-specific features (watchlists, mutations).")
    print()

    updates: dict[str, str] = {}

    # Client ID
    existing_id = current.get("ANILIST_CLIENT_ID", "")
    if not is_placeholder(existing_id):
        warn(f"ANILIST_CLIENT_ID already set ({existing_id}). Press Enter to keep.")
    val = prompt_plain("ANILIST_CLIENT_ID", hint="numeric app ID from settings")
    if val:
        updates["ANILIST_CLIENT_ID"] = val
        ok("ANILIST_CLIENT_ID saved.")
    else:
        if not is_placeholder(existing_id):
            ok("ANILIST_CLIENT_ID kept (unchanged).")
        else:
            warn("ANILIST_CLIENT_ID skipped.")

    # Client Secret
    existing_secret = current.get("ANILIST_CLIENT_SECRET", "")
    if not is_placeholder(existing_secret):
        warn(f"ANILIST_CLIENT_SECRET already set ({existing_secret[:6]}…). Press Enter to keep.")
    val = prompt_secret("ANILIST_CLIENT_SECRET", hint="from developer settings page")
    if val:
        updates["ANILIST_CLIENT_SECRET"] = val
        ok("ANILIST_CLIENT_SECRET saved.")
    else:
        if not is_placeholder(existing_secret):
            ok("ANILIST_CLIENT_SECRET kept (unchanged).")
        else:
            warn("ANILIST_CLIENT_SECRET skipped.")

    # Access Token (optional — only for authenticated queries)
    print()
    info("ANILIST_ACCESS_TOKEN is optional. Leave blank for public-only access.")
    info("Get one via: https://docs.anilist.co/guide/auth/authorization-code")
    existing_token = current.get("ANILIST_ACCESS_TOKEN", "")
    if not is_placeholder(existing_token):
        warn(f"ANILIST_ACCESS_TOKEN already set ({existing_token[:12]}…). Press Enter to keep.")
    val = prompt_secret("ANILIST_ACCESS_TOKEN", hint="optional JWT, press Enter to skip")
    if val:
        updates["ANILIST_ACCESS_TOKEN"] = val
        ok("ANILIST_ACCESS_TOKEN saved.")
    else:
        if not is_placeholder(existing_token):
            ok("ANILIST_ACCESS_TOKEN kept (unchanged).")
        else:
            updates["ANILIST_ACCESS_TOKEN"] = ""
            info("ANILIST_ACCESS_TOKEN left blank (unauthenticated mode).")

    return updates


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'═' * 56}")
    print("  AniMind — Environment Setup")
    print(f"{'═' * 56}{RESET}")
    print(f"  Writing to: {BOLD}{ENV_FILE}{RESET}\n")

    # Bootstrap .env from .env.example if missing
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
            ok(f"Created {ENV_FILE} from .env.example")
        else:
            ENV_FILE.touch()
            ok(f"Created empty {ENV_FILE}")

    current = load_env(ENV_FILE)
    all_updates: dict[str, str] = {}

    # Section 1: ShopAIKey
    all_updates.update(collect_shopaikey(current))

    # Section 2: AniList
    all_updates.update(collect_anilist(current))

    # Write everything
    update_env(ENV_FILE, all_updates)

    header("Setup Complete")
    ok(f"Saved {len(all_updates)} values to {ENV_FILE}")
    print()
    info("Next steps:")
    print(f"    {CYAN}1.{RESET} cd backend && pip install -r requirements.txt")
    print(f"    {CYAN}2.{RESET} python scripts/fetch_anilist.py   # ~14 min at 30 req/min")
    print(f"    {CYAN}3.{RESET} python scripts/ingest.py --limit 100  # test first")
    print(f"    {CYAN}4.{RESET} python scripts/ingest.py          # full ingest")
    print()
    warn(".env is in .gitignore — never commit it.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted — partial values may have been saved.{RESET}\n")
        sys.exit(1)
