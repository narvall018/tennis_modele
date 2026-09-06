"""Live prices from The Odds API, with the quota treated as the scarce thing it is.

The free tier allows 500 requests a month. That is plenty for a person checking
a card before it starts and nowhere near enough to poll on every page render, so
every call here is deliberate: results are cached, the remaining allowance is
read back from the response headers and surfaced, and nothing fetches unless a
caller explicitly asks for it.

The key is resolved from, in order: the ``ODDS_API_KEY`` environment variable,
Streamlit secrets, and finally the encoded key already embedded in ``app.py``.
That last fallback exists because the key is the user's own and already lives in
this repository; it is never printed, only used.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "eu,uk"
DEFAULT_MARKET = "h2h"

TENNIS_SPORT_PREFIX = "tennis_"
MMA_SPORT_KEY = "mma_mixed_martial_arts"

# Some books quote MMA three-way, adding a draw at around 33.0. A draw is rare
# and most operators void the bet, so the market modelled here is the two
# fighters. Dropping the leg is what lets a three-way quote be compared with a
# two-way one at all; keeping it would silently discard every such event.
IGNORED_OUTCOMES = frozenset({"draw", "tie"})


@dataclass
class OddsResponse:
    ok: bool
    events: list[dict[str, Any]] = field(default_factory=list)
    remaining: int | None = None
    used: int | None = None
    error: str = ""
    fetched_at_utc: str = ""


def _key_from_legacy_app(root: Path) -> str:
    """Read the base64 key already embedded in the project's own app.py.

    Parsing it rather than duplicating it keeps a single source of truth: if the
    user rotates the key in one place, nothing here goes stale pointing at a
    copy.
    """
    app_path = root / "app.py"
    if not app_path.exists():
        return ""
    try:
        source = app_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    match = re.search(r'_ENCODED_API_KEY\s*=\s*"([A-Za-z0-9+/=]+)"', source)
    if not match:
        return ""
    try:
        return base64.b64decode(match.group(1)).decode("utf-8").strip()
    except (ValueError, UnicodeDecodeError):
        return ""


def resolve_api_key(root: Path) -> tuple[str, str]:
    """Return the key and where it came from, without ever logging its value."""
    for variable in ("ODDS_API_KEY", "TENNIS_ODDS_API_KEY", "UFC_ODDS_API_KEY"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value, f"variable d'environnement {variable}"
    try:
        import streamlit as st

        value = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        if value:
            return value, "secrets Streamlit"
    except Exception:  # noqa: BLE001 - secrets are optional
        pass
    value = _key_from_legacy_app(root)
    if value:
        return value, "clé encodée dans app.py"
    return "", "aucune clé trouvée"


def _request(path: str, key: str, params: dict[str, Any]) -> OddsResponse:
    query = {"apiKey": key, **params}
    try:
        response = requests.get(f"{BASE_URL}/{path}", params=query, timeout=30)
    except requests.RequestException as error:
        return OddsResponse(False, error=f"réseau: {error}")
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if response.status_code == 401:
        return OddsResponse(False, error="clé refusée (401)")
    if response.status_code == 429:
        return OddsResponse(False, error="quota mensuel épuisé (429)", remaining=0)
    if not response.ok:
        return OddsResponse(False, error=f"HTTP {response.status_code}")
    return OddsResponse(
        True,
        events=response.json(),
        remaining=int(remaining) if remaining and remaining.isdigit() else None,
        used=int(used) if used and used.isdigit() else None,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def active_sports(root: Path) -> OddsResponse:
    key, _ = resolve_api_key(root)
    if not key:
        return OddsResponse(False, error="aucune clé disponible")
    return _request("sports/", key, {})


def fetch_h2h_odds(root: Path, sport_key: str, regions: str = DEFAULT_REGIONS) -> OddsResponse:
    """One request, one sport. Each call costs quota, so callers must cache."""
    key, _ = resolve_api_key(root)
    if not key:
        return OddsResponse(False, error="aucune clé disponible")
    return _request(
        f"sports/{sport_key}/odds",
        key,
        {"regions": regions, "markets": DEFAULT_MARKET, "oddsFormat": "decimal"},
    )


def _is_ignored(name: str) -> bool:
    return name.strip().lower() in IGNORED_OUTCOMES


def best_prices(event: dict[str, Any]) -> dict[str, float]:
    """Best decimal price per participant across the returned bookmakers.

    Taking the best available quote is the right choice for a live page — it is
    what a bettor would actually shop for — but it is *not* comparable to the
    historical `maximum` column, which mixes prices captured at different
    moments. These are simultaneous quotes from one API response.
    """
    prices: dict[str, float] = {}
    for bookmaker in event.get("bookmakers") or []:
        for market in bookmaker.get("markets") or []:
            if market.get("key") != DEFAULT_MARKET:
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                price = outcome.get("price")
                if not name or _is_ignored(name):
                    continue
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                if price > prices.get(name, 0.0):
                    prices[name] = float(price)
    return prices


def consensus_prices(event: dict[str, Any]) -> dict[str, float]:
    """Median price per participant — a fairer reference than the best quote."""
    collected: dict[str, list[float]] = {}
    for bookmaker in event.get("bookmakers") or []:
        for market in bookmaker.get("markets") or []:
            if market.get("key") != DEFAULT_MARKET:
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                price = outcome.get("price")
                if name and not _is_ignored(name) and isinstance(price, (int, float)) and price > 1.0:
                    collected.setdefault(name, []).append(float(price))
    result: dict[str, float] = {}
    for name, values in collected.items():
        values.sort()
        middle = len(values) // 2
        result[name] = (
            values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
        )
    return result


def devig(prices: dict[str, float]) -> dict[str, float]:
    """No-vig probabilities for a two-way market."""
    if len(prices) != 2:
        return {}
    inverse = {name: 1.0 / price for name, price in prices.items()}
    total = sum(inverse.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in inverse.items()}
