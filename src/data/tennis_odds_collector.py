"""Append-only prospective ATP moneyline collector using The Odds API."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SPORTS_URL = "https://api.the-odds-api.com/v4/sports/"
ODDS_URL_TEMPLATE = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"


def _stable_id(*values: object) -> str:
    text = "|".join(str(value or "").strip() for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_tennis_odds_payload(
    payload: list[dict[str, Any]],
    sport_key: str,
    sport_title: str,
    fetched_at: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in payload:
        event_id = str(event.get("id") or "")
        commence_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if not event_id or pd.isna(commence_time):
            continue
        for bookmaker in event.get("bookmakers") or []:
            bookmaker_key = str(bookmaker.get("key") or "")
            for market in bookmaker.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = sorted(
                    [
                        (str(item.get("name") or "").strip(), pd.to_numeric(item.get("price"), errors="coerce"))
                        for item in market.get("outcomes") or []
                    ],
                    key=lambda item: item[0],
                )
                outcomes = [
                    (name, float(price))
                    for name, price in outcomes
                    if name and pd.notna(price) and float(price) > 1.0
                ]
                if len(outcomes) != 2:
                    continue
                fighter_1, odds_1 = outcomes[0]
                fighter_2, odds_2 = outcomes[1]
                market_update = pd.to_datetime(market.get("last_update"), utc=True, errors="coerce")
                rows.append(
                    {
                        "snapshot_id": _stable_id(event_id, bookmaker_key, fetched_at.isoformat()),
                        "fetched_at": fetched_at,
                        "sport_key": sport_key,
                        "sport_title": sport_title,
                        "tour_scope": "ATP",
                        "event_external_id": event_id,
                        "commence_time": commence_time,
                        "home_team_raw": str(event.get("home_team") or ""),
                        "away_team_raw": str(event.get("away_team") or ""),
                        "player_1": fighter_1,
                        "player_2": fighter_2,
                        "odds_player_1": odds_1,
                        "odds_player_2": odds_2,
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": str(bookmaker.get("title") or ""),
                        "bookmaker_last_update": pd.to_datetime(
                            bookmaker.get("last_update"), utc=True, errors="coerce"
                        ),
                        "market_last_update": market_update,
                        "temporal_status": (
                            "PRE_MATCH" if fetched_at < commence_time else "POST_START_EXCLUDED"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def collect_current_atp_odds(
    base_dir: str | Path,
    api_key: str | None = None,
    maximum_sports: int | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ.get("TENNIS_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "Cle absente: definir TENNIS_ODDS_API_KEY (ou ODDS_API_KEY) avant la collecte"
        )
    fetched_at = pd.Timestamp(datetime.now(timezone.utc)).floor("s")
    session = requests.Session()
    sports_response = session.get(SPORTS_URL, params={"apiKey": key}, timeout=60)
    sports_response.raise_for_status()
    sports = [
        item for item in sports_response.json()
        if item.get("active") and str(item.get("key") or "").startswith("tennis_atp_")
    ]
    sports = sorted(sports, key=lambda item: str(item.get("key")))
    if maximum_sports is not None:
        sports = sports[: max(0, int(maximum_sports))]
    if not sports:
        raise RuntimeError("Aucun tournoi ATP actif retourne par le fournisseur")

    batches: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    last_headers = sports_response.headers
    for sport in sports:
        sport_key = str(sport["key"])
        try:
            response = session.get(
                ODDS_URL_TEMPLATE.format(sport_key=sport_key),
                params={
                    "apiKey": key,
                    "regions": "eu,uk,us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=60,
            )
            response.raise_for_status()
            last_headers = response.headers
            batch = normalise_tennis_odds_payload(
                response.json(), sport_key, str(sport.get("title") or ""), fetched_at
            )
            if not batch.empty:
                batches.append(batch)
        except requests.RequestException as exc:
            failures.append({"sport_key": sport_key, "error": type(exc).__name__})
    current = pd.concat(batches, ignore_index=True, sort=False) if batches else pd.DataFrame()
    if current.empty:
        raise RuntimeError("Aucune cote ATP exploitable collectee")
    current = current[current["temporal_status"].eq("PRE_MATCH")].copy()
    if current.empty:
        raise RuntimeError("Toutes les cotes retournees etaient posterieures au debut")

    destination = Path(base_dir).resolve() / "data" / "prospective"
    destination.mkdir(parents=True, exist_ok=True)
    snapshots_path = destination / "atp_moneyline_snapshots.parquet"
    if snapshots_path.exists():
        previous = pd.read_parquet(snapshots_path)
        combined = pd.concat([previous, current], ignore_index=True, sort=False)
    else:
        combined = current
    combined = combined.drop_duplicates("snapshot_id", keep="last").sort_values(
        ["fetched_at", "commence_time", "event_external_id", "bookmaker_key"]
    )
    combined.to_parquet(snapshots_path, index=False)
    audit = {
        "fetched_at_utc": fetched_at.isoformat(),
        "active_atp_sports_requested": int(len(sports)),
        "failed_sports": failures,
        "new_rows": int(len(current)),
        "new_events": int(current["event_external_id"].nunique()),
        "new_bookmakers": int(current["bookmaker_key"].nunique()),
        "total_rows": int(len(combined)),
        "quota_requests_remaining": last_headers.get("x-requests-remaining"),
        "quota_requests_used": last_headers.get("x-requests-used"),
        "artifact_sha256": _sha256(snapshots_path),
        "rule": "Only fetched_at strictly before commence_time is retained.",
    }
    (destination / "latest_atp_collection_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    return audit
