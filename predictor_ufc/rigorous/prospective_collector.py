"""Collecte append-only des cotes MMA a partir de The Odds API.

Le flux MMA peut contenir d'autres organisations que l'UFC. Les lignes sont donc
conservees comme MMA brutes et ne deviennent UFC qu'apres appariement explicite.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .data_pipeline import sha256_file, stable_id


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"


def normalise_odds_api_payload(payload: list[dict[str, Any]], fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Une ligne par combat/bookmaker/snapshot, sans resultat ni reorientation future."""
    rows: list[dict[str, Any]] = []
    for event in payload:
        event_id = str(event.get("id") or "")
        commence_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = [
                    (str(item.get("name") or "").strip(), pd.to_numeric(item.get("price"), errors="coerce"))
                    for item in market.get("outcomes") or []
                ]
                outcomes = sorted(
                    [(name, float(price)) for name, price in outcomes if name and pd.notna(price) and price > 1.0],
                    key=lambda item: item[0],
                )
                if len(outcomes) != 2:
                    continue
                fighter_1, odds_1 = outcomes[0]
                fighter_2, odds_2 = outcomes[1]
                bookmaker_key = str(bookmaker.get("key") or "")
                rows.append(
                    {
                        "snapshot_id": stable_id(event_id, bookmaker_key, fetched_at.isoformat()),
                        "fetched_at": fetched_at,
                        "event_external_id": event_id,
                        "sport_key": str(event.get("sport_key") or ""),
                        "sport_title": str(event.get("sport_title") or ""),
                        "commence_time": commence_time,
                        "home_team_raw": str(event.get("home_team") or ""),
                        "away_team_raw": str(event.get("away_team") or ""),
                        "fighter_1": fighter_1,
                        "fighter_2": fighter_2,
                        "odds_fighter_1": odds_1,
                        "odds_fighter_2": odds_2,
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": str(bookmaker.get("title") or ""),
                        "bookmaker_last_update": pd.to_datetime(
                            bookmaker.get("last_update"), utc=True, errors="coerce"
                        ),
                        "market_last_update": pd.to_datetime(
                            market.get("last_update"), utc=True, errors="coerce"
                        ),
                        "ufc_match_status": "UNVERIFIED_MMA_EVENT",
                    }
                )
    return pd.DataFrame(rows)


def collect_current_odds(base_dir: Path, api_key: str | None = None) -> dict[str, Any]:
    """Interroge le flux courant puis ajoute un snapshot immuable au journal local."""
    key = api_key or os.environ.get("UFC_ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "Cle absente: definir UFC_ODDS_API_KEY (ou THE_ODDS_API_KEY) avant collect-odds"
        )
    fetched_at = pd.Timestamp(datetime.now(timezone.utc)).floor("s")
    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": key,
            "regions": "us,uk,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    current = normalise_odds_api_payload(payload, fetched_at)
    if current.empty:
        raise RuntimeError("Le fournisseur n'a retourne aucune moneyline MMA exploitable")

    destination = base_dir / "data" / "rigorous" / "prospective"
    destination.mkdir(parents=True, exist_ok=True)
    snapshots_path = destination / "mma_moneyline_snapshots.parquet"
    if snapshots_path.exists():
        previous = pd.read_parquet(snapshots_path)
        combined = pd.concat([previous, current], ignore_index=True, sort=False)
    else:
        combined = current
    combined = combined.drop_duplicates("snapshot_id", keep="last").sort_values(
        ["fetched_at", "event_external_id", "bookmaker_key"]
    )
    combined.to_parquet(snapshots_path, index=False)

    audit = {
        "fetched_at_utc": fetched_at.isoformat(),
        "endpoint": ODDS_API_URL,
        "new_rows": int(len(current)),
        "new_events_unverified_mma": int(current["event_external_id"].nunique()),
        "new_bookmakers": int(current["bookmaker_key"].nunique()),
        "total_rows": int(len(combined)),
        "quota_requests_remaining": response.headers.get("x-requests-remaining"),
        "quota_requests_used": response.headers.get("x-requests-used"),
        "artifact_sha256": sha256_file(snapshots_path),
        "warning": "MMA brut: appariement UFC explicite obligatoire avant modelisation.",
    }
    (destination / "latest_collection_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    return audit
