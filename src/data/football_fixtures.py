"""Upcoming football fixtures with their quoted prices.

Football-Data publishes a rolling file of matches not yet played, carrying the
same bookmaker columns as the historical seasons. It is the only free source in
this project that gives a fixture list and a price together, which is what makes
a live prediction page possible for football at all.

Pinnacle is absent from the fixtures file, so the market reference here is the
market average — a wider margin than the historical analyses used. Any edge
displayed against it is therefore measured against a more expensive price than
the one those analyses reported.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data.football_pipeline import LEAGUES, _http_bytes


FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

# Columns the fixtures file publishes; Pinnacle is not among them.
FIXTURE_PRICE_GROUPS = {
    "market_average": ("AvgH", "AvgD", "AvgA"),
    "bet365": ("B365H", "B365D", "B365A"),
    "market_maximum": ("MaxH", "MaxD", "MaxA"),
}


def fetch_fixtures() -> pd.DataFrame:
    payload = _http_bytes(FIXTURES_URL, timeout=60)
    if payload is None:
        raise RuntimeError("Football-Data n'a pas renvoyé de fichier de rencontres")
    # The file carries a UTF-8 byte-order mark that would otherwise corrupt the
    # first column name.
    return pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", on_bad_lines="skip")


def normalize_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).strip().lstrip("﻿") for column in frame.columns]
    if "Div" not in frame.columns:
        raise RuntimeError(f"Colonne Div absente; colonnes vues: {list(frame.columns)[:6]}")

    frame = frame[frame["Div"].isin(LEAGUES)].copy()
    parsed = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(frame["Date"], format="%d/%m/%y", errors="coerce")
    frame["match_date"] = parsed.fillna(fallback)
    frame = frame.dropna(subset=["match_date"])

    frame["home_team"] = frame["HomeTeam"].astype(str).str.strip()
    frame["away_team"] = frame["AwayTeam"].astype(str).str.strip()
    frame = frame[frame["home_team"].ne("") & frame["away_team"].ne("")]
    frame["league"] = frame["Div"]
    frame["country"] = frame["league"].map(lambda code: LEAGUES[code][0])
    frame["division_rank"] = frame["league"].map(lambda code: LEAGUES[code][1])
    frame["kickoff"] = frame.get("Time", "").astype(str)
    frame["match_id"] = (
        frame["league"].astype(str) + ":"
        + frame["match_date"].dt.strftime("%Y%m%d") + ":"
        + frame["home_team"] + ":" + frame["away_team"]
    )

    published = [
        "match_id", "match_date", "kickoff", "league", "country", "division_rank",
        "home_team", "away_team",
    ]
    for _, columns in FIXTURE_PRICE_GROUPS.items():
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            published.append(column)
    frame = frame.drop_duplicates("match_id", keep="last")
    return frame[published].sort_values(["match_date", "league"]).reset_index(drop=True)


def load_fixtures(min_date: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch fixtures, dropping any whose kickoff has already passed.

    The published file keeps a few days of matches that have already been played;
    showing them as "upcoming" with a stale price would invite a bet that can no
    longer be placed.
    """
    raw = fetch_fixtures()
    fixtures = normalize_fixtures(raw)
    cutoff = pd.Timestamp(min_date) if min_date is not None else pd.Timestamp.today().normalize()
    published_total = len(fixtures)
    fixtures = fixtures[fixtures["match_date"] >= cutoff].reset_index(drop=True)
    meta = {
        "dropped_already_played": int(published_total - len(fixtures)),
        "cutoff": str(cutoff.date()),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": FIXTURES_URL,
        "fixtures": int(len(fixtures)),
        "date_min": str(fixtures["match_date"].min().date()) if len(fixtures) else None,
        "date_max": str(fixtures["match_date"].max().date()) if len(fixtures) else None,
        "price_reference": "moyenne de marché (Pinnacle absent du fichier de rencontres)",
    }
    return fixtures, meta
