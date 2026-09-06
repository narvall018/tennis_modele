"""Tours and tiers that the main ATP pipeline does not collect.

The ATP main-tour table in :mod:`src.data.tennis_pipeline` is the only market the
project has ever modelled, and every walk-forward search run so far has already
seen all of it.  This module publishes the three bodies of match data that were
left out, each for a different reason:

* **WTA main tour** — a second priced market.  Tennis-Data publishes WTA
  workbooks with the same bookmaker columns as the ATP ones from 2007 onwards,
  and TennisMyLife publishes the matching WTA match files.  No previous search
  in this repository has read a single WTA row, so the tour is untouched
  evaluation ground rather than a bigger version of the same sample.
* **ATP Challenger** and **ATP qualifying** — unpriced matches.  They carry no
  betting market, so they can never be bet on here; they exist so that a
  player's rating already reflects the matches he actually played before he
  shows up in a main draw.  Qualifiers and players moving up from the Challenger
  circuit are exactly the entries a main-tour-only rating knows least about.

Nothing published here is oriented, filtered, or scored using the result of the
match, and the unpriced tables deliberately carry no odds columns at all.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import concurrent.futures
import io
import numpy as np
import pandas as pd

from src.data.tennis_pipeline import (
    DataQualityError,
    SourceSnapshot,
    TENNIS_MYLIFE_INVENTORY_URL,
    _atomic_csv,
    _atomic_json,
    _http_bytes,
    _http_json,
    _odds_coverage,
    add_stable_player_orientation,
    attach_odds,
    fetch_odds_snapshot,
    normalize_legacy_odds,
    normalize_rich_matches,
    transform_tennis_data_raw,
)


# Tennis-Data publishes WTA workbooks with bookmaker prices from 2007 onwards.
WTA_ODDS_FIRST_YEAR = 2007
# TennisMyLife match files start in 2000 for every tour used here.
MATCH_FIRST_YEAR = 2000

UNPRICED_SEGMENTS = {
    "challenger": {
        "file_template": "{year}_challenger.csv",
        "id_prefix": "chal:",
        "tour": "atp",
        "segment": "challenger",
    },
    "qualifying": {
        "file_template": "atp_quali/{year}_atp_quali.csv",
        "id_prefix": "quali:",
        "tour": "atp",
        "segment": "qualifying",
    },
}


def _download_inventory_files(names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download named TennisMyLife files, keeping their published timestamps."""
    inventory = _http_json(TENNIS_MYLIFE_INVENTORY_URL)
    files = {item["name"]: item for item in inventory.get("files", [])}
    available = [name for name in names if name in files]
    if not available:
        raise DataQualityError(f"None of the requested TennisMyLife files exist: {names}")

    def download(name: str) -> pd.DataFrame:
        item = files[name]
        frame = pd.read_csv(io.BytesIO(_http_bytes(item["url"])), low_memory=False)
        frame["_source_file"] = name
        frame["_source_updated_at"] = item.get("mtime", "")
        return frame

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        frames = list(executor.map(download, available))
    details = {
        "requested_files": names,
        "downloaded_files": available,
        "missing_files": [name for name in names if name not in files],
        "source_updated_at": max(str(files[name].get("mtime", "")) for name in available),
    }
    return pd.concat(frames, ignore_index=True, sort=False), details


def fetch_wta_matches(start_year: int, end_year: int) -> tuple[pd.DataFrame, SourceSnapshot, dict[str, Any]]:
    names = [f"{year}_wta.csv" for year in range(start_year, end_year + 1)]
    names.append("wta_ongoing_tourneys.csv")
    matches, details = _download_inventory_files(names)
    snapshot = SourceSnapshot(
        name="TennisMyLife WTA match database",
        updated_at=details["source_updated_at"],
        url=TENNIS_MYLIFE_INVENTORY_URL,
    )
    return matches, snapshot, details


def fetch_unpriced_matches(
    segment: str, start_year: int, end_year: int
) -> tuple[pd.DataFrame, SourceSnapshot, dict[str, Any]]:
    config = UNPRICED_SEGMENTS[segment]
    names = [config["file_template"].format(year=year) for year in range(start_year, end_year + 1)]
    if segment == "challenger":
        names.append("challenger_ongoing_tourneys.csv")
    matches, details = _download_inventory_files(names)
    snapshot = SourceSnapshot(
        name=f"TennisMyLife ATP {segment} match database",
        updated_at=details["source_updated_at"],
        url=TENNIS_MYLIFE_INVENTORY_URL,
    )
    return matches, snapshot, details


def _add_empty_odds_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Give an unpriced table the odds columns, explicitly empty.

    The columns exist so downstream code can concatenate priced and unpriced
    matches without inventing a price for a market that was never quoted.
    """
    result = frame.copy()
    for column in [
        "winner_odds",
        "loser_odds",
        "odds_match_confidence",
        "odds_date_delta_days",
        "odds_tournament_similarity",
        "market_overround",
        "winner_market_prob_no_vig",
        "loser_market_prob_no_vig",
    ]:
        result[column] = np.nan
    for column in ["odds_source_match_date", "odds_source_tournament"]:
        result[column] = ""
    return result


def _segment_summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    dates = pd.to_datetime(frame["match_date"], errors="coerce")
    players = pd.unique(
        pd.concat([frame["winner_id"].astype(str), frame["loser_id"].astype(str)], ignore_index=True)
    )
    return {
        "label": label,
        "rows": int(len(frame)),
        "date_min": str(dates.min().date()) if len(frame) else None,
        "date_max": str(dates.max().date()) if len(frame) else None,
        "distinct_players": int(len(players)),
        "distinct_tournaments": int(frame["tourney_id"].nunique()),
        "duplicate_match_ids": int(len(frame) - frame["match_id"].nunique()),
        "completed_rows": int((frame["match_status"] == "completed").sum()),
        "rows_by_year": {
            str(year): int(count) for year, count in dates.dt.year.value_counts().sort_index().items()
        },
    }


def validate_expansion(report: dict[str, Any], today: date) -> None:
    """Refuse to publish a snapshot that fails a check a later model would rely on."""
    wta = report["wta"]
    if wta["matches"]["duplicate_match_ids"] != 0:
        raise DataQualityError("Duplicate WTA match identifiers")
    if wta["odds"]["priced_rows"] < 20_000:
        raise DataQualityError(f"Implausibly few priced WTA matches: {wta['odds']['priced_rows']}")
    if wta["odds"]["one_sided_pairs"] != 0:
        raise DataQualityError("A WTA match kept only one side of a price pair")
    max_date = pd.to_datetime(wta["matches"]["date_max"]).date()
    if (today - max_date).days > 21:
        raise DataQualityError(f"Stale WTA matches, newest is {max_date}")
    for segment in report["unpriced"].values():
        if segment["duplicate_match_ids"] != 0:
            raise DataQualityError(f"Duplicate identifiers in {segment['label']}")
        if segment["rows"] == 0:
            raise DataQualityError(f"Empty segment {segment['label']}")
        newest = pd.to_datetime(segment["date_max"]).date()
        if (today - newest).days > 45:
            raise DataQualityError(f"Stale {segment['label']}, newest is {newest}")


def run_expansion_update(project_root: str | Path, as_of_date: date | None = None) -> dict[str, Any]:
    """Download, validate, and atomically publish the WTA and unpriced tables."""
    root = Path(project_root).resolve()
    today = as_of_date or datetime.now().date()

    wta_odds_raw, wta_odds_source = fetch_odds_snapshot(WTA_ODDS_FIRST_YEAR, today.year, tour="wta")
    wta_odds = normalize_legacy_odds(transform_tennis_data_raw(wta_odds_raw), today=today)
    wta_raw, wta_source, wta_details = fetch_wta_matches(MATCH_FIRST_YEAR, today.year)
    wta_matches = normalize_rich_matches(wta_raw, today=today, id_prefix="wta:")
    wta_enriched = add_stable_player_orientation(attach_odds(wta_matches, wta_odds))
    wta_enriched.insert(0, "tour", "wta")
    wta_enriched.insert(1, "segment", "main")

    unpriced_frames: list[pd.DataFrame] = []
    unpriced_summaries: dict[str, Any] = {}
    unpriced_details: dict[str, Any] = {}
    for segment, config in UNPRICED_SEGMENTS.items():
        raw, _, details = fetch_unpriced_matches(segment, MATCH_FIRST_YEAR, today.year)
        normalized = normalize_rich_matches(raw, today=today, id_prefix=config["id_prefix"])
        oriented = add_stable_player_orientation(_add_empty_odds_columns(normalized))
        oriented.insert(0, "tour", config["tour"])
        oriented.insert(1, "segment", config["segment"])
        unpriced_frames.append(oriented)
        unpriced_summaries[segment] = _segment_summary(oriented, f"atp {segment}")
        unpriced_details[segment] = details
    unpriced = pd.concat(unpriced_frames, ignore_index=True, sort=False)
    unpriced = unpriced.sort_values(["match_date", "tourney_id", "match_num"]).reset_index(drop=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(today),
        "wta": {
            "matches": _segment_summary(wta_enriched, "wta main"),
            "odds": {
                **_odds_coverage(wta_odds, "Odd_1", "Odd_2"),
                "priced_rows": int(
                    (wta_enriched["winner_odds"] > 1.0).fillna(False).sum()
                ),
                "workbook_rows": int(len(wta_odds_raw)),
                "normalized_rows": int(len(wta_odds)),
                "first_priced_year": WTA_ODDS_FIRST_YEAR,
                "median_match_confidence": float(
                    pd.to_numeric(wta_enriched["odds_match_confidence"], errors="coerce").median()
                ),
                "median_date_delta_days": float(
                    pd.to_numeric(wta_enriched["odds_date_delta_days"], errors="coerce").median()
                ),
            },
        },
        "unpriced": unpriced_summaries,
        "sources": {
            "wta_odds": {
                "name": wta_odds_source.name,
                "url": wta_odds_source.url,
                "updated_at": wta_odds_source.updated_at,
            },
            "wta_matches": {
                "name": wta_source.name,
                "url": wta_source.url,
                "updated_at": wta_source.updated_at,
                **wta_details,
            },
            "unpriced_matches": unpriced_details,
        },
    }
    validate_expansion(report, today=today)

    _atomic_csv(wta_odds, root / "data" / "wta_tennis.csv")
    _atomic_csv(
        wta_odds_raw,
        root / "data" / "raw" / "tennis_data" / "wta_odds_2007_current.csv.gz",
        gzip=True,
    )
    _atomic_csv(
        wta_raw,
        root / "data" / "raw" / "tennis_mylife" / "wta_matches_2000_current.csv.gz",
        gzip=True,
    )
    _atomic_csv(
        wta_enriched,
        root / "data" / "processed" / "wta_matches_enriched.csv.gz",
        gzip=True,
    )
    _atomic_csv(
        unpriced,
        root / "data" / "processed" / "atp_unpriced_matches.csv.gz",
        gzip=True,
    )
    _atomic_json(report, root / "data" / "quality" / "tour_expansion_quality.json")
    return report
