"""Reproducible ATP data ingestion and enrichment.

The project uses two complementary public snapshots:

* Tennis-Data's official yearly workbooks for historical pre-match odds and
  settlement status.
* TennisMyLife for match/player identifiers and detailed match statistics.

The odds table remains compatible with the legacy application.  The richer table
keeps post-match statistics explicitly prefixed so they cannot accidentally be
used as pre-match features later on.
"""

from __future__ import annotations

import concurrent.futures
import gzip as gzip_module
import hashlib
import io
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TENNIS_DATA_BASE_URL = "http://www.tennis-data.co.uk"
TENNIS_DATA_INDEX_URL = f"{TENNIS_DATA_BASE_URL}/alldata.php"
TENNIS_MYLIFE_INVENTORY_URL = "https://stats.tennismylife.org/api/data-files"

LEGACY_COLUMNS = [
    "Tournament",
    "Date",
    "Series",
    "Court",
    "Surface",
    "Round",
    "Best of",
    "Player_1",
    "Player_2",
    "Winner",
    "Rank_1",
    "Rank_2",
    "Pts_1",
    "Pts_2",
    "Odd_1",
    "Odd_2",
    "Score",
    "Status",
    "Odds_source",
    "B365_1",
    "B365_2",
    "Pinnacle_1",
    "Pinnacle_2",
    "Avg_1",
    "Avg_2",
    "Max_1",
    "Max_2",
    "Betfair_1",
    "Betfair_2",
]

POSTMATCH_STATS = [
    "ace",
    "df",
    "svpt",
    "1stIn",
    "1stWon",
    "2ndWon",
    "SvGms",
    "bpSaved",
    "bpFaced",
]

TOURNAMENT_ALIASES = {
    "australian": "australian_open",
    "roland garros": "french_open",
    "french open": "french_open",
    "wimbledon": "wimbledon",
    "us open": "us_open",
    "indian wells": "indian_wells",
    "paribas": "indian_wells",
    "miami": "miami",
    "monte carlo": "monte_carlo",
    "madrid": "madrid",
    "rome": "rome",
    "italia": "rome",
    "canada": "canada",
    "canadian": "canada",
    "rogers cup": "canada",
    "national bank": "canada",
    "cincinnati": "cincinnati",
    "western southern": "cincinnati",
    "shanghai": "shanghai",
    "paris masters": "paris_masters",
}

ROUND_ALIASES = {
    "r128": "r128",
    "1st round": "r128_or_r64",
    "r64": "r64",
    "2nd round": "r64_or_r32",
    "r32": "r32",
    "3rd round": "r32_or_r16",
    "r16": "r16",
    "4th round": "r16",
    "quarterfinals": "qf",
    "quarter finals": "qf",
    "qf": "qf",
    "semifinals": "sf",
    "semi finals": "sf",
    "sf": "sf",
    "the final": "f",
    "final": "f",
    "f": "f",
    "round robin": "rr",
    "rr": "rr",
}


class DataQualityError(RuntimeError):
    """Raised before publication when a downloaded snapshot fails hard checks."""


@dataclass(frozen=True)
class SourceSnapshot:
    name: str
    updated_at: str
    url: str


HOST_FALLBACKS = {
    "www.tennis-data.co.uk": ("tennis-data.co.uk",),
    "tennis-data.co.uk": ("www.tennis-data.co.uk",),
}


def _url_variants(url: str) -> list[str]:
    """Return the URL followed by equivalent hosts serving the same files."""
    parsed = urllib.parse.urlsplit(url)
    variants = [url]
    for host in HOST_FALLBACKS.get(parsed.netloc, ()):
        variants.append(urllib.parse.urlunsplit(parsed._replace(netloc=host)))
    return variants


def _http_bytes(url: str, timeout: int = 120, attempts: int = 3) -> bytes:
    """Download a URL, retrying transient failures and equivalent hosts.

    Tennis-Data intermittently answers ``503`` on one of its two hostnames while
    the other keeps serving the same workbooks, so a failure is only final once
    every variant has been retried.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        for candidate in _url_variants(url):
            request = urllib.request.Request(
                candidate,
                headers={"User-Agent": "tennis-modele-data-pipeline/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
        if attempt + 1 < attempts:
            time.sleep(2.0 * (attempt + 1))
    raise DataQualityError(f"Download failed for {url}: {last_error}") from last_error


def _http_json(url: str, timeout: int = 60) -> dict[str, Any]:
    return json.loads(_http_bytes(url, timeout=timeout).decode("utf-8"))


def _deaccent(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in value if not unicodedata.combining(char))


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z]+", _deaccent(value).lower())


def _full_name_key(value: Any) -> str:
    """Compact key for a ``First ... Last`` name from TennisMyLife."""
    tokens = _tokens(value)
    if not tokens:
        return ""
    return f"{tokens[-1]}|{tokens[0][0]}"


def _abbreviated_name_key(value: Any) -> str:
    """Compact key for a ``Last F.`` name from Tennis-Data."""
    tokens = _tokens(value)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[-2]}|{tokens[-1][0]}"


def _pair_key(left: str, right: str) -> str:
    return "~".join(sorted((left, right)))


def _normal_text(value: Any) -> str:
    return " ".join(_tokens(value))


def _tournament_key(value: Any) -> str:
    normalized = _normal_text(value)
    for fragment, canonical in TOURNAMENT_ALIASES.items():
        if fragment in normalized:
            return canonical
    stopwords = {
        "atp",
        "open",
        "tennis",
        "championships",
        "championship",
        "masters",
        "international",
        "presented",
        "by",
        "the",
    }
    words = [word for word in normalized.split() if word not in stopwords]
    return " ".join(words)


def _round_key(value: Any) -> str:
    normalized = _normal_text(value)
    return ROUND_ALIASES.get(normalized, normalized)


def _tournament_similarity(left: Any, right: Any) -> float:
    key_left = _tournament_key(left)
    key_right = _tournament_key(right)
    if key_left and key_left == key_right:
        return 1.0
    if not key_left or not key_right:
        return 0.0
    return SequenceMatcher(None, key_left, key_right).ratio()


def _swap_score(score: Any) -> Any:
    if pd.isna(score):
        return score
    swapped: list[str] = []
    for part in str(score).split():
        match = re.match(r"^([^\-]+)-([^\-]+)$", part)
        swapped.append(f"{match.group(2)}-{match.group(1)}" if match else part)
    return " ".join(swapped)


def _stable_first_player(
    match_date: Any,
    tournament: Any,
    round_name: Any,
    player_a: str,
    player_b: str,
) -> str:
    ordered = sorted((str(player_a), str(player_b)))
    identity = "|".join(
        [str(match_date), str(tournament), str(round_name), ordered[0], ordered[1]]
    )
    bit = hashlib.sha256(identity.encode("utf-8")).digest()[0] & 1
    return ordered[bit]


def deterministic_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Give every match a stable, outcome-independent P1/P2 orientation."""
    result = df.copy()
    should_swap = []
    for row in result.itertuples(index=False):
        target = _stable_first_player(
            getattr(row, "Date"),
            getattr(row, "Tournament"),
            getattr(row, "Round"),
            getattr(row, "Player_1"),
            getattr(row, "Player_2"),
        )
        should_swap.append(target != getattr(row, "Player_1"))
    mask = pd.Series(should_swap, index=result.index)
    pairs = [
        ("Player_1", "Player_2"),
        ("Rank_1", "Rank_2"),
        ("Pts_1", "Pts_2"),
        ("Odd_1", "Odd_2"),
        ("B365_1", "B365_2"),
        ("Pinnacle_1", "Pinnacle_2"),
        ("Avg_1", "Avg_2"),
        ("Max_1", "Max_2"),
        ("Betfair_1", "Betfair_2"),
    ]
    for left, right in pairs:
        if left not in result or right not in result:
            continue
        old_left = result.loc[mask, left].copy()
        result.loc[mask, left] = result.loc[mask, right].values
        result.loc[mask, right] = old_left.values
    result.loc[mask, "Score"] = result.loc[mask, "Score"].map(_swap_score)
    return result


def _first_valid_odd(row: pd.Series, columns: Iterable[str]) -> tuple[float, str]:
    for column in columns:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 1.0:
            return float(value), column
    return -1.0, "missing"


def _valid_odd_pair(row: pd.Series, left: str, right: str) -> tuple[float, float] | None:
    left_value = pd.to_numeric(pd.Series([row.get(left)]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([row.get(right)]), errors="coerce").iloc[0]
    if pd.notna(left_value) and pd.notna(right_value) and left_value > 1.0 and right_value > 1.0:
        return float(left_value), float(right_value)
    return None


def _match_status(comment: Any) -> str:
    normalized = str(comment or "").strip().lower()
    if not normalized or normalized == "completed":
        return "completed"
    if "retir" in normalized or "rrtir" in normalized:
        return "retired"
    if "walk" in normalized or "w/o" in normalized:
        return "walkover"
    if "award" in normalized:
        return "awarded"
    if "disqual" in normalized or "default" in normalized:
        return "defaulted"
    return normalized.replace(" ", "_")


def transform_tennis_data_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transform official Tennis-Data workbooks without outcome-based filtering.

    ``Odd_1``/``Odd_2`` use the market-average pair whenever available.  Both
    prices always come from the same market source.  Individual bookmaker,
    average, maximum, and exchange prices are retained for sensitivity checks.
    Retirements and walkovers are retained because their status is unknowable
    when a pre-match betting decision is made.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df_raw.iterrows():
        if pd.isna(row.get("Winner")) or pd.isna(row.get("Loser")):
            continue
        match_date = pd.to_datetime(row.get("Date"), errors="coerce")
        if pd.isna(match_date):
            continue
        winner = str(row["Winner"]).strip()
        loser = str(row["Loser"]).strip()
        if not winner or not loser or winner == loser:
            continue

        named_pairs = [
            ("AvgW", "AvgL", "market_average"),
            ("B365W", "B365L", "bet365"),
            ("PSW", "PSL", "pinnacle"),
            ("MaxW", "MaxL", "market_maximum"),
        ]
        selected = None
        selected_source = "missing"
        for left, right, source_name in named_pairs:
            selected = _valid_odd_pair(row, left, right)
            if selected is not None:
                selected_source = source_name
                break
        winner_odd, loser_odd = selected if selected is not None else (-1.0, -1.0)

        def individual_pair(left: str, right: str) -> tuple[float, float]:
            pair = _valid_odd_pair(row, left, right)
            return pair if pair is not None else (-1.0, -1.0)

        b365_w, b365_l = individual_pair("B365W", "B365L")
        pin_w, pin_l = individual_pair("PSW", "PSL")
        avg_w, avg_l = individual_pair("AvgW", "AvgL")
        max_w, max_l = individual_pair("MaxW", "MaxL")
        bfe_w, bfe_l = individual_pair("BFEW", "BFEL")
        best_of = pd.to_numeric(pd.Series([row.get("Best of")]), errors="coerce").iloc[0]
        best_of = int(best_of) if pd.notna(best_of) and best_of > 0 else 3
        score_parts = []
        for set_number in range(1, 6):
            winner_games = pd.to_numeric(
                pd.Series([row.get(f"W{set_number}")]), errors="coerce"
            ).iloc[0]
            loser_games = pd.to_numeric(
                pd.Series([row.get(f"L{set_number}")]), errors="coerce"
            ).iloc[0]
            if pd.notna(winner_games) and pd.notna(loser_games):
                score_parts.append(f"{int(winner_games)}-{int(loser_games)}")
        rows.append(
            {
                "Tournament": row.get("Tournament", ""),
                "Date": match_date.strftime("%Y-%m-%d"),
                "Series": row.get("Series", ""),
                "Court": row.get("Court", "Outdoor"),
                "Surface": row.get("Surface", ""),
                "Round": row.get("Round", ""),
                "Best of": best_of,
                "Player_1": winner,
                "Player_2": loser,
                "Winner": winner,
                "Rank_1": row.get("WRank", -1),
                "Rank_2": row.get("LRank", -1),
                "Pts_1": row.get("WPts", -1),
                "Pts_2": row.get("LPts", -1),
                "Odd_1": winner_odd,
                "Odd_2": loser_odd,
                "Score": " ".join(score_parts),
                "Status": _match_status(row.get("Comment", "Completed")),
                "Odds_source": selected_source,
                "B365_1": b365_w,
                "B365_2": b365_l,
                "Pinnacle_1": pin_w,
                "Pinnacle_2": pin_l,
                "Avg_1": avg_w,
                "Avg_2": avg_l,
                "Max_1": max_w,
                "Max_2": max_l,
                "Betfair_1": bfe_w,
                "Betfair_2": bfe_l,
            }
        )
    return deterministic_orientation(pd.DataFrame(rows, columns=LEGACY_COLUMNS))


def fetch_odds_snapshot(
    start_year: int = 2000,
    end_year: int | None = None,
    tour: str = "atp",
) -> tuple[pd.DataFrame, SourceSnapshot]:
    """Download the official yearly Tennis-Data workbooks for one tour.

    Some older URLs end in ``.xlsx`` but contain the legacy XLS binary format;
    pandas chooses the appropriate installed engine from the file signature.
    The WTA workbooks live under a ``<year>w`` directory and label the event
    grade ``Tier`` instead of ``Series``.
    """
    if tour not in {"atp", "wta"}:
        raise ValueError(f"Unknown tour: {tour}")
    final_year = end_year or datetime.now().year
    years = list(range(start_year, final_year + 1))
    directory_suffix = "w" if tour == "wta" else ""

    def download_year(year: int) -> pd.DataFrame:
        url = f"{TENNIS_DATA_BASE_URL}/{year}{directory_suffix}/{year}.xlsx"
        data = _http_bytes(url)
        frame = pd.read_excel(io.BytesIO(data))
        frame["_source_file"] = f"{year}{directory_suffix}.xlsx"
        return frame

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        frames = list(executor.map(download_year, years))
    raw = pd.concat(frames, ignore_index=True, sort=False)
    if "Series" not in raw.columns and "Tier" in raw.columns:
        raw["Series"] = raw["Tier"]
    return raw, SourceSnapshot(
        name=f"Tennis-Data official {tour.upper()} workbooks",
        updated_at=datetime.now(timezone.utc).isoformat(),
        url=TENNIS_DATA_INDEX_URL,
    )


def normalize_legacy_odds(frame: pd.DataFrame, today: date) -> pd.DataFrame:
    missing = sorted(set(LEGACY_COLUMNS) - set(frame.columns))
    if missing:
        raise DataQualityError(f"Missing legacy odds columns: {missing}")
    result = frame[LEGACY_COLUMNS].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    for column in [
        "Best of", "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2",
        "B365_1", "B365_2", "Pinnacle_1", "Pinnacle_2", "Avg_1", "Avg_2",
        "Max_1", "Max_2", "Betfair_1", "Betfair_2",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in [
        "Tournament", "Series", "Court", "Surface", "Round", "Player_1", "Player_2",
        "Winner", "Score", "Status", "Odds_source",
    ]:
        result[column] = result[column].fillna("").astype(str).str.strip()
    result = result.dropna(subset=["Date"])
    result = result[result["Date"].dt.date <= today]
    result = result[
        result["Player_1"].ne("")
        & result["Player_2"].ne("")
        & result["Player_1"].ne(result["Player_2"])
        & (result["Winner"].eq(result["Player_1"]) | result["Winner"].eq(result["Player_2"]))
    ]
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    result = deterministic_orientation(result)
    pair = np.sort(result[["Player_1", "Player_2"]].astype(str).to_numpy(), axis=1)
    result["_dedup_key"] = [
        "|".join((day, tournament, round_name, left, right))
        for day, tournament, round_name, left, right in zip(
            result["Date"], result["Tournament"], result["Round"], pair[:, 0], pair[:, 1]
        )
    ]
    result = result.drop_duplicates("_dedup_key", keep="last").drop(columns="_dedup_key")
    return result.sort_values(["Date", "Tournament", "Round", "Player_1"]).reset_index(drop=True)


def fetch_tennis_mylife(
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SourceSnapshot, dict[str, Any]]:
    inventory = _http_json(TENNIS_MYLIFE_INVENTORY_URL)
    files = {item["name"]: item for item in inventory.get("files", [])}
    season_names = [f"{year}.csv" for year in range(start_year, end_year + 1)]
    required_names = season_names + ["ongoing_tourneys.csv"]
    missing = [name for name in required_names if name not in files]
    if missing:
        raise DataQualityError(f"Missing TennisMyLife files: {missing}")

    def download_csv(name: str) -> pd.DataFrame:
        item = files[name]
        data = _http_bytes(item["url"])
        frame = pd.read_csv(io.BytesIO(data), low_memory=False)
        frame["_source_file"] = name
        frame["_source_updated_at"] = item.get("mtime", "")
        return frame

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        frames = list(executor.map(download_csv, required_names))
    matches = pd.concat(frames, ignore_index=True, sort=False)

    ranking_candidates = [
        item for name, item in files.items() if re.fullmatch(r"atp_rankings_\d{4}-\d{2}-\d{2}\.csv", name)
    ]
    if ranking_candidates:
        ranking_item = max(ranking_candidates, key=lambda item: item["name"])
        rankings = pd.read_csv(io.BytesIO(_http_bytes(ranking_item["url"])), low_memory=False)
    else:
        ranking_item = {"name": "", "url": "", "mtime": ""}
        rankings = pd.DataFrame(columns=["playerId", "rank", "points"])

    updated_at = max(str(files[name].get("mtime", "")) for name in required_names)
    snapshot = SourceSnapshot(
        name="TennisMyLife match database",
        updated_at=updated_at,
        url=TENNIS_MYLIFE_INVENTORY_URL,
    )
    source_details = {
        "season_files": required_names,
        "ranking_file": ranking_item.get("name", ""),
        "ranking_updated_at": ranking_item.get("mtime", ""),
        "ranking_url": ranking_item.get("url", ""),
    }
    return matches, rankings, snapshot, source_details


def normalize_rich_matches(raw: pd.DataFrame, today: date, id_prefix: str = "atp:") -> pd.DataFrame:
    required = {
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "match_num",
        "winner_id",
        "winner_name",
        "loser_id",
        "loser_name",
        "score",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataQualityError(f"Missing rich match columns: {missing}")
    result = raw.copy()
    raw_dates = pd.to_numeric(result["tourney_date"], errors="coerce").astype("Int64").astype(str)
    result["match_date"] = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
    result = result.dropna(subset=["match_date", "winner_id", "loser_id", "winner_name", "loser_name"])
    result = result[result["match_date"].dt.date <= today]
    result["match_id"] = (
        id_prefix
        + result["tourney_id"].astype(str)
        + ":"
        + result["match_num"].astype(str)
        + ":"
        + result["winner_id"].astype(str)
        + ":"
        + result["loser_id"].astype(str)
    )
    # Ongoing files intentionally repeat matches already present in the season file.
    result = result.drop_duplicates("match_id", keep="last").copy()
    score = result["score"].fillna("").astype(str).str.upper()
    result["match_status"] = np.select(
        [score.str.contains(r"W/O|WALKOVER", regex=True), score.str.contains("RET"), score.str.contains("DEF")],
        ["walkover", "retired", "defaulted"],
        default="completed",
    )
    result["match_date"] = result["match_date"].dt.strftime("%Y-%m-%d")
    return result.sort_values(["match_date", "tourney_id", "match_num"]).reset_index(drop=True)


def _odds_match_candidates(
    rich: pd.DataFrame,
    odds: pd.DataFrame,
) -> list[tuple[float, int, int, float, float, float]]:
    odds_by_pair: dict[str, list[int]] = {}
    for index, pair in odds["_pair_key"].items():
        odds_by_pair.setdefault(pair, []).append(index)

    candidates: list[tuple[float, int, int, float, float, float]] = []
    for rich_index, row in rich.iterrows():
        for odds_index in odds_by_pair.get(row["_pair_key"], []):
            odds_row = odds.loc[odds_index]
            if row["_year"] != odds_row["_year"]:
                continue
            delta = abs((row["_date_ts"] - odds_row["_date_ts"]).days)
            if delta > 14:
                continue
            tournament_similarity = _tournament_similarity(row["tourney_name"], odds_row["Tournament"])
            round_equal = _round_key(row.get("round", "")) == _round_key(odds_row.get("Round", ""))
            if delta > 3 and tournament_similarity < 0.35 and not round_equal:
                continue
            score = 100.0 if delta == 0 else max(0.0, 45.0 - 2.0 * delta)
            score += 35.0 * tournament_similarity
            score += 15.0 if round_equal else 0.0
            if delta == 0 and tournament_similarity >= 0.35:
                confidence = 0.99
            elif delta == 0:
                confidence = 0.95
            elif delta <= 3 and (tournament_similarity >= 0.35 or round_equal):
                confidence = 0.90
            elif delta <= 3:
                confidence = 0.85
            elif tournament_similarity >= 0.80:
                confidence = 0.90
            elif tournament_similarity >= 0.35 and round_equal:
                confidence = 0.85
            elif tournament_similarity >= 0.35:
                confidence = 0.80
            else:
                confidence = 0.72
            candidates.append(
                (score, rich_index, odds_index, float(delta), tournament_similarity, confidence)
            )
    return sorted(candidates, reverse=True)


def attach_odds(rich_matches: pd.DataFrame, legacy_odds: pd.DataFrame) -> pd.DataFrame:
    rich = rich_matches.copy()
    odds = legacy_odds.copy()
    rich["_winner_key"] = rich["winner_name"].map(_full_name_key)
    rich["_loser_key"] = rich["loser_name"].map(_full_name_key)
    rich["_pair_key"] = [
        _pair_key(left, right) for left, right in zip(rich["_winner_key"], rich["_loser_key"])
    ]
    rich["_date_ts"] = pd.to_datetime(rich["match_date"])
    rich["_year"] = rich["_date_ts"].dt.year

    odds["_p1_key"] = odds["Player_1"].map(_abbreviated_name_key)
    odds["_p2_key"] = odds["Player_2"].map(_abbreviated_name_key)
    odds["_winner_key"] = odds["Winner"].map(_abbreviated_name_key)
    odds["_pair_key"] = [
        _pair_key(left, right) for left, right in zip(odds["_p1_key"], odds["_p2_key"])
    ]
    odds["_date_ts"] = pd.to_datetime(odds["Date"])
    odds["_year"] = odds["_date_ts"].dt.year

    rich["winner_odds"] = np.nan
    rich["loser_odds"] = np.nan
    rich["odds_match_confidence"] = np.nan
    rich["odds_date_delta_days"] = np.nan
    rich["odds_tournament_similarity"] = np.nan
    rich["odds_source_match_date"] = ""
    rich["odds_source_tournament"] = ""

    used_rich: set[int] = set()
    used_odds: set[int] = set()
    for _, rich_index, odds_index, delta, tournament_similarity, confidence in _odds_match_candidates(
        rich, odds
    ):
        if rich_index in used_rich or odds_index in used_odds:
            continue
        rich_row = rich.loc[rich_index]
        odds_row = odds.loc[odds_index]
        # Reject canonical-name collisions or source disagreements about the winner.
        if odds_row["_winner_key"] != rich_row["_winner_key"]:
            continue
        if odds_row["_p1_key"] == rich_row["_winner_key"]:
            winner_odds, loser_odds = odds_row["Odd_1"], odds_row["Odd_2"]
        elif odds_row["_p2_key"] == rich_row["_winner_key"]:
            winner_odds, loser_odds = odds_row["Odd_2"], odds_row["Odd_1"]
        else:
            continue
        rich.at[rich_index, "winner_odds"] = winner_odds
        rich.at[rich_index, "loser_odds"] = loser_odds
        rich.at[rich_index, "odds_match_confidence"] = confidence
        rich.at[rich_index, "odds_date_delta_days"] = delta
        rich.at[rich_index, "odds_tournament_similarity"] = tournament_similarity
        rich.at[rich_index, "odds_source_match_date"] = odds_row["Date"]
        rich.at[rich_index, "odds_source_tournament"] = odds_row["Tournament"]
        used_rich.add(rich_index)
        used_odds.add(odds_index)

    valid = rich["winner_odds"].gt(1.0) & rich["loser_odds"].gt(1.0)
    rich["market_overround"] = np.where(
        valid,
        1.0 / rich["winner_odds"] + 1.0 / rich["loser_odds"],
        np.nan,
    )
    rich["winner_market_prob_no_vig"] = np.where(
        valid,
        (1.0 / rich["winner_odds"]) / rich["market_overround"],
        np.nan,
    )
    rich["loser_market_prob_no_vig"] = np.where(
        valid,
        (1.0 / rich["loser_odds"]) / rich["market_overround"],
        np.nan,
    )
    return rich.drop(
        columns=["_winner_key", "_loser_key", "_pair_key", "_date_ts", "_year"],
        errors="ignore",
    )


def add_stable_player_orientation(enriched: pd.DataFrame) -> pd.DataFrame:
    """Add stable P1/P2 columns and mark post-match fields as such."""
    result = enriched.copy()
    p1_is_winner = []
    for row in result.itertuples(index=False):
        target = _stable_first_player(
            getattr(row, "match_date"),
            getattr(row, "tourney_id"),
            getattr(row, "round", ""),
            str(getattr(row, "winner_id")),
            str(getattr(row, "loser_id")),
        )
        p1_is_winner.append(target == str(getattr(row, "winner_id")))
    p1_wins = pd.Series(p1_is_winner, index=result.index)
    result["player_1_won"] = p1_wins.astype("int8")

    pre_fields = ["id", "name", "hand", "ht", "ioc", "age", "rank", "rank_points", "seed", "entry"]
    for field in pre_fields:
        winner_col = f"winner_{field}"
        loser_col = f"loser_{field}"
        if winner_col in result.columns and loser_col in result.columns:
            result[f"player_1_{field}"] = np.where(p1_wins, result[winner_col], result[loser_col])
            result[f"player_2_{field}"] = np.where(p1_wins, result[loser_col], result[winner_col])

    result["player_1_odds"] = np.where(p1_wins, result["winner_odds"], result["loser_odds"])
    result["player_2_odds"] = np.where(p1_wins, result["loser_odds"], result["winner_odds"])
    result["player_1_market_prob_no_vig"] = np.where(
        p1_wins,
        result["winner_market_prob_no_vig"],
        result["loser_market_prob_no_vig"],
    )
    result["player_2_market_prob_no_vig"] = np.where(
        p1_wins,
        result["loser_market_prob_no_vig"],
        result["winner_market_prob_no_vig"],
    )
    for stat in POSTMATCH_STATS:
        winner_col = f"w_{stat}"
        loser_col = f"l_{stat}"
        if winner_col in result.columns and loser_col in result.columns:
            result[f"postmatch_player_1_{stat}"] = np.where(
                p1_wins, result[winner_col], result[loser_col]
            )
            result[f"postmatch_player_2_{stat}"] = np.where(
                p1_wins, result[loser_col], result[winner_col]
            )
    return result


def build_current_players(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    winner = matches[
        ["match_date", "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age"]
    ].rename(
        columns={
            "winner_id": "player_id",
            "winner_name": "player_name",
            "winner_hand": "hand",
            "winner_ht": "height_cm",
            "winner_ioc": "nationality",
            "winner_age": "age_at_last_match",
        }
    )
    loser = matches[
        ["match_date", "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age"]
    ].rename(
        columns={
            "loser_id": "player_id",
            "loser_name": "player_name",
            "loser_hand": "hand",
            "loser_ht": "height_cm",
            "loser_ioc": "nationality",
            "loser_age": "age_at_last_match",
        }
    )
    players = pd.concat([winner, loser], ignore_index=True)
    players["player_id"] = players["player_id"].astype(str)
    players = players.sort_values("match_date").drop_duplicates("player_id", keep="last")
    if not rankings.empty:
        ranking_frame = rankings.rename(
            columns={"playerId": "player_id", "rank": "current_rank", "points": "current_rank_points"}
        ).copy()
        ranking_frame["player_id"] = ranking_frame["player_id"].astype(str)
        players = players.merge(ranking_frame, on="player_id", how="left")
    return players.sort_values(["current_rank", "player_name"], na_position="last").reset_index(drop=True)


def _odds_coverage(frame: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    odds_left = pd.to_numeric(frame[left], errors="coerce")
    odds_right = pd.to_numeric(frame[right], errors="coerce")
    valid_left = odds_left.gt(1.0)
    valid_right = odds_right.gt(1.0)
    valid = valid_left & valid_right
    overround = 1.0 / odds_left[valid] + 1.0 / odds_right[valid]
    return {
        "rows": int(len(frame)),
        "valid_pairs": int(valid.sum()),
        "coverage": float(valid.mean()),
        "one_sided_pairs": int((valid_left ^ valid_right).sum()),
        "overround_median": float(overround.median()) if len(overround) else None,
        "overround_below_0_95": int(overround.lt(0.95).sum()),
        "overround_above_1_25": int(overround.gt(1.25).sum()),
    }


def _score_winner_mismatches(frame: pd.DataFrame) -> tuple[int, int]:
    checked = 0
    mismatches = 0
    for row in frame.itertuples(index=False):
        if getattr(row, "Status", "completed") != "completed":
            continue
        sets = []
        for part in str(row.Score).split():
            match = re.match(r"^(\d+)-(\d+)", part)
            if match:
                sets.append((int(match.group(1)), int(match.group(2))))
        player_1_sets = sum(left > right for left, right in sets)
        player_2_sets = sum(right > left for left, right in sets)
        if not sets or player_1_sets == player_2_sets:
            continue
        checked += 1
        score_winner = row.Player_1 if player_1_sets > player_2_sets else row.Player_2
        mismatches += int(score_winner != row.Winner)
    return checked, mismatches


def build_quality_report(
    legacy: pd.DataFrame,
    rich: pd.DataFrame,
    enriched: pd.DataFrame,
    players: pd.DataFrame,
    today: date,
    odds_source: SourceSnapshot,
    stats_source: SourceSnapshot,
    source_details: dict[str, Any],
) -> dict[str, Any]:
    legacy_dates = pd.to_datetime(legacy["Date"])
    rich_dates = pd.to_datetime(rich["match_date"])
    legacy_pair = np.sort(legacy[["Player_1", "Player_2"]].astype(str).to_numpy(), axis=1)
    legacy_keys = [
        "|".join(values)
        for values in zip(
            legacy["Date"].astype(str),
            legacy["Tournament"].astype(str),
            legacy["Round"].astype(str),
            legacy_pair[:, 0],
            legacy_pair[:, 1],
        )
    ]
    yearly_odds: dict[str, Any] = {}
    for year, group in legacy.assign(_year=legacy_dates.dt.year).groupby("_year"):
        yearly_odds[str(int(year))] = _odds_coverage(group, "Odd_1", "Odd_2")
    score_rows_checked, score_winner_mismatches = _score_winner_mismatches(legacy)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": today.isoformat(),
        "sources": {
            "odds": odds_source.__dict__,
            "match_stats": stats_source.__dict__,
            **source_details,
        },
        "legacy_odds_dataset": {
            "rows": int(len(legacy)),
            "min_date": legacy_dates.min().date().isoformat(),
            "max_date": legacy_dates.max().date().isoformat(),
            "duplicate_match_keys": int(pd.Series(legacy_keys).duplicated().sum()),
            "winner_not_in_players": int(
                (~(legacy["Winner"].eq(legacy["Player_1"]) | legacy["Winner"].eq(legacy["Player_2"]))).sum()
            ),
            "score_rows_checked": score_rows_checked,
            "score_winner_mismatches": score_winner_mismatches,
            "status_counts": {
                str(status): int(count)
                for status, count in legacy.get(
                    "Status", pd.Series("completed", index=legacy.index)
                ).value_counts().items()
            },
            "odds": _odds_coverage(legacy, "Odd_1", "Odd_2"),
            "bookmaker_odds": {
                "bet365": _odds_coverage(legacy, "B365_1", "B365_2"),
                "pinnacle": _odds_coverage(legacy, "Pinnacle_1", "Pinnacle_2"),
                "market_average": _odds_coverage(legacy, "Avg_1", "Avg_2"),
                "market_maximum": _odds_coverage(legacy, "Max_1", "Max_2"),
                "betfair_exchange": _odds_coverage(legacy, "Betfair_1", "Betfair_2"),
            },
            "odds_by_year": yearly_odds,
        },
        "rich_match_dataset": {
            "rows": int(len(rich)),
            "min_date": rich_dates.min().date().isoformat(),
            "max_date": rich_dates.max().date().isoformat(),
            "duplicate_match_ids": int(rich["match_id"].duplicated().sum()),
            "completed": int(rich["match_status"].eq("completed").sum()),
            "retired": int(rich["match_status"].eq("retired").sum()),
            "walkovers": int(rich["match_status"].eq("walkover").sum()),
            "serve_stats_coverage": float(rich["w_svpt"].notna().mean()) if "w_svpt" in rich else 0.0,
            "players": int(len(players)),
        },
        "enrichment": {
            "rich_rows_with_matched_odds": int(enriched["odds_match_confidence"].notna().sum()),
            "rich_rows_with_valid_odds": int(
                (enriched["winner_odds"].gt(1.0) & enriched["loser_odds"].gt(1.0)).sum()
            ),
            "median_match_confidence": float(enriched["odds_match_confidence"].median()),
            "median_date_delta_days": float(enriched["odds_date_delta_days"].median()),
        },
    }
    return report


def validate_quality(report: dict[str, Any], today: date) -> None:
    legacy = report["legacy_odds_dataset"]
    rich = report["rich_match_dataset"]
    current_year_odds = legacy["odds_by_year"].get(str(today.year), {"coverage": 0.0})
    errors = []
    if legacy["rows"] < 50_000:
        errors.append("legacy odds snapshot unexpectedly contains fewer than 50,000 rows")
    if rich["rows"] < 60_000:
        errors.append("rich match snapshot unexpectedly contains fewer than 60,000 rows")
    if legacy["duplicate_match_keys"]:
        errors.append("legacy odds snapshot contains duplicate match keys")
    if rich["duplicate_match_ids"]:
        errors.append("rich match snapshot contains duplicate match ids")
    if legacy["winner_not_in_players"]:
        errors.append("legacy odds snapshot contains invalid winners")
    if legacy["score_winner_mismatches"] / max(legacy["score_rows_checked"], 1) > 0.001:
        errors.append("more than 0.1% of parsed scores disagree with the winner column")
    if legacy["odds"]["overround_below_0_95"] + legacy["odds"]["overround_above_1_25"] > 20:
        errors.append("more than 20 odds pairs have an implausible bookmaker overround")
    if current_year_odds["coverage"] < 0.90:
        errors.append(f"current-year odds coverage is only {current_year_odds['coverage']:.1%}")
    if (today - date.fromisoformat(legacy["max_date"])).days > 21:
        errors.append("legacy odds snapshot is more than 21 days old")
    if (today - date.fromisoformat(rich["max_date"])).days > 7:
        errors.append("rich match snapshot is more than 7 days old")
    if errors:
        raise DataQualityError("; ".join(errors))


def _atomic_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if gzip else ".csv"
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if gzip:
            # An empty gzip filename plus mtime=0 makes identical snapshots
            # byte-for-byte stable; random atomic temp names never leak into the
            # archive header and do not create noisy weekly Git diffs.
            with temporary.open("wb") as raw_stream:
                with gzip_module.GzipFile(
                    filename="", mode="wb", fileobj=raw_stream, mtime=0
                ) as compressed_stream:
                    with io.TextIOWrapper(compressed_stream, encoding="utf-8", newline="") as text_stream:
                        frame.to_csv(text_stream, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_change_summary(previous: pd.DataFrame | None, current: pd.DataFrame) -> dict[str, int]:
    if previous is None or previous.empty:
        return {"previous_rows": 0, "current_rows": int(len(current)), "rows_added": int(len(current)), "rows_removed": 0}

    def keys(frame: pd.DataFrame) -> set[str]:
        pair = np.sort(frame[["Player_1", "Player_2"]].astype(str).to_numpy(), axis=1)
        return {
            "|".join(values)
            for values in zip(
                frame["Date"].astype(str),
                frame["Tournament"].astype(str),
                frame["Round"].astype(str),
                pair[:, 0],
                pair[:, 1],
            )
        }

    old_keys, new_keys = keys(previous), keys(current)
    return {
        "previous_rows": int(len(previous)),
        "current_rows": int(len(current)),
        "rows_added": int(len(new_keys - old_keys)),
        "rows_removed": int(len(old_keys - new_keys)),
    }


def run_data_update(project_root: str | Path, as_of_date: date | None = None) -> dict[str, Any]:
    """Download, validate, enrich, and atomically publish every ATP table."""
    root = Path(project_root).resolve()
    today = as_of_date or datetime.now().date()
    legacy_path = root / "data" / "atp_tennis.csv"
    previous = pd.read_csv(legacy_path, low_memory=False) if legacy_path.exists() else None

    odds_raw, odds_source = fetch_odds_snapshot(2000, today.year)
    legacy = normalize_legacy_odds(transform_tennis_data_raw(odds_raw), today=today)
    rich_raw, rankings, stats_source, source_details = fetch_tennis_mylife(2000, today.year)
    rich = normalize_rich_matches(rich_raw, today=today)
    enriched = add_stable_player_orientation(attach_odds(rich, legacy))
    players = build_current_players(rich, rankings)
    report = build_quality_report(
        legacy,
        rich,
        enriched,
        players,
        today,
        odds_source,
        stats_source,
        source_details,
    )
    validate_quality(report, today=today)
    changes = _dataset_change_summary(previous, legacy)
    report["publication_changes"] = changes

    _atomic_csv(legacy, legacy_path)
    _atomic_csv(
        odds_raw,
        root / "data" / "raw" / "tennis_data" / "atp_odds_2000_current.csv.gz",
        gzip=True,
    )
    _atomic_csv(
        rich_raw,
        root / "data" / "raw" / "tennis_mylife" / "atp_matches_2000_current.csv.gz",
        gzip=True,
    )
    _atomic_csv(
        rankings,
        root / "data" / "raw" / "tennis_mylife" / "atp_rankings_current.csv",
    )
    _atomic_csv(
        enriched,
        root / "data" / "processed" / "atp_matches_enriched.csv.gz",
        gzip=True,
    )
    _atomic_csv(players, root / "data" / "processed" / "atp_players_current.csv")
    _atomic_json(report, root / "data" / "quality" / "atp_data_quality.json")
    _atomic_json(
        {
            "generated_at": report["generated_at"],
            "tables": {
                "data/atp_tennis.csv": {
                    "role": "legacy-compatible ATP match and pre-match odds table",
                    "rows": len(legacy),
                    "feature_time": "pre_match plus outcome and post-event settlement status",
                    "odds_semantics": (
                        "Odd_1/Odd_2 are a same-source market-average pair where available; "
                        "named bookmaker, maximum, and exchange prices are retained separately"
                    ),
                },
                "data/raw/tennis_data/atp_odds_2000_current.csv.gz": {
                    "role": "unaltered concatenation of official yearly Tennis-Data workbooks",
                    "rows": len(odds_raw),
                },
                "data/raw/tennis_mylife/atp_matches_2000_current.csv.gz": {
                    "role": "unaltered downloaded match-stat rows with source metadata",
                    "rows": len(rich_raw),
                },
                "data/processed/atp_matches_enriched.csv.gz": {
                    "role": "deduplicated rich matches joined to odds with confidence",
                    "rows": len(enriched),
                    "leakage_warning": "columns prefixed postmatch_ and outcome columns are unavailable before a match",
                },
                "data/processed/atp_players_current.csv": {
                    "role": "latest player profile and current ranking snapshot",
                    "rows": len(players),
                },
            },
            "column_groups": {
                "identifiers": ["match_id", "tourney_id", "match_num", "player_1_id", "player_2_id"],
                "pre_match": [
                    "match_date",
                    "surface",
                    "draw_size",
                    "tourney_level",
                    "indoor",
                    "best_of",
                    "round",
                    "player_1_hand",
                    "player_1_ht",
                    "player_1_ioc",
                    "player_1_age",
                    "player_1_rank",
                    "player_1_rank_points",
                    "player_2_hand",
                    "player_2_ht",
                    "player_2_ioc",
                    "player_2_age",
                    "player_2_rank",
                    "player_2_rank_points",
                    "player_1_odds",
                    "player_2_odds",
                    "market_overround",
                ],
                "outcome": ["winner_id", "winner_name", "loser_id", "loser_name", "player_1_won", "score", "match_status"],
                "post_match": [column for column in enriched.columns if column.startswith("postmatch_")]
                + ["minutes"],
            },
            "sources": report["sources"],
        },
        root / "data" / "data_manifest.json",
    )
    return report


def recalculate_elo_artifacts(
    project_root: str | Path,
    progress_callback=None,
) -> dict[str, Any]:
    """Rebuild legacy and v3 Elo artifacts from a clean state."""
    import joblib

    from src.features.elo_system import TennisEloEngine

    root = Path(project_root).resolve()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "data" / "atp_tennis.csv", low_memory=False)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Player_1", "Player_2", "Winner"])
    frame = frame.sort_values(["Date", "Tournament", "Round", "Player_1"]).reset_index(drop=True)
    if "Status" in frame:
        performance_frame = frame[frame["Status"].fillna("completed").eq("completed")].copy()
    else:
        performance_frame = frame

    def progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    progress("🧮 Calcul des Elo globaux (K=32)...")
    initial = 1500.0
    global_ratings: dict[str, float] = {}
    for row in performance_frame.itertuples(index=False):
        player_1 = str(row.Player_1)
        player_2 = str(row.Player_2)
        winner = str(row.Winner)
        rating_1 = global_ratings.get(player_1, initial)
        rating_2 = global_ratings.get(player_2, initial)
        expected_1 = 1.0 / (1.0 + 10.0 ** ((rating_2 - rating_1) / 400.0))
        score_1 = 1.0 if winner == player_1 else 0.0
        global_ratings[player_1] = rating_1 + 32.0 * (score_1 - expected_1)
        global_ratings[player_2] = rating_2 + 32.0 * ((1.0 - score_1) - (1.0 - expected_1))

    progress("🎾 Calcul des Elo par surface (K=40)...")
    surfaces = sorted(performance_frame["Surface"].dropna().astype(str).unique())
    surface_ratings: dict[str, dict[str, float]] = {surface: {} for surface in surfaces}
    for row in performance_frame.itertuples(index=False):
        surface = str(row.Surface)
        player_1 = str(row.Player_1)
        player_2 = str(row.Player_2)
        winner = str(row.Winner)
        rating_1 = surface_ratings[surface].get(player_1, initial)
        rating_2 = surface_ratings[surface].get(player_2, initial)
        expected_1 = 1.0 / (1.0 + 10.0 ** ((rating_2 - rating_1) / 400.0))
        score_1 = 1.0 if winner == player_1 else 0.0
        surface_ratings[surface][player_1] = rating_1 + 40.0 * (score_1 - expected_1)
        surface_ratings[surface][player_2] = rating_2 + 40.0 * (
            (1.0 - score_1) - (1.0 - expected_1)
        )

    progress("💾 Sauvegarde des Elo et profils legacy...")
    joblib.dump(
        {"global": global_ratings, "surface": surface_ratings},
        models_dir / "elo_ratings.pkl",
    )
    player_stats = {
        player: {
            "elo_global": rating,
            "elo_by_surface": {
                surface: values[player]
                for surface, values in surface_ratings.items()
                if player in values
            },
        }
        for player, rating in global_ratings.items()
    }
    joblib.dump(player_stats, models_dir / "player_stats.pkl")

    progress("🔄 Reconstruction propre de TennisEloEngine v3...")
    previous_path = models_dir / "elo_engine_v3.pkl"
    if previous_path.exists():
        previous = TennisEloEngine.load(str(previous_path))
        engine = TennisEloEngine(
            k_global=previous.k_global,
            k_surface=previous.k_surface,
            k_momentum=previous.k_momentum,
            initial_rating=previous.initial,
            half_life_days=previous.half_life,
            decay_enabled=previous.decay_enabled,
            min_decay_factor=previous.min_decay,
        )
    else:
        engine = TennisEloEngine()
    engine.fit(frame, progress_callback=progress_callback)
    with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".pkl", delete=False) as handle:
        temporary_engine = Path(handle.name)
    try:
        engine.save(str(temporary_engine))
        os.replace(temporary_engine, previous_path)
    finally:
        temporary_engine.unlink(missing_ok=True)

    progress("💾 Sauvegarde des matchs des 365 derniers jours...")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=365)
    recent_columns = [
        "Date",
        "Tournament",
        "Series",
        "Surface",
        "Round",
        "Best of",
        "Player_1",
        "Player_2",
        "Winner",
        "Rank_1",
        "Rank_2",
        "Pts_1",
        "Pts_2",
        "Score",
    ]
    recent = performance_frame.loc[performance_frame["Date"] >= cutoff, recent_columns].copy()
    _atomic_csv(recent, models_dir / "recent_matches.csv")
    return {
        "total_players": len(global_ratings),
        "total_matches": len(frame),
        "completed_matches_used_for_ratings": len(performance_frame),
        "recent_matches": len(recent),
        "top_players": sorted(global_ratings.items(), key=lambda item: item[1], reverse=True)[:10],
    }
