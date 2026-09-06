"""Football odds ingestion — a market the tennis and UFC work never touched.

Football-Data publishes, for 22 European divisions, something neither the tennis
nor the UFC sources provide:

* **Opening and closing prices for the same match.** Every other dataset in this
  repository carries one undated price, which made the question "could you have
  beaten the closing line?" untestable. Here it is a column subtraction.
* **Three markets instead of one.** Beside the 1X2 market there is over/under
  2.5 goals and the Asian handicap, whose margin is roughly a third of a tennis
  moneyline's. A bias too small to clear 6% may still clear 2%.
* **Divisions of very different depth.** The Premier League and the English
  fourth tier are priced by the same books with very different attention.

The normalisation below keeps every price separately and never mixes two books
into one pair. Opening and closing stay in distinct columns: collapsing them
would destroy the only timing information the project has ever had.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.tennis_pipeline import DataQualityError, _atomic_csv, _atomic_json


FOOTBALL_DATA_BASE_URL = "https://football-data.co.uk"
SEASON_FILE_URL = FOOTBALL_DATA_BASE_URL + "/mmz4281/{season}/{league}.csv"

# Division codes as Football-Data publishes them, with the depth of coverage
# that matters when asking where a market is thinnest.
LEAGUES = {
    "E0": ("England", 1), "E1": ("England", 2), "E2": ("England", 3),
    "E3": ("England", 4), "EC": ("England", 5),
    "SC0": ("Scotland", 1), "SC1": ("Scotland", 2), "SC2": ("Scotland", 3), "SC3": ("Scotland", 4),
    "D1": ("Germany", 1), "D2": ("Germany", 2),
    "I1": ("Italy", 1), "I2": ("Italy", 2),
    "SP1": ("Spain", 1), "SP2": ("Spain", 2),
    "F1": ("France", 1), "F2": ("France", 2),
    "N1": ("Netherlands", 1), "B1": ("Belgium", 1), "P1": ("Portugal", 1),
    "T1": ("Turkey", 1), "G1": ("Greece", 1),
}

# Pinnacle opening and closing appear from 2015/16; the full market set from
# 2019/20. Earlier seasons are still downloaded for results and 1X2 prices.
FIRST_SEASON_START_YEAR = 2000

# One entry per (market, book, timing). The pair or triple must always come from
# the same book at the same moment, never assembled across sources.
PRICE_GROUPS = {
    "1x2": {
        "bet365_open": ("B365H", "B365D", "B365A"),
        "bet365_close": ("B365CH", "B365CD", "B365CA"),
        "pinnacle_open": ("PSH", "PSD", "PSA"),
        "pinnacle_close": ("PSCH", "PSCD", "PSCA"),
        "market_average_open": ("AvgH", "AvgD", "AvgA"),
        "market_average_close": ("AvgCH", "AvgCD", "AvgCA"),
        "market_maximum_open": ("MaxH", "MaxD", "MaxA"),
        "market_maximum_close": ("MaxCH", "MaxCD", "MaxCA"),
        # The exchange is not a bookmaker: its price carries no built-in margin,
        # and the cost of using it is a commission taken on net winnings only.
        # It is therefore the one venue where a bias smaller than a bookmaker's
        # margin could still survive.
        "betfair_exchange_open": ("BFEH", "BFED", "BFEA"),
        "betfair_exchange_close": ("BFECH", "BFECD", "BFECA"),
    },
    "over_under_25": {
        "bet365_open": ("B365>2.5", "B365<2.5"),
        "bet365_close": ("B365C>2.5", "B365C<2.5"),
        "pinnacle_open": ("P>2.5", "P<2.5"),
        "pinnacle_close": ("PC>2.5", "PC<2.5"),
        "market_average_open": ("Avg>2.5", "Avg<2.5"),
        "market_average_close": ("AvgC>2.5", "AvgC<2.5"),
        "betfair_exchange_open": ("BFE>2.5", "BFE<2.5"),
        "betfair_exchange_close": ("BFEC>2.5", "BFEC<2.5"),
    },
    "asian_handicap": {
        "bet365_open": ("B365AHH", "B365AHA"),
        "bet365_close": ("B365CAHH", "B365CAHA"),
        "pinnacle_open": ("PAHH", "PAHA"),
        "pinnacle_close": ("PCAHH", "PCAHA"),
        "market_average_open": ("AvgAHH", "AvgAHA"),
        "market_average_close": ("AvgCAHH", "AvgCAHA"),
        "betfair_exchange_open": ("BFEAHH", "BFEAHA"),
        "betfair_exchange_close": ("BFECAHH", "BFECAHA"),
    },
}

# Betfair's standard commission on net market winnings. The exchange price is
# published gross, so the commission is the haircut — and unlike a bookmaker's
# margin it applies to winnings only, never to a losing stake.
BETFAIR_COMMISSION = 0.05
BETFAIR_COMMISSION_DISCOUNTED = 0.02
HANDICAP_LINES = {"open": "AHh", "close": "AHCh"}

CORE_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Match statistics. They describe what happened *during* the match, so they are
# published with a ``postmatch_`` prefix and may only ever enter a model as a
# lagged team aggregate — never for the match being predicted.
MATCH_STATISTICS = {
    "HS": "postmatch_home_shots", "AS": "postmatch_away_shots",
    "HST": "postmatch_home_shots_on_target", "AST": "postmatch_away_shots_on_target",
    "HC": "postmatch_home_corners", "AC": "postmatch_away_corners",
    "HF": "postmatch_home_fouls", "AF": "postmatch_away_fouls",
    "HY": "postmatch_home_yellow", "AY": "postmatch_away_yellow",
    "HR": "postmatch_home_red", "AR": "postmatch_away_red",
}


def _season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


# Football-Data answers a request for a season a division did not play with a
# 300 "Multiple Choices" rather than a 404, and sometimes serves its homepage
# with a 200. Both mean "no such file", not "the download failed".
MISSING_FILE_STATUSES = {300, 403, 404}


def _http_bytes(url: str, timeout: int = 90, attempts: int = 3) -> bytes | None:
    """Download, retrying transient failures. Returns None for a missing file."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, headers={"User-Agent": "tennis-modele-data-pipeline/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            # A CSV starts with its header; an HTML page is the site telling us
            # the file does not exist while answering 200.
            if payload.lstrip()[:1] == b"<":
                return None
            return payload
        except urllib.error.HTTPError as error:
            if error.code in MISSING_FILE_STATUSES:
                return None
            last_error = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    raise DataQualityError(f"Download failed for {url}: {last_error}")


def fetch_season(league: str, start_year: int) -> pd.DataFrame | None:
    payload = _http_bytes(SEASON_FILE_URL.format(season=_season_code(start_year), league=league))
    if payload is None or not payload.strip():
        return None
    frame = pd.read_csv(
        io.BytesIO(payload), encoding="latin-1", on_bad_lines="skip", low_memory=False
    )
    if not {"HomeTeam", "AwayTeam"}.issubset(frame.columns):
        return None
    frame["_league"] = league
    frame["_season_start"] = start_year
    return frame


def fetch_all(start_year: int, end_year: int, progress=None) -> pd.DataFrame:
    tasks = [
        (league, year)
        for league in LEAGUES
        for year in range(start_year, end_year + 1)
    ]

    def download(task: tuple[str, int]) -> pd.DataFrame | None:
        league, year = task
        return fetch_season(league, year)

    frames: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for index, frame in enumerate(executor.map(download, tasks)):
            if progress and index % 50 == 0:
                progress(f"Football-Data: {index}/{len(tasks)} fichiers")
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise DataQualityError("No football season file could be downloaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize(raw: pd.DataFrame, today: date) -> pd.DataFrame:
    """One row per match, results and every price group kept separate."""
    missing = sorted(set(CORE_COLUMNS) - set(raw.columns))
    if missing:
        raise DataQualityError(f"Missing football columns: {missing}")

    frame = raw.copy()
    # Football-Data mixes two-digit and four-digit years across eras.
    parsed = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(frame["Date"], format="%d/%m/%y", errors="coerce")
    frame["match_date"] = parsed.fillna(fallback)
    frame = frame.dropna(subset=["match_date"]).copy()
    frame = frame[frame["match_date"].dt.date <= today]

    frame["home_team"] = frame["HomeTeam"].astype(str).str.strip()
    frame["away_team"] = frame["AwayTeam"].astype(str).str.strip()
    frame = frame[frame["home_team"].ne("") & frame["away_team"].ne("")]
    frame = frame[frame["home_team"].ne(frame["away_team"])]

    for column, target in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        frame[target] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
    frame["result"] = frame["FTR"].astype(str).str.upper().str[0]
    frame = frame[frame["result"].isin(["H", "D", "A"])]
    # The published result must agree with the published score.
    derived = np.select(
        [frame["home_goals"] > frame["away_goals"], frame["home_goals"] < frame["away_goals"]],
        ["H", "A"], default="D",
    )
    frame = frame[frame["result"].to_numpy() == derived].copy()
    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]
    frame["goal_difference"] = frame["home_goals"] - frame["away_goals"]

    frame["country"] = frame["_league"].map(lambda code: LEAGUES[code][0])
    frame["division_rank"] = frame["_league"].map(lambda code: LEAGUES[code][1])
    frame["season_start"] = frame["_season_start"]
    frame["league"] = frame["_league"]
    frame["match_id"] = (
        frame["league"].astype(str) + ":"
        + frame["match_date"].dt.strftime("%Y%m%d") + ":"
        + frame["home_team"] + ":" + frame["away_team"]
    )

    published = [
        "match_id", "match_date", "league", "country", "division_rank", "season_start",
        "home_team", "away_team", "home_goals", "away_goals", "total_goals",
        "goal_difference", "result",
    ]
    for source, target in MATCH_STATISTICS.items():
        frame[target] = pd.to_numeric(frame.get(source), errors="coerce")
        published.append(target)
    for market, books in PRICE_GROUPS.items():
        for book, columns in books.items():
            for column in columns:
                if column not in frame.columns:
                    frame[column] = np.nan
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                published.append(column)
    for timing, column in HANDICAP_LINES.items():
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        published.append(column)

    frame = frame.drop_duplicates("match_id", keep="last")
    frame["match_date"] = frame["match_date"].dt.strftime("%Y-%m-%d")
    return frame[published].sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)


def _group_coverage(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    prices = frame[list(columns)]
    complete = prices.notna().all(axis=1) & prices.gt(1.0).all(axis=1)
    overround = (1.0 / prices[complete]).sum(axis=1) if complete.any() else pd.Series(dtype=float)
    return {
        "complete_rows": int(complete.sum()),
        "coverage": float(complete.mean()),
        "overround_median": float(overround.median()) if len(overround) else None,
        "implied_arbitrage_rate": float((overround < 1.0).mean()) if len(overround) else None,
    }


def build_quality_report(frame: pd.DataFrame, today: date) -> dict[str, Any]:
    dates = pd.to_datetime(frame["match_date"])
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(today),
        "matches": int(len(frame)),
        "leagues": int(frame["league"].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "duplicate_match_ids": int(len(frame) - frame["match_id"].nunique()),
        "result_distribution": {
            key: float(value) for key, value in frame["result"].value_counts(normalize=True).items()
        },
        "matches_by_league": {
            key: int(value) for key, value in frame["league"].value_counts().sort_index().items()
        },
        "markets": {},
    }
    for market, books in PRICE_GROUPS.items():
        report["markets"][market] = {
            book: _group_coverage(frame, columns) for book, columns in books.items()
        }
    both = (
        frame[list(PRICE_GROUPS["1x2"]["pinnacle_open"])].notna().all(axis=1)
        & frame[list(PRICE_GROUPS["1x2"]["pinnacle_close"])].notna().all(axis=1)
    )
    report["matches_with_pinnacle_open_and_close"] = int(both.sum())
    return report


def validate(report: dict[str, Any], today: date) -> None:
    if report["duplicate_match_ids"] != 0:
        raise DataQualityError("Duplicate football match identifiers")
    if report["matches"] < 100_000:
        raise DataQualityError(f"Implausibly few football matches: {report['matches']}")
    if report["matches_with_pinnacle_open_and_close"] < 20_000:
        raise DataQualityError(
            "Too few matches carry both a Pinnacle opening and closing price: "
            f"{report['matches_with_pinnacle_open_and_close']}"
        )
    home_rate = report["result_distribution"].get("H", 0.0)
    if not 0.35 <= home_rate <= 0.55:
        raise DataQualityError(f"Implausible home win rate: {home_rate}")
    newest = pd.to_datetime(report["date_max"]).date()
    if (today - newest).days > 30:
        raise DataQualityError(f"Stale football data, newest match is {newest}")


def run_football_update(
    project_root: str | Path, as_of_date: date | None = None, progress=None
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    today = as_of_date or datetime.now().date()
    raw = fetch_all(FIRST_SEASON_START_YEAR, today.year, progress=progress)
    frame = normalize(raw, today=today)
    report = build_quality_report(frame, today=today)
    validate(report, today=today)

    _atomic_csv(frame, root / "data" / "football" / "football_matches.csv.gz", gzip=True)
    _atomic_csv(raw, root / "data" / "raw" / "football_data" / "football_raw.csv.gz", gzip=True)
    _atomic_json(report, root / "data" / "quality" / "football_data_quality.json")
    return report
