"""Construction d'une base UFC versionnee avec resultats, profils et cotes.

Les combats/profils historiques proviennent d'une extraction UFCStats publique.
La liste des evenements termines et les evenements manquants sont ensuite verifies
et completes directement sur UFCStats. Les cotes sont conservees avec leur source
et leur horodatage; aucune ligne posterieure au cutoff pre-combat n'est eligible.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


UFCSTATS_COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/DanMcInerney/mma-ai/main/"
    "data/raw/ufcstats/competitions.csv"
)
UFCSTATS_FIGHTERS_URL = (
    "https://raw.githubusercontent.com/DanMcInerney/mma-ai/main/"
    "data/raw/ufcstats/individuals.csv"
)
UFCSTATS_REPO_API = "https://api.github.com/repos/DanMcInerney/mma-ai/commits/main"
RANKINGS_URL = (
    "https://raw.githubusercontent.com/martj42/ufc_rankings_history/"
    "master/rankings_history.csv"
)
RANKINGS_REPO_API = "https://api.github.com/repos/martj42/ufc_rankings_history/commits/master"
ODDS_DATASET_SLUG = "jerzyszocik/ufc-betting-odds-daily-dataset"
ODDS_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{ODDS_DATASET_SLUG}"
ODDS_METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{ODDS_DATASET_SLUG}"
COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"

RANKED_DIVISIONS = (
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Light Heavyweight",
    "Bantamweight",
    "Featherweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Heavyweight",
    "Flyweight",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*values: object) -> str:
    text = "|".join(str(v or "").strip() for v in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def id_from_url(value: object) -> str:
    return str(value or "").rstrip("/").split("/")[-1]


def normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def canonical_weight_class(value: object) -> str:
    """Ramene les combats de titre vers leur division classee."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^UFC\s+", "", text, flags=re.I)
    text = re.sub(r"^Interim\s+", "", text, flags=re.I)
    text = re.sub(r"\s+Title$", "", text, flags=re.I)
    return text if text in RANKED_DIVISIONS else ""


def fight_key(date: object, fighter_1: object, fighter_2: object) -> str:
    parsed = pd.Timestamp(date).strftime("%Y-%m-%d")
    names = sorted((normalise_name(fighter_1), normalise_name(fighter_2)))
    return f"{parsed}|{names[0]}|{names[1]}"


def parse_pair(value: object) -> tuple[float, float]:
    match = re.match(r"\s*([0-9.]+)\s+of\s+([0-9.]+)\s*$", str(value or ""))
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def parse_clock(value: object) -> float:
    match = re.match(r"\s*(\d+):(\d{2})\s*$", str(value or ""))
    if not match:
        return np.nan
    return float(int(match.group(1)) * 60 + int(match.group(2)))


def parse_height_cm(value: object) -> float:
    match = re.match(r"\s*(\d+)\s*'\s*(\d+)\s*\"?", str(value or ""))
    if not match:
        return np.nan
    return (int(match.group(1)) * 12 + int(match.group(2))) * 2.54


def parse_inches_cm(value: object) -> float:
    match = re.search(r"([0-9.]+)", str(value or ""))
    return float(match.group(1)) * 2.54 if match else np.nan


def parse_weight_lbs(value: object) -> float:
    match = re.search(r"([0-9.]+)", str(value or ""))
    return float(match.group(1)) if match else np.nan


def _sum_numeric(row: pd.Series, prefix: str, metric: str) -> float:
    values = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_rd{round_no}_{metric}") for round_no in range(1, 6)]),
        errors="coerce",
    )
    return float(values.sum(min_count=1))


def _sum_pairs(row: pd.Series, prefix: str, metric: str) -> tuple[float, float]:
    pairs = [parse_pair(row.get(f"{prefix}_rd{round_no}_{metric}")) for round_no in range(1, 6)]
    landed = pd.Series([pair[0] for pair in pairs], dtype=float).sum(min_count=1)
    attempted = pd.Series([pair[1] for pair in pairs], dtype=float).sum(min_count=1)
    return float(landed), float(attempted)


def _sum_clocks(row: pd.Series, prefix: str) -> float:
    values = pd.Series(
        [parse_clock(row.get(f"{prefix}_rd{round_no}_Ctrl")) for round_no in range(1, 6)],
        dtype=float,
    )
    return float(values.sum(min_count=1))


def _fight_duration_seconds(round_value: object, clock_value: object) -> float:
    round_no = pd.to_numeric(pd.Series([round_value]), errors="coerce").iloc[0]
    clock = parse_clock(clock_value)
    if pd.isna(round_no) or pd.isna(clock):
        return np.nan
    return float(max(0, int(round_no) - 1) * 300 + clock)


class UFCStatsClient:
    """Client requests avec resolution du petit proof-of-work UFCStats."""

    def __init__(self, delay_seconds: float = 0.25) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self.delay_seconds = delay_seconds
        self.last_request = 0.0

    def _paced_get(self, url: str) -> requests.Response:
        delay = self.delay_seconds - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        response = self.session.get(url, timeout=30)
        self.last_request = time.monotonic()
        response.raise_for_status()
        return response

    def get(self, url: str) -> requests.Response:
        response = self._paced_get(url)
        if "Checking your browser" not in response.text:
            return response

        nonce_match = re.search(r'nonce\s*=\s*"([0-9a-fA-F]+)"', response.text)
        target_match = re.search(r"target\s*=\s*new Array\((\d+)\+1\)", response.text)
        if not nonce_match:
            raise RuntimeError("Defi UFCStats detecte mais nonce introuvable")
        nonce = nonce_match.group(1)
        target_len = int(target_match.group(1)) if target_match else 2
        target = "0" * target_len
        proof = 0
        while proof < 50_000_000:
            if hashlib.sha256(f"{nonce}:{proof}".encode()).hexdigest().startswith(target):
                break
            proof += 1
        if proof >= 50_000_000:
            raise RuntimeError("Defi UFCStats non resolu")
        challenge_url = requests.compat.urljoin(url, "/__c")
        solved = self.session.post(challenge_url, data={"nonce": nonce, "n": proof}, timeout=30)
        solved.raise_for_status()
        response = self._paced_get(url)
        if "Checking your browser" in response.text:
            raise RuntimeError("UFCStats refuse la session apres resolution du defi")
        return response


@dataclass(frozen=True)
class EventRecord:
    name: str
    date: pd.Timestamp
    url: str


def get_completed_events(client: UFCStatsClient) -> list[EventRecord]:
    soup = BeautifulSoup(client.get(COMPLETED_EVENTS_URL).text, "html.parser")
    events: list[EventRecord] = []
    for row in soup.select("table.b-statistics__table-events tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date = row.select_one(".b-statistics__date")
        if not link or not date:
            continue
        parsed = pd.to_datetime(date.get_text(" ", strip=True), format="%B %d, %Y", errors="coerce")
        if pd.notna(parsed):
            events.append(EventRecord(link.get_text(" ", strip=True), parsed, str(link.get("href"))))
    if not events:
        raise RuntimeError("Aucun evenement termine lu sur UFCStats")
    return events


def _two_texts(cell: Any) -> list[str]:
    values = [item.get_text(" ", strip=True) for item in cell.find_all("p", recursive=False)]
    if len(values) < 2:
        values = [part.strip() for part in cell.get_text("|", strip=True).split("|")]
    return values[:2]


def _parse_event_index(client: UFCStatsClient, event: EventRecord) -> tuple[str, list[dict[str, Any]]]:
    soup = BeautifulSoup(client.get(event.url).text, "html.parser")
    location = ""
    for item in soup.select(".b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if text.lower().startswith("location"):
            location = re.sub(r"^location\s*:?\s*", "", text, flags=re.I)
    fights: list[dict[str, Any]] = []
    for row in soup.select("table.b-fight-details__table tbody tr[data-link]"):
        cells = row.find_all("td", recursive=False)
        fighter_links = cells[1].select("a.b-link") if len(cells) > 1 else []
        if len(cells) < 10 or len(fighter_links) != 2:
            continue
        method_values = _two_texts(cells[7])
        fights.append(
            {
                "fight_url": str(row.get("data-link")),
                "event_name": event.name,
                "event_date": event.date,
                "event_url": event.url,
                "event_location": location,
                "red_name": fighter_links[0].get_text(" ", strip=True),
                "blue_name": fighter_links[1].get_text(" ", strip=True),
                "red_url": str(fighter_links[0].get("href")),
                "blue_url": str(fighter_links[1].get("href")),
                "weight_class": cells[6].get_text(" ", strip=True),
                "method": method_values[0] if method_values else "",
                "details": method_values[1] if len(method_values) > 1 else "",
                "round": cells[8].get_text(" ", strip=True),
                "time": cells[9].get_text(" ", strip=True),
            }
        )
    return location, fights


def _parse_fight_detail(client: UFCStatsClient, indexed: dict[str, Any]) -> dict[str, Any]:
    soup = BeautifulSoup(client.get(indexed["fight_url"]).text, "html.parser")
    statuses: dict[str, str] = {}
    urls: dict[str, str] = {}
    for person in soup.select(".b-fight-details__person"):
        link = person.select_one(".b-fight-details__person-name a")
        status = person.select_one(".b-fight-details__person-status")
        if link:
            name = link.get_text(" ", strip=True)
            urls[normalise_name(name)] = str(link.get("href"))
            statuses[normalise_name(name)] = status.get_text(" ", strip=True).upper() if status else ""

    totals_table = soup.find("table")
    row = totals_table.select_one("tbody tr") if totals_table else None
    cells = row.find_all("td", recursive=False) if row else []
    if len(cells) < 10:
        raise ValueError(f"Table de totaux absente: {indexed['fight_url']}")
    names = [a.get_text(" ", strip=True) for a in cells[0].select("a")]
    if len(names) != 2:
        names = _two_texts(cells[0])
    if len(names) != 2:
        raise ValueError(f"Deux combattants introuvables: {indexed['fight_url']}")

    columns = ["kd", "sig", "sig_pct", "total", "td", "td_pct", "sub", "rev", "ctrl"]
    values: dict[str, list[str]] = {}
    for column, cell in zip(columns, cells[1:10]):
        pair = _two_texts(cell)
        if len(pair) != 2:
            pair = ["", ""]
        values[column] = pair

    fighter_rows: list[dict[str, Any]] = []
    for side in range(2):
        sig_lnd, sig_att = parse_pair(values["sig"][side])
        total_lnd, total_att = parse_pair(values["total"][side])
        td_lnd, td_att = parse_pair(values["td"][side])
        name = names[side]
        fighter_rows.append(
            {
                "name": name,
                "url": urls.get(normalise_name(name), ""),
                "result": statuses.get(normalise_name(name), ""),
                "kd": pd.to_numeric(values["kd"][side], errors="coerce"),
                "sig_lnd": sig_lnd,
                "sig_att": sig_att,
                "total_lnd": total_lnd,
                "total_att": total_att,
                "td_lnd": td_lnd,
                "td_att": td_att,
                "sub_att": pd.to_numeric(values["sub"][side], errors="coerce"),
                "rev": pd.to_numeric(values["rev"][side], errors="coerce"),
                "ctrl_secs": parse_clock(values["ctrl"][side]),
            }
        )
    return {**indexed, "fighters": fighter_rows}


def scrape_missing_events(
    client: UFCStatsClient,
    events: Iterable[EventRecord],
    after_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    missing = sorted((event for event in events if event.date > after_date), key=lambda event: event.date)
    rows: list[dict[str, Any]] = []
    for event in missing:
        _, indexed_fights = _parse_event_index(client, event)
        print(f"  UFCStats {event.date.date()} — {event.name}: {len(indexed_fights)} combats", flush=True)
        for indexed in indexed_fights:
            rows.append(_parse_fight_detail(client, indexed))
    return rows


def canonicalise_source_fights(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        event_date = pd.to_datetime(row.get("event_date"), format="%B %d, %Y", errors="coerce")
        if pd.isna(event_date):
            continue
        p1: dict[str, Any] = {}
        p2: dict[str, Any] = {}
        for prefix, output in (("p1", p1), ("p2", p2)):
            output["kd"] = _sum_numeric(row, prefix, "KD")
            output["sig_lnd"], output["sig_att"] = _sum_pairs(row, prefix, "Sig_str")
            output["total_lnd"], output["total_att"] = _sum_pairs(row, prefix, "Total_str")
            output["td_lnd"], output["td_att"] = _sum_pairs(row, prefix, "Td")
            output["sub_att"] = _sum_numeric(row, prefix, "Sub_att")
            output["rev"] = _sum_numeric(row, prefix, "Rev")
            output["ctrl_secs"] = _sum_clocks(row, prefix)
        event_url = str(row.get("event_url") or "")
        p1_url = str(row.get("player1_url") or "")
        p2_url = str(row.get("player2_url") or "")
        result = str(row.get("result") or "").strip().upper()
        fight_id = stable_id(event_url, *sorted((p1_url, p2_url)))
        rows.append(
            {
                "fight_id": fight_id,
                "fight_url": "",
                "event_id": id_from_url(event_url),
                "event_url": event_url,
                "event_name": "",
                "event_date": event_date,
                "event_location": str(row.get("event_location") or ""),
                "weight_class": re.sub(r"\s+Bout$", "", str(row.get("weightclass") or "")),
                "method": str(row.get("method") or ""),
                "details": str(row.get("details") or ""),
                "finish_round": pd.to_numeric(row.get("round"), errors="coerce"),
                "finish_time": str(row.get("time") or ""),
                "duration_secs": _fight_duration_seconds(row.get("round"), row.get("time")),
                "fighter_1": str(row.get("player1") or ""),
                "fighter_2": str(row.get("player2") or ""),
                "fighter_1_url": p1_url,
                "fighter_2_url": p2_url,
                "fighter_1_id": id_from_url(p1_url),
                "fighter_2_id": id_from_url(p2_url),
                "result_1": result,
                "y": 1.0 if result == "W" else 0.0 if result == "L" else np.nan,
                **{f"p1_{key}": value for key, value in p1.items()},
                **{f"p2_{key}": value for key, value in p2.items()},
                "record_source": "DanMcInerney/mma-ai UFCStats extract",
            }
        )
    return pd.DataFrame(rows)


def canonicalise_supplemental_fights(source: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fight in source:
        fighters = fight["fighters"]
        if len(fighters) != 2:
            continue
        p1, p2 = fighters
        result = str(p1.get("result") or "").upper()
        fight_url = str(fight["fight_url"])
        rows.append(
            {
                "fight_id": id_from_url(fight_url),
                "fight_url": fight_url,
                "event_id": id_from_url(fight["event_url"]),
                "event_url": fight["event_url"],
                "event_name": fight["event_name"],
                "event_date": fight["event_date"],
                "event_location": fight["event_location"],
                "weight_class": fight["weight_class"],
                "method": fight["method"],
                "details": fight["details"],
                "finish_round": pd.to_numeric(fight["round"], errors="coerce"),
                "finish_time": fight["time"],
                "duration_secs": _fight_duration_seconds(fight["round"], fight["time"]),
                "fighter_1": p1["name"],
                "fighter_2": p2["name"],
                "fighter_1_url": p1["url"],
                "fighter_2_url": p2["url"],
                "fighter_1_id": id_from_url(p1["url"]),
                "fighter_2_id": id_from_url(p2["url"]),
                "result_1": result,
                "y": 1.0 if result == "W" else 0.0 if result == "L" else np.nan,
                **{f"p1_{key}": p1.get(key) for key in ("kd", "sig_lnd", "sig_att", "total_lnd", "total_att", "td_lnd", "td_att", "sub_att", "rev", "ctrl_secs")},
                **{f"p2_{key}": p2.get(key) for key in ("kd", "sig_lnd", "sig_att", "total_lnd", "total_att", "td_lnd", "td_att", "sub_att", "rev", "ctrl_secs")},
                "record_source": "UFCStats direct",
            }
        )
    return pd.DataFrame(rows)


def canonicalise_fighters(source: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "fighter_id": source["url"].map(id_from_url),
            "fighter_url": source["url"],
            "fighter_name": source["name"],
            "nickname": source["nickname"].replace("--", np.nan),
            "dob": pd.to_datetime(source["dob"], format="%b %d, %Y", errors="coerce"),
            "height_cm": source["height"].map(parse_height_cm),
            "reach_cm": source["reach"].map(parse_inches_cm),
            "weight_lbs": source["weight"].map(parse_weight_lbs),
            "stance": source["stance"].replace("--", np.nan),
        }
    )
    return result.drop_duplicates("fighter_id", keep="last").reset_index(drop=True)


def scrape_fighter_profiles(client: UFCStatsClient, urls: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for url in sorted(set(str(url) for url in urls if str(url))):
        soup = BeautifulSoup(client.get(url).text, "html.parser")
        name_node = soup.select_one(".b-content__title-highlight")
        name = name_node.get_text(" ", strip=True) if name_node else ""
        values: dict[str, str] = {}
        for item in soup.select(".b-list__box-list-item"):
            text = " ".join(item.stripped_strings)
            if ":" in text:
                key, value = text.split(":", 1)
                values[key.strip().lower()] = value.strip()
        rows.append(
            {
                "fighter_id": id_from_url(url),
                "fighter_url": url,
                "fighter_name": name,
                "nickname": np.nan,
                "dob": pd.to_datetime(values.get("dob"), format="%b %d, %Y", errors="coerce"),
                "height_cm": parse_height_cm(values.get("height")),
                "reach_cm": parse_inches_cm(values.get("reach")),
                "weight_lbs": parse_weight_lbs(values.get("weight")),
                "stance": values.get("stance") or np.nan,
            }
        )
    return pd.DataFrame(rows)


def canonicalise_rankings(source: pd.DataFrame) -> pd.DataFrame:
    """Conserve l'historique complet et cree des cles de jointure auditables."""
    rankings = source.rename(columns={"date": "ranking_date"}).copy()
    rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], errors="coerce")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["fighter"] = rankings["fighter"].astype(str).str.strip()
    rankings["fighter_key"] = rankings["fighter"].map(normalise_name)
    rankings["division_key"] = rankings["weightclass"].map(canonical_weight_class)
    rankings["ranking_bucket"] = rankings["division_key"]
    rankings.loc[rankings["weightclass"].isin(["Pound-for-Pound", "Men's Pound-for-Pound"]), "ranking_bucket"] = "p4p_men"
    rankings.loc[rankings["weightclass"].eq("Women's Pound-for-Pound"), "ranking_bucket"] = "p4p_women"
    rankings = rankings.dropna(subset=["ranking_date", "rank"])
    rankings = rankings[rankings["fighter_key"].ne("") & rankings["ranking_bucket"].ne("")]
    rankings = rankings.drop_duplicates(
        ["ranking_date", "ranking_bucket", "fighter_key"], keep="last"
    )
    return rankings.sort_values(["ranking_date", "ranking_bucket", "fighter_key"]).reset_index(drop=True)


def _attach_one_ranking(
    fights: pd.DataFrame,
    rankings: pd.DataFrame,
    fighter_column: str,
    bucket: pd.Series,
    prefix: str,
    maximum_age_days: int = 14,
) -> pd.DataFrame:
    """Jointure as-of stricte: jamais le snapshot du jour ou un snapshot futur."""
    left = pd.DataFrame(
        {
            "_row": np.arange(len(fights)),
            "event_date": pd.to_datetime(fights["event_date"], errors="coerce"),
            "fighter_key": fights[fighter_column].map(normalise_name),
            "ranking_bucket": bucket,
        }
    )
    right = rankings[["ranking_date", "ranking_bucket", "fighter_key", "rank"]].copy()
    left = left.sort_values(["event_date", "ranking_bucket", "fighter_key"])
    right = right.sort_values(["ranking_date", "ranking_bucket", "fighter_key"])
    joined = pd.merge_asof(
        left,
        right,
        left_on="event_date",
        right_on="ranking_date",
        by=["ranking_bucket", "fighter_key"],
        direction="backward",
        allow_exact_matches=False,
    )
    joined[f"{prefix}_snapshot_age_days"] = (
        joined["event_date"] - joined["ranking_date"]
    ).dt.total_seconds() / 86_400.0
    stale = joined[f"{prefix}_snapshot_age_days"].gt(maximum_age_days)
    joined.loc[stale, ["rank", "ranking_date", f"{prefix}_snapshot_age_days"]] = np.nan
    joined = joined.sort_values("_row")
    return pd.DataFrame(
        {
            f"{prefix}": joined["rank"].to_numpy(),
            f"{prefix}_snapshot_date": joined["ranking_date"].to_numpy(),
            f"{prefix}_snapshot_age_days": joined[f"{prefix}_snapshot_age_days"].to_numpy(),
        }
    )


def attach_prefight_rankings(fights: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Ajoute rangs de division et pound-for-pound connus avant chaque combat."""
    fights = fights.reset_index(drop=True)
    result = fights[["fight_id", "event_date", "fighter_1", "fighter_2", "weight_class"]].copy()
    division_bucket = fights["weight_class"].map(canonical_weight_class)
    p4p_bucket = np.where(division_bucket.str.startswith("Women's"), "p4p_women", "p4p_men")

    division_rankings = rankings[rankings["division_key"].ne("")]
    p4p_rankings = rankings[rankings["ranking_bucket"].isin(["p4p_men", "p4p_women"])]
    for side in (1, 2):
        joined = _attach_one_ranking(
            fights, division_rankings, f"fighter_{side}", division_bucket,
            f"division_rank_{side}",
        )
        result = pd.concat([result, joined], axis=1)
        joined_p4p = _attach_one_ranking(
            fights, p4p_rankings, f"fighter_{side}", pd.Series(p4p_bucket, index=fights.index),
            f"p4p_rank_{side}",
        )
        result = pd.concat([result, joined_p4p], axis=1)

    for family in ("division_rank", "p4p_rank"):
        for side in (1, 2):
            rank = result[f"{family}_{side}"]
            result[f"{family}_known_{side}"] = rank.notna()
            result[f"{family}_points_{side}"] = np.where(rank.notna(), 16.0 - rank, 0.0)
        result[f"{family}_points_diff"] = (
            result[f"{family}_points_1"] - result[f"{family}_points_2"]
        )
        result[f"{family}_known_count"] = (
            result[f"{family}_known_1"].astype(int) + result[f"{family}_known_2"].astype(int)
        )
    return result


def match_odds_to_fights(odds: pd.DataFrame, fights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    odds = odds.copy()
    odds["event_date"] = pd.to_datetime(odds["event_date"], errors="coerce")
    odds["fight_key"] = [fight_key(d, a, b) for d, a, b in zip(odds.event_date, odds.fighter_1, odds.fighter_2)]
    fight_map = fights[["fight_id", "event_date", "fighter_1", "fighter_2"]].copy()
    fight_map["fight_key"] = [fight_key(d, a, b) for d, a, b in zip(fight_map.event_date, fight_map.fighter_1, fight_map.fighter_2)]
    collisions = int(fight_map["fight_key"].duplicated(keep=False).sum())
    fight_map = fight_map.drop_duplicates("fight_key", keep=False)
    matched = odds.merge(fight_map[["fight_key", "fight_id", "fighter_1", "fighter_2"]], on="fight_key", how="inner", suffixes=("_odds", "_official"))
    same_orientation = matched["fighter_1_odds"].map(normalise_name) == matched["fighter_1_official"].map(normalise_name)
    matched["odds_fighter_1"] = np.where(same_orientation, matched["odds_1"], matched["odds_2"])
    matched["odds_fighter_2"] = np.where(same_orientation, matched["odds_2"], matched["odds_1"])
    matched["collected_at"] = pd.to_datetime(matched["adding_date"], format="mixed", utc=True, errors="coerce")
    matched["source"] = matched["source"].fillna("unknown").astype(str)
    matched["region"] = matched["region"].fillna("unknown").astype(str)
    event_utc = matched["event_date"].dt.tz_localize("UTC")
    matched["temporal_quality"] = np.select(
        [matched["source"].eq("zewnetrzne"), matched["collected_at"].lt(event_utc)],
        ["legacy_unverified", "timestamped_pre_event"],
        default="post_event_or_unknown",
    )
    keep = [
        "fight_id", "fight_key", "event_date", "fighter_1_official", "fighter_2_official",
        "odds_fighter_1", "odds_fighter_2", "f1_ko_odds", "f2_ko_odds", "f1_sub_odds",
        "f2_sub_odds", "f1_dec_odds", "f2_dec_odds", "collected_at", "source", "region",
        "temporal_quality",
    ]
    matched = matched[keep].rename(columns={"fighter_1_official": "fighter_1", "fighter_2_official": "fighter_2"})
    matched = matched[
        matched["odds_fighter_1"].between(1.01, 101) & matched["odds_fighter_2"].between(1.01, 101)
    ].reset_index(drop=True)
    report = {
        "odds_input_rows": int(len(odds)),
        "odds_matched_rows": int(len(matched)),
        "odds_matched_fights": int(matched["fight_id"].nunique()),
        "fight_key_collisions_excluded": collisions,
        "coverage_fights": float(matched["fight_id"].nunique() / max(1, fights["fight_id"].nunique())),
        "temporal_quality": {str(k): int(v) for k, v in matched["temporal_quality"].value_counts().items()},
    }
    return matched, report


def choose_lines(odds: pd.DataFrame) -> pd.DataFrame:
    """Ligne predeclaree: legacy unique; sinon Pinnacle puis BetOnline, cutoff J-1.

    La priorite de bookmaker evite de choisir retrospectivement le meilleur prix.
    Une cote horodatee doit avoir ete observee au plus tard 24 h avant la date UFC
    et pas plus de 14 jours avant ce cutoff.
    """
    legacy = odds[odds["temporal_quality"].eq("legacy_unverified")].copy()
    legacy = legacy.sort_values(["fight_id", "collected_at"]).drop_duplicates("fight_id", keep="last")
    legacy["line_protocol"] = "legacy_unverified_single_line"

    timed = odds[odds["temporal_quality"].eq("timestamped_pre_event")].copy()
    timed["cutoff"] = timed["event_date"].dt.tz_localize("UTC") - pd.Timedelta(days=1)
    timed = timed[
        timed["collected_at"].le(timed["cutoff"])
        & ((timed["cutoff"] - timed["collected_at"]) <= pd.Timedelta(days=14))
        & timed["source"].isin(["Pinnacle", "BetOnline.ag"])
    ].copy()
    timed["book_priority"] = timed["source"].map({"Pinnacle": 0, "BetOnline.ag": 1})
    timed = timed.sort_values(["fight_id", "book_priority", "collected_at"], ascending=[True, True, False])
    timed = timed.drop_duplicates("fight_id", keep="first")
    timed["line_protocol"] = "pinnacle_else_betonline_at_Tminus24h_max_age14d"

    lines = pd.concat([legacy, timed], ignore_index=True, sort=False)
    inv1 = 1.0 / lines["odds_fighter_1"]
    inv2 = 1.0 / lines["odds_fighter_2"]
    lines["market_p1"] = inv1 / (inv1 + inv2)
    lines["overround"] = inv1 + inv2
    return lines.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def choose_method_props(odds: pd.DataFrame) -> pd.DataFrame:
    """Conserve les props disponibles sans leur donner une qualite non acquise."""
    prop_columns = [
        "f1_ko_odds", "f2_ko_odds", "f1_sub_odds", "f2_sub_odds",
        "f1_dec_odds", "f2_dec_odds",
    ]
    available = odds.copy()
    for column in prop_columns:
        available[column] = pd.to_numeric(available[column], errors="coerce")
    available = available[available[prop_columns].ge(1.01).any(axis=1)]

    legacy = available[available["temporal_quality"].eq("legacy_unverified")].copy()
    legacy = legacy.sort_values(["fight_id", "collected_at"]).drop_duplicates("fight_id", keep="last")
    legacy["props_protocol"] = "legacy_unverified_single_snapshot"

    timed = available[available["temporal_quality"].eq("timestamped_pre_event")].copy()
    timed["cutoff"] = timed["event_date"].dt.tz_localize("UTC") - pd.Timedelta(days=1)
    timed = timed[
        timed["collected_at"].le(timed["cutoff"])
        & ((timed["cutoff"] - timed["collected_at"]) <= pd.Timedelta(days=14))
        & timed["source"].isin(["Pinnacle", "BetOnline.ag"])
    ].copy()
    timed["book_priority"] = timed["source"].map({"Pinnacle": 0, "BetOnline.ag": 1})
    timed = timed.sort_values(
        ["fight_id", "book_priority", "collected_at"], ascending=[True, True, False]
    ).drop_duplicates("fight_id", keep="first")
    timed["props_protocol"] = "pinnacle_else_betonline_at_Tminus24h_max_age14d"
    return pd.concat([legacy, timed], ignore_index=True, sort=False).sort_values(
        ["event_date", "fight_id"]
    ).reset_index(drop=True)


def build_line_trajectories(odds: pd.DataFrame) -> pd.DataFrame:
    """Snapshots pre-combat fixes, issus d'un meme bookmaker par combat.

    Le snapshot retenu est le dernier disponible avant chaque horizon, avec une
    anciennete maximale de trois jours. La priorite de bookmaker est fixe et ne
    depend ni du prix obtenu ni du resultat du combat.
    """
    timed = odds[
        odds["temporal_quality"].eq("timestamped_pre_event")
        & odds["source"].isin(["Pinnacle", "BetOnline.ag"])
    ].copy()
    timed["collected_at"] = pd.to_datetime(timed["collected_at"], utc=True, errors="coerce")
    timed = timed.dropna(subset=["collected_at"])
    horizons = (14, 7, 3, 1)
    priorities = ("Pinnacle", "BetOnline.ag")
    rows: list[dict[str, Any]] = []
    for fight_id, fight_quotes in timed.groupby("fight_id", sort=False):
        selected: dict[str, Any] | None = None
        for source in priorities:
            book = fight_quotes[fight_quotes["source"].eq(source)].sort_values("collected_at")
            if book.empty:
                continue
            event_date = pd.Timestamp(book["event_date"].iloc[0]).tz_localize("UTC")
            row: dict[str, Any] = {
                "fight_id": fight_id,
                "event_date": book["event_date"].iloc[0],
                "fighter_1": book["fighter_1"].iloc[0],
                "fighter_2": book["fighter_2"].iloc[0],
                "source": source,
                "region": book["region"].iloc[0],
                "trajectory_protocol": "same_book_fixed_horizons_max_snapshot_age3d",
            }
            observed_times: set[pd.Timestamp] = set()
            for horizon in horizons:
                cutoff = event_date - pd.Timedelta(days=horizon)
                eligible = book[
                    book["collected_at"].le(cutoff)
                    & book["collected_at"].ge(cutoff - pd.Timedelta(days=3))
                ]
                if eligible.empty:
                    continue
                quote = eligible.iloc[-1]
                inv1 = 1.0 / float(quote["odds_fighter_1"])
                inv2 = 1.0 / float(quote["odds_fighter_2"])
                suffix = f"tminus_{horizon}d"
                row[f"observed_at_{suffix}"] = quote["collected_at"]
                row[f"snapshot_age_hours_{suffix}"] = float(
                    (cutoff - quote["collected_at"]).total_seconds() / 3600.0
                )
                row[f"odds_1_{suffix}"] = float(quote["odds_fighter_1"])
                row[f"odds_2_{suffix}"] = float(quote["odds_fighter_2"])
                row[f"market_p1_{suffix}"] = inv1 / (inv1 + inv2)
                row[f"overround_{suffix}"] = inv1 + inv2
                observed_times.add(quote["collected_at"])
            # J-1 est obligatoire; une trajectoire exige deux observations distinctes.
            if "market_p1_tminus_1d" in row and len(observed_times) >= 2:
                selected = row
                break
        if selected is not None:
            for earlier in (14, 7, 3):
                selected[f"market_move_p1_{earlier}d_to_1d"] = (
                    selected.get("market_p1_tminus_1d", np.nan)
                    - selected.get(f"market_p1_tminus_{earlier}d", np.nan)
                )
            rows.append(selected)
    return pd.DataFrame(rows).sort_values(["event_date", "fight_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def _quality_report(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    odds: pd.DataFrame,
    lines: pd.DataFrame,
    prefight_rankings: pd.DataFrame,
    trajectories: pd.DataFrame,
    props: pd.DataFrame,
) -> dict[str, Any]:
    valid = fights["y"].notna()
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fights": int(len(fights)),
        "decisive_fights": int(valid.sum()),
        "events": int(fights["event_id"].nunique()),
        "fighters_in_fights": int(pd.unique(pd.concat([fights.fighter_1_id, fights.fighter_2_id])).size),
        "fighter_profiles": int(len(fighters)),
        "date_min": str(fights["event_date"].min().date()),
        "date_max": str(fights["event_date"].max().date()),
        "duplicate_fight_ids": int(fights["fight_id"].duplicated().sum()),
        "invalid_result_rows": int((~fights["result_1"].isin(["W", "L", "D", "NC", ""])).sum()),
        "stats_coverage": {
            col: float(fights[col].notna().mean())
            for col in ["p1_sig_lnd", "p2_sig_lnd", "p1_td_lnd", "p2_td_lnd", "p1_ctrl_secs", "p2_ctrl_secs"]
        },
        "odds_snapshot_rows_matched": int(len(odds)),
        "odds_fights_matched": int(odds["fight_id"].nunique()),
        "selected_line_fights": int(lines["fight_id"].nunique()),
        "selected_line_protocols": {str(k): int(v) for k, v in lines["line_protocol"].value_counts().items()},
        "selected_line_sources": {str(k): int(v) for k, v in lines["source"].value_counts().items()},
        "prefight_rankings": {
            "fights_with_at_least_one_division_rank": int(
                prefight_rankings["division_rank_known_count"].gt(0).sum()
            ),
            "fights_with_two_division_ranks": int(
                prefight_rankings["division_rank_known_count"].eq(2).sum()
            ),
            "future_or_same_day_snapshots": int(sum(
                (
                    pd.to_datetime(prefight_rankings[column], errors="coerce")
                    >= pd.to_datetime(prefight_rankings["event_date"], errors="coerce")
                ).sum()
                for column in [
                    "division_rank_1_snapshot_date", "division_rank_2_snapshot_date",
                    "p4p_rank_1_snapshot_date", "p4p_rank_2_snapshot_date",
                ]
            )),
            "maximum_allowed_snapshot_age_days": 14,
        },
        "line_trajectory_fights": int(len(trajectories)),
        "line_trajectory_complete_t14_to_t1": int(
            trajectories.get("market_p1_tminus_14d", pd.Series(dtype=float)).notna().sum()
        ),
        "method_props_fights": int(props["fight_id"].nunique()) if not props.empty else 0,
        "method_props_temporal_quality": (
            {str(k): int(v) for k, v in props["temporal_quality"].value_counts().items()}
            if not props.empty else {}
        ),
    }
    return report


def update_dataset(base_dir: Path) -> dict[str, Any]:
    raw_dir = base_dir / "data" / "rigorous" / "raw"
    processed_dir = base_dir / "data" / "rigorous" / "processed"
    quality_dir = base_dir / "data" / "rigorous" / "quality"
    for directory in (raw_dir, processed_dir, quality_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    print("Telechargement de l'extraction UFCStats historique...", flush=True)
    competitions_response = session.get(UFCSTATS_COMPETITIONS_URL, timeout=120)
    competitions_response.raise_for_status()
    competitions_bytes = competitions_response.content
    fighters_response = session.get(UFCSTATS_FIGHTERS_URL, timeout=120)
    fighters_response.raise_for_status()
    fighters_bytes = fighters_response.content
    repo_meta_response = session.get(UFCSTATS_REPO_API, timeout=30)
    repo_meta_response.raise_for_status()
    repo_meta = repo_meta_response.json()
    competitions_source = pd.read_csv(io.BytesIO(competitions_bytes), low_memory=False)
    fighters_source = pd.read_csv(io.BytesIO(fighters_bytes), low_memory=False)

    print("Telechargement de l'historique hebdomadaire des classements UFC...", flush=True)
    rankings_response = session.get(RANKINGS_URL, timeout=120)
    rankings_response.raise_for_status()
    rankings_bytes = rankings_response.content
    rankings_meta_response = session.get(RANKINGS_REPO_API, timeout=30)
    rankings_meta_response.raise_for_status()
    rankings_meta = rankings_meta_response.json()
    rankings_source = pd.read_csv(io.BytesIO(rankings_bytes), low_memory=False)
    rankings = canonicalise_rankings(rankings_source)

    source_fights = canonicalise_source_fights(competitions_source)
    source_max = source_fights["event_date"].max()
    print(f"Extraction secondaire arretee au {source_max.date()}; verification UFCStats...", flush=True)
    client = UFCStatsClient()
    official_events = get_completed_events(client)
    official_latest = max(event.date for event in official_events)
    supplemental_raw = scrape_missing_events(client, official_events, source_max)
    supplemental = canonicalise_supplemental_fights(supplemental_raw)
    fights = pd.concat([source_fights, supplemental], ignore_index=True, sort=False)
    fights = fights.sort_values(["event_date", "event_id", "fight_id"]).drop_duplicates(
        "fight_id", keep="last"
    ).reset_index(drop=True)
    event_names = {event.url: event.name for event in official_events}
    fights["event_name"] = fights["event_url"].map(event_names).fillna(fights["event_name"])
    fights["fight_key"] = [fight_key(d, a, b) for d, a, b in zip(fights.event_date, fights.fighter_1, fights.fighter_2)]

    fighters = canonicalise_fighters(fighters_source)
    fighter_ids = set(fighters["fighter_id"])
    fight_profiles = pd.concat(
        [
            fights[["fighter_1_id", "fighter_1_url"]].rename(columns={"fighter_1_id": "fighter_id", "fighter_1_url": "fighter_url"}),
            fights[["fighter_2_id", "fighter_2_url"]].rename(columns={"fighter_2_id": "fighter_id", "fighter_2_url": "fighter_url"}),
        ],
        ignore_index=True,
    ).drop_duplicates("fighter_id")
    missing_profile_urls = fight_profiles.loc[~fight_profiles["fighter_id"].isin(fighter_ids), "fighter_url"]
    if len(missing_profile_urls):
        print(f"Completion de {len(missing_profile_urls)} profils UFCStats recents...", flush=True)
        direct_profiles = scrape_fighter_profiles(client, missing_profile_urls)
        fighters = pd.concat([fighters, direct_profiles], ignore_index=True).drop_duplicates("fighter_id", keep="last")

    print("Telechargement des snapshots de cotes...", flush=True)
    odds_meta_response = session.get(ODDS_METADATA_URL, timeout=30)
    odds_meta_response.raise_for_status()
    odds_meta = odds_meta_response.json()
    odds_download_response = session.get(ODDS_DOWNLOAD_URL, timeout=180)
    odds_download_response.raise_for_status()
    odds_zip_bytes = odds_download_response.content
    with zipfile.ZipFile(io.BytesIO(odds_zip_bytes)) as archive:
        odds_csv_bytes = archive.read("UFC_betting_odds.csv")
    odds_source = pd.read_csv(io.BytesIO(odds_csv_bytes), low_memory=False)
    matched_odds, odds_match_report = match_odds_to_fights(odds_source, fights)
    lines = choose_lines(matched_odds)
    props = choose_method_props(matched_odds)
    trajectories = build_line_trajectories(matched_odds)
    prefight_rankings = attach_prefight_rankings(fights, rankings)

    source_comp_path = raw_dir / "ufcstats_competitions.parquet"
    source_fighters_path = raw_dir / "ufcstats_fighters.parquet"
    source_rankings_path = raw_dir / "ufc_rankings_history.parquet"
    source_odds_path = raw_dir / "ufc_odds_snapshots_matched.parquet"
    fights_path = processed_dir / "fights.parquet"
    fighters_path = processed_dir / "fighters.parquet"
    lines_path = processed_dir / "moneyline_quotes.parquet"
    props_path = processed_dir / "method_props_quotes.parquet"
    trajectories_path = processed_dir / "moneyline_trajectories.parquet"
    prefight_rankings_path = processed_dir / "prefight_rankings.parquet"
    competitions_source.to_parquet(source_comp_path, index=False)
    fighters_source.to_parquet(source_fighters_path, index=False)
    rankings.to_parquet(source_rankings_path, index=False)
    matched_odds.to_parquet(source_odds_path, index=False)
    fights.to_parquet(fights_path, index=False)
    fighters.to_parquet(fighters_path, index=False)
    lines.to_parquet(lines_path, index=False)
    props.to_parquet(props_path, index=False)
    trajectories.to_parquet(trajectories_path, index=False)
    prefight_rankings.to_parquet(prefight_rankings_path, index=False)

    quality = _quality_report(
        fights, fighters, matched_odds, lines, prefight_rankings, trajectories, props
    )
    quality.update(
        {
            "official_latest_completed_event": str(official_latest.date()),
            "secondary_source_latest_event": str(source_max.date()),
            "direct_ufcstats_supplemental_fights": int(len(supplemental)),
            "odds_matching": odds_match_report,
        }
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ufcstats": "http://ufcstats.com",
            "historical_extract_repository": "https://github.com/DanMcInerney/mma-ai",
            "historical_extract_commit": repo_meta.get("sha"),
            "historical_competitions_url": UFCSTATS_COMPETITIONS_URL,
            "historical_competitions_download_sha256": sha256_bytes(competitions_bytes),
            "historical_fighters_url": UFCSTATS_FIGHTERS_URL,
            "historical_fighters_download_sha256": sha256_bytes(fighters_bytes),
            "rankings_repository": "https://github.com/martj42/ufc_rankings_history",
            "rankings_repository_commit": rankings_meta.get("sha"),
            "rankings_url": RANKINGS_URL,
            "rankings_download_sha256": sha256_bytes(rankings_bytes),
            "rankings_date_min": str(rankings["ranking_date"].min().date()),
            "rankings_date_max": str(rankings["ranking_date"].max().date()),
            "odds_dataset": f"https://www.kaggle.com/datasets/{ODDS_DATASET_SLUG}",
            "odds_dataset_version": odds_meta.get("currentVersionNumber"),
            "odds_dataset_last_updated": odds_meta.get("lastUpdated"),
            "odds_dataset_license": odds_meta.get("licenseName"),
            "odds_csv_download_sha256": sha256_bytes(odds_csv_bytes),
        },
        "artifacts": {},
        "limitations": [
            "Les cotes legacy 2010-2024 n'ont ni bookmaker ni timestamp pre-combat verifiables.",
            "Les cotes horodatees sont des snapshots gratuits, pas une garantie d'execution.",
            "Les profils UFCStats actuels peuvent differer de la valeur affichee historiquement.",
            "Les classements historiques proviennent d'une archive tierce du classement UFC publie.",
            "Les props legacy n'ont pas d'horodatage verifiable et sont exclus des preuves economiques.",
        ],
    }
    for path in (
        source_comp_path, source_fighters_path, source_rankings_path, source_odds_path,
        fights_path, fighters_path, lines_path, props_path, trajectories_path,
        prefight_rankings_path,
    ):
        manifest["artifacts"][str(path.relative_to(base_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (quality_dir / "data_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")
    (quality_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"quality": quality, "manifest": manifest}
