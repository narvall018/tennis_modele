"""Upcoming UFC cards, and the descriptor model's opinion on them.

The app previously said this tab needed a paid odds key. That was only half
true: a *price* needs one, but the fight list does not. UFCStats publishes the
scheduled cards, and this package already carries a client that answers the
site's proof-of-work challenge, so the fixtures are reachable with no key at all.

What the key still buys is the market price, and with it the gap between the
model and the market. Without it the page shows an opinion and says so — which
is more useful than an empty tab claiming nothing can be done.

Fighter state is read from the same feature table the model was trained on, so a
prediction here uses exactly the descriptors it was fitted on and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .data_pipeline import UFCStatsClient, id_from_url, normalise_name


UPCOMING_EVENTS_URL = "http://ufcstats.com/statistics/events/upcoming"


@dataclass(frozen=True)
class UpcomingFight:
    event_name: str
    event_date: pd.Timestamp
    fighter_1: str
    fighter_2: str
    fighter_1_id: str
    fighter_2_id: str
    weight_class: str


def fetch_upcoming_events(client: UFCStatsClient, limit: int = 4) -> list[tuple[str, pd.Timestamp, str]]:
    """The next scheduled cards, nearest first."""
    soup = BeautifulSoup(client.get(UPCOMING_EVENTS_URL).text, "html.parser")
    events: list[tuple[str, pd.Timestamp, str]] = []
    for row in soup.select("table.b-statistics__table-events tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date_cell = row.select_one(".b-statistics__date")
        if not link or not date_cell:
            continue
        parsed = pd.to_datetime(
            date_cell.get_text(" ", strip=True), format="%B %d, %Y", errors="coerce"
        )
        if pd.notna(parsed):
            events.append((link.get_text(" ", strip=True), parsed, str(link.get("href"))))
    events.sort(key=lambda item: item[1])
    return events[:limit]


def fetch_card(client: UFCStatsClient, event_name: str, event_date: pd.Timestamp,
               url: str) -> list[UpcomingFight]:
    """Parse one scheduled card into its bouts."""
    soup = BeautifulSoup(client.get(url).text, "html.parser")
    fights: list[UpcomingFight] = []
    for row in soup.select("tr.b-fight-details__table-row"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        names = [item.get_text(" ", strip=True) for item in cells[1].select("p")]
        links = [str(item.get("href")) for item in cells[1].select("a")]
        if len(names) < 2 or len(links) < 2:
            continue
        weight_class = cells[6].get_text(" ", strip=True) if len(cells) > 6 else ""
        fights.append(UpcomingFight(
            event_name=event_name,
            event_date=event_date,
            fighter_1=names[0],
            fighter_2=names[1],
            fighter_1_id=id_from_url(links[0]),
            fighter_2_id=id_from_url(links[1]),
            weight_class=re.sub(r"\s+", " ", weight_class).strip(),
        ))
    return fights


def export_state_table(states: dict[str, Any], profiles: dict[str, dict[str, Any]],
                       as_of: pd.Timestamp) -> pd.DataFrame:
    """Freeze fighter state as plain columns, portable across environments.

    Pickling the state objects tied the file to one pandas and one importable
    module: the app runs on Python 3.9 / pandas 2.2 while training happens on
    3.12 / pandas 3.0, and the pickle failed on both counts. A parquet of
    numbers and ISO dates has neither problem.
    """
    from .model_pipeline import _state_features

    rows: list[dict[str, Any]] = []
    for fighter_id, state in states.items():
        computed = _state_features(state, as_of)
        profile = profiles.get(fighter_id, {})
        last = getattr(state, "last_date", None)
        rows.append({
            "fighter_id": str(fighter_id),
            # layoff is recomputed per fixture date, so only the anchor is stored
            **{key: value for key, value in computed.items() if key != "layoff_days"},
            "last_date": pd.Timestamp(last).date().isoformat() if pd.notna(last) else "",
            "reach_cm": pd.to_numeric(profile.get("reach_cm"), errors="coerce"),
            "height_cm": pd.to_numeric(profile.get("height_cm"), errors="coerce"),
            "stance": str(profile.get("stance") or ""),
            "dob": (
                pd.Timestamp(profile.get("dob")).date().isoformat()
                if pd.notna(pd.to_datetime(profile.get("dob"), errors="coerce")) else ""
            ),
        })
    return pd.DataFrame(rows)


def features_from_state_table(
    fights: list[UpcomingFight], table: pd.DataFrame
) -> pd.DataFrame:
    """Descriptors for scheduled bouts, from the portable state table."""
    from .model_pipeline import STATS_FEATURES

    indexed = table.set_index("fighter_id")
    level_keys = [
        "elo", "experience", "career_win_rate", "recent_win_rate", "sig_landed_pm",
        "sig_absorbed_pm", "sig_accuracy", "td_landed_p15", "td_accuracy",
        "sub_attempts_p15", "ctrl_share", "kd_p15",
    ]
    rows: list[dict[str, Any]] = []
    for fight in fights:
        left_id, right_id = fight.fighter_1_id, fight.fighter_2_id
        known = left_id in indexed.index and right_id in indexed.index
        row: dict[str, Any] = {"both_known": known}
        if not known:
            for column in STATS_FEATURES:
                row[column] = np.nan
            row["elo_1"] = row["elo_2"] = np.nan
            row["experience_1"] = row["experience_2"] = np.nan
            rows.append(row)
            continue
        left, right = indexed.loc[left_id], indexed.loc[right_id]
        for key in level_keys:
            row[f"{key}_diff"] = float(left[key]) - float(right[key])

        def layoff(record: Any) -> float:
            stamp = pd.to_datetime(record["last_date"], errors="coerce")
            return float((fight.event_date - stamp).days) if pd.notna(stamp) else np.nan

        row["layoff_days_diff"] = layoff(left) - layoff(right)
        row["reach_diff"] = float(left["reach_cm"]) - float(right["reach_cm"])
        row["height_diff"] = float(left["height_cm"]) - float(right["height_cm"])

        def age(record: Any) -> float:
            born = pd.to_datetime(record["dob"], errors="coerce")
            return float((fight.event_date - born).days / 365.25) if pd.notna(born) else np.nan

        row["age_diff"] = age(left) - age(right)
        stance_1, stance_2 = str(left["stance"] or ""), str(right["stance"] or "")
        row["stance_matchup"] = float(bool(stance_1 and stance_2 and stance_1 != stance_2))
        row["elo_1"] = float(left["elo"])
        row["elo_2"] = float(right["elo"])
        row["experience_1"] = float(left["experience"])
        row["experience_2"] = float(right["experience"])
        rows.append(row)
    return pd.DataFrame(rows)


def features_for_upcoming(
    fights: list[UpcomingFight],
    states: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Descriptors for scheduled bouts, from the model's own end-of-history state.

    A fighter absent from the state table keeps NaN rather than a division
    average: a debutant is exactly the case where an invented prior would be most
    confidently wrong.
    """
    from .model_pipeline import STATS_FEATURES, _age, _state_features

    rows: list[dict[str, Any]] = []
    for fight in fights:
        first = states.get(fight.fighter_1_id)
        second = states.get(fight.fighter_2_id)
        row: dict[str, Any] = {
            "fighter_1_id": fight.fighter_1_id,
            "fighter_2_id": fight.fighter_2_id,
            "both_known": first is not None and second is not None,
        }
        if first is None or second is None:
            for column in STATS_FEATURES:
                row[column] = np.nan
            rows.append(row)
            continue

        left = _state_features(first, fight.event_date)
        right = _state_features(second, fight.event_date)
        for key in (
            "elo", "experience", "career_win_rate", "recent_win_rate", "sig_landed_pm",
            "sig_absorbed_pm", "sig_accuracy", "td_landed_p15", "td_accuracy",
            "sub_attempts_p15", "ctrl_share", "kd_p15", "layoff_days",
        ):
            row[f"{key}_diff"] = left[key] - right[key]
        profile_1 = profiles.get(fight.fighter_1_id, {})
        profile_2 = profiles.get(fight.fighter_2_id, {})
        row["reach_diff"] = (
            pd.to_numeric(profile_1.get("reach_cm"), errors="coerce")
            - pd.to_numeric(profile_2.get("reach_cm"), errors="coerce")
        )
        row["height_diff"] = (
            pd.to_numeric(profile_1.get("height_cm"), errors="coerce")
            - pd.to_numeric(profile_2.get("height_cm"), errors="coerce")
        )
        row["age_diff"] = _age(profile_1, fight.event_date) - _age(profile_2, fight.event_date)
        stance_1 = str(profile_1.get("stance") or "")
        stance_2 = str(profile_2.get("stance") or "")
        row["stance_matchup"] = float(bool(stance_1 and stance_2 and stance_1 != stance_2))
        row["elo_1"] = left["elo"]
        row["elo_2"] = right["elo"]
        rows.append(row)
    return pd.DataFrame(rows)


def _model_version_note(base_dir: Path, error: Exception) -> str:
    """Explain an unpickling failure in terms someone can act on."""
    import json

    import sklearn

    trained_with = "inconnue"
    metadata = base_dir.parent / "models" / "ufc" / "metadata.json"
    if metadata.exists():
        try:
            trained_with = json.loads(metadata.read_text(encoding="utf-8")).get(
                "sklearn_version", "inconnue"
            )
        except (OSError, ValueError):
            pass
    if trained_with not in ("inconnue", sklearn.__version__):
        return (
            f"Probabilités indisponibles: modèle entraîné avec scikit-learn "
            f"{trained_with}, exécuté sous {sklearn.__version__}. Épinglez "
            f"`scikit-learn=={trained_with}` ou réentraînez depuis la page Mise à jour."
        )
    return f"Probabilités indisponibles ({error}); réentraînez le modèle UFC."


def collect_upcoming(base_dir: Path, limit: int = 4) -> dict[str, Any]:
    """Fetch the next cards and return them with fighter recognition flags."""
    # Resolve first: Path(".").parent is Path("."), which would silently look
    # for the model in the wrong place and report it missing.
    base_dir = Path(base_dir).resolve()
    processed = base_dir / "data" / "rigorous" / "processed"
    features_path = processed / "features.parquet"
    table_path = base_dir.parent / "models" / "ufc" / "fighter_states.parquet"
    # Either source of fighter state is enough. A deployment ships the frozen
    # table without the multi-hundred-megabyte fight history, and requiring both
    # would make the page fail there for no reason.
    if not table_path.exists() and not features_path.exists():
        return {
            "available": False,
            "reason": (
                "Ni models/ufc/fighter_states.parquet ni features.parquet: "
                "lancer l'entraînement UFC depuis la page Mise à jour."
            ),
        }

    client = UFCStatsClient()
    events = fetch_upcoming_events(client, limit=limit)
    fights: list[UpcomingFight] = []
    for name, date, url in events:
        fights.extend(fetch_card(client, name, date, url))
    if not fights:
        return {"available": False, "reason": "Aucun combat programmé publié par UFCStats"}

    from .model_pipeline import STATS_FEATURES, build_features

    # Replaying every fight to score a dozen bouts is the slow path; use the
    # portable table the trainer froze whenever it is there.
    if table_path.exists():
        descriptors = features_from_state_table(fights, pd.read_parquet(table_path))
    else:
        _, states, profiles = build_features(base_dir, return_states=True)
        descriptors = features_for_upcoming(fights, states, profiles)

    rows = []
    probabilities = None
    model_note = ""
    model_path = base_dir.parent / "models" / "ufc" / "ufc_descriptor_model.joblib"
    if model_path.exists():
        import joblib

        matrix = descriptors[STATS_FEATURES].to_numpy(dtype=float)
        usable = descriptors["both_known"].to_numpy()
        try:
            model = joblib.load(model_path)
            probabilities = np.full(len(descriptors), np.nan)
            if usable.any():
                probabilities[usable] = model.predict_proba(matrix[usable])[:, 1]
        except Exception as error:  # noqa: BLE001 - version mismatch is the usual cause
            # A serialised estimator only reloads under the version that fit it.
            # The card itself is still worth showing, so the failure is reported
            # rather than allowed to take the whole page down.
            probabilities = None
            model_note = _model_version_note(base_dir, error)

    for index, fight in enumerate(fights):
        rows.append({
            "événement": fight.event_name,
            "date": fight.event_date.date(),
            "combattant_1": fight.fighter_1,
            "combattant_2": fight.fighter_2,
            "catégorie": fight.weight_class,
            "p_combattant_1": probabilities[index] if probabilities is not None else np.nan,
            "expérience_1": descriptors.iloc[index].get("experience_1", np.nan),
            "expérience_2": descriptors.iloc[index].get("experience_2", np.nan),
            "elo_1": descriptors.iloc[index].get("elo_1", np.nan),
            "elo_2": descriptors.iloc[index].get("elo_2", np.nan),
            "les_deux_connus": bool(descriptors.iloc[index]["both_known"]),
        })
    frame = pd.DataFrame(rows)
    # Ordering happens once prices are attached, in the app layer: without a
    # price there is no expected value to rank by, only an opinion.
    frame["confiance"] = (frame["p_combattant_1"] - 0.5).abs()
    frame = frame.sort_values(
        ["les_deux_connus", "confiance"], ascending=[False, False]
    ).reset_index(drop=True)

    return {
        "available": True,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": UPCOMING_EVENTS_URL,
        "events": [{"name": name, "date": str(date.date())} for name, date, _ in events],
        "fights": frame,
        "model_available": model_path.exists() and probabilities is not None,
        "model_note": model_note,
        "note": (
            "Cartes programmées lues sans clé d'API. Les cotes, elles, exigent une clé "
            "The Odds API: sans elle aucun écart au marché ne peut être calculé."
        ),
    }
