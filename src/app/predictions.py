"""Probabilities for upcoming matches, and how they compare to the price.

The app needs, per sport: a fixture list, a calibrated probability, the quoted
price, and the difference between them. That difference is labelled an *écart*
and never an edge, because eight studies in this repository say it is not one.

Football is fully wired: Football-Data publishes fixtures with prices, and the
model persisted by `scripts/train_football_model.py` scores them from stored team
state. Tennis and UFC have models but no free fixture-with-price feed, so they
report what they can and say plainly what is missing rather than showing an empty
table that looks like a failure.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.app.odds_api import (
    MMA_SPORT_KEY,
    TENNIS_SPORT_PREFIX,
    active_sports,
    consensus_prices,
    devig,
    fetch_h2h_odds,
    resolve_api_key,
)
from src.data.football_fixtures import FIXTURE_PRICE_GROUPS, load_fixtures
from scripts.export_tennis_ratings import normalise_player
from src.features.football_features import FEATURE_COLUMNS, features_for_fixtures


RESULT_ORDER = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Domicile", "D": "Nul", "A": "Extérieur"}


@dataclass
class SportPredictions:
    sport: str
    available: bool
    rows: pd.DataFrame
    meta: dict[str, Any]
    unavailable_reason: str = ""


# A recommendation is only as good as the state behind it. Below this many
# recorded matches a team's rolling descriptors are mostly noise, so its
# expected value is discounted rather than trusted at face value.
RELIABLE_HISTORY = 10

# Every study in this repository measured these models as *worse* than the
# price. A large disagreement is therefore evidence against the model, not a
# find: ranking by raw expected value would put an Elo that prices a 3% shot at
# 38% on top, with a nominal +1000% return. Past this gap the row is not ranked.
MAX_PLAUSIBLE_DISAGREEMENT = 0.15


def recommendation_score(
    expected_value: float,
    left_history: float,
    right_history: float,
    both_known: bool,
    disagreement: float | None = None,
) -> float:
    """Rank fixtures by what actually makes a recommendation good.

    Used by all three sports, with ``*_history`` being matches played for a team
    or fights recorded for a fighter. Sorting by the raw model-market gap would
    put the worst rows on top: the gap is widest exactly where the model is least
    reliable, which is a competitor it has barely seen. So the ranking is
    expected value, discounted by how much history stands behind it, and zero
    whenever the expectation is negative or a competitor is unknown.
    """
    if not both_known or not np.isfinite(expected_value) or expected_value <= 0:
        return 0.0
    if disagreement is not None and (
        not np.isfinite(disagreement) or abs(disagreement) > MAX_PLAUSIBLE_DISAGREEMENT
    ):
        return 0.0
    played = min(
        left_history if np.isfinite(left_history) else 0.0,
        right_history if np.isfinite(right_history) else 0.0,
    )
    confidence = min(played / RELIABLE_HISTORY, 1.0)
    return float(expected_value * confidence)


def _model_error(root: Path, model_dir: Path, error: Exception) -> str:
    """Turn an unpickling failure into something the user can act on.

    A serialised scikit-learn estimator only reloads under the version that fit
    it. The raw AttributeError names a private attribute and tells nobody
    anything, so the versions are compared and the fix is stated.
    """
    import sklearn

    trained_with = "inconnue"
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            trained_with = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "sklearn_version", "inconnue"
            )
        except (OSError, ValueError):
            pass
    if trained_with not in ("inconnue", sklearn.__version__):
        return (
            f"Modèle inutilisable: entraîné avec scikit-learn {trained_with}, "
            f"exécuté sous {sklearn.__version__}. Un modèle sérialisé n'est pas "
            "portable entre versions. Épinglez `scikit-learn=="
            f"{trained_with}` dans requirements.txt, ou réentraînez depuis la page "
            "Mise à jour."
        )
    return (
        f"Modèle illisible sous scikit-learn {sklearn.__version__}: {error}. "
        "Réentraînez-le depuis la page Mise à jour."
    )


def _devig_row(prices: np.ndarray) -> np.ndarray:
    inverse = 1.0 / prices
    total = inverse.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(prices), np.nan)
    return inverse / total


def football_predictions(root: Path, price_source: str = "market_average") -> SportPredictions:
    model_dir = root / "models" / "football"
    model_path = model_dir / "football_model.joblib"
    states_path = model_dir / "team_states.parquet"
    if not model_path.exists() or not states_path.exists():
        return SportPredictions(
            "Football", False, pd.DataFrame(), {},
            "Modèle absent. Lancer: python3 scripts/train_football_model.py",
        )

    try:
        fixtures, meta = load_fixtures()
    except Exception as error:  # noqa: BLE001 - surfaced to the user as a message
        return SportPredictions(
            "Football", False, pd.DataFrame(), {},
            f"Rencontres indisponibles: {error}",
        )
    if fixtures.empty:
        return SportPredictions(
            "Football", False, pd.DataFrame(), meta,
            "Aucune rencontre à venir publiée pour les divisions suivies.",
        )

    states = pd.read_parquet(states_path)
    features = features_for_fixtures(fixtures, states)
    matrix = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    try:
        probabilities = joblib.load(model_path).predict_proba(matrix)
    except Exception as error:  # noqa: BLE001 - version mismatch is the usual cause
        return SportPredictions(
            "Football", False, pd.DataFrame(), meta, _model_error(root, model_dir, error)
        )

    columns = FIXTURE_PRICE_GROUPS[price_source]
    prices = fixtures[list(columns)].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures.itertuples(index=False)):
        row_prices = prices[index]
        quoted = np.isfinite(row_prices).all() and (row_prices > 1.0).all()
        market = _devig_row(row_prices) if quoted else np.full(3, np.nan)
        model_probability = probabilities[index]
        # The biggest disagreement, whichever way it points.
        gaps = model_probability - market if quoted else np.full(3, np.nan)
        best = int(np.nanargmax(gaps)) if quoted and np.isfinite(gaps).any() else None
        rows.append({
            "match_id": fixture.match_id,
            "date": fixture.match_date.date(),
            "heure": fixture.kickoff,
            "division": fixture.league,
            "pays": fixture.country,
            "domicile": fixture.home_team,
            "extérieur": fixture.away_team,
            "p_domicile": model_probability[0],
            "p_nul": model_probability[1],
            "p_extérieur": model_probability[2],
            "cote_domicile": row_prices[0] if quoted else np.nan,
            "cote_nul": row_prices[1] if quoted else np.nan,
            "cote_extérieur": row_prices[2] if quoted else np.nan,
            "marché_domicile": market[0],
            "marché_nul": market[1],
            "marché_extérieur": market[2],
            "issue_écart_max": OUTCOME_LABELS[RESULT_ORDER[best]] if best is not None else "",
            "écart": gaps[best] if best is not None else np.nan,
            "cote_écart_max": row_prices[best] if best is not None else np.nan,
            "pari": OUTCOME_LABELS[RESULT_ORDER[best]] if best is not None else "",
            "cote_pari": row_prices[best] if best is not None else np.nan,
            "p_pari": model_probability[best] if best is not None else np.nan,
            "espérance": (
                model_probability[best] * row_prices[best] - 1.0 if best is not None else np.nan
            ),
            "équipes_connues": bool(features.iloc[index]["home_known"] and
                                    features.iloc[index]["away_known"]),
            "matchs_domicile": features.iloc[index]["home_matches_played"],
            "matchs_extérieur": features.iloc[index]["away_matches_played"],
        })
    frame = pd.DataFrame(rows)
    frame["score"] = [
        recommendation_score(
            row["espérance"], row["matchs_domicile"], row["matchs_extérieur"],
            row["équipes_connues"], disagreement=row["écart"],
        )
        for _, row in frame.iterrows()
    ]
    # Best recommendation first; ties fall back to kickoff order.
    frame = frame.sort_values(
        ["score", "date", "division"], ascending=[False, True, True]
    ).reset_index(drop=True)
    meta["price_source"] = price_source
    meta["ranking"] = (
        "Classé par espérance au prix affiché, escomptée par l'historique disponible "
        f"des deux équipes ({RELIABLE_HISTORY} matchs pour une confiance pleine). "
        "Un score nul signifie espérance négative ou équipe inconnue."
    )
    meta["model_note"] = (
        "Probabilités d'un modèle calibré qui n'a pas vu la cote. L'écart affiché "
        "n'est pas un avantage: mesuré, il vaut −0,00080 de log-loss contre le prix."
    )
    return SportPredictions("Football", True, frame, meta)


def _load_tennis_ratings(root: Path) -> dict[str, pd.DataFrame]:
    """Ratings indexed by normalised name, with collisions resolved explicitly.

    Two distinct players can normalise to the same key. Silently keeping one
    would attach a stranger's rating to a live match, so a key claimed by more
    than one name is dropped entirely; a key repeated by the same name keeps the
    most recently active row.
    """
    tables: dict[str, pd.DataFrame] = {}
    for tour in ("atp", "wta"):
        path = root / "models" / "tennis" / f"{tour}_player_ratings.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        distinct_names = table.groupby("player_key")["player"].nunique()
        ambiguous = set(distinct_names[distinct_names > 1].index)
        table = table[~table["player_key"].isin(ambiguous)]
        table = table.sort_values("last_date").drop_duplicates("player_key", keep="last")
        tables[tour] = table.set_index("player_key")
    return tables


def _rating_history(
    ratings: dict[str, pd.DataFrame], tour: str, left: str, right: str
) -> tuple[float, float]:
    """Matches recorded for each player, used to discount a thin opinion."""
    table = ratings.get(tour)
    if table is None:
        return 0.0, 0.0
    keys = (normalise_player(left), normalise_player(right))
    return tuple(
        float(table.loc[key, "matches"]) if key in table.index else 0.0 for key in keys
    )


def _elo_probability(
    ratings: dict[str, pd.DataFrame], tour: str, favourite: str, outsider: str,
    surface_weight: float = 0.5,
) -> tuple[float, bool]:
    """Elo win probability for the favourite, or NaN if either player is unrated."""
    table = ratings.get(tour)
    if table is None:
        return np.nan, False
    left_key, right_key = normalise_player(favourite), normalise_player(outsider)
    if left_key not in table.index or right_key not in table.index:
        return np.nan, False
    left, right = table.loc[left_key], table.loc[right_key]
    # The US Open and most of the calendar are hard courts; blending the global
    # and hard-court ratings is the engine's own convention, not a fitted choice.
    gap = (1.0 - surface_weight) * (float(left["elo"]) - float(right["elo"]))
    gap += surface_weight * (float(left["elo_hard"]) - float(right["elo_hard"]))
    return float(1.0 / (1.0 + 10.0 ** (-gap / 400.0))), True


def tennis_predictions(root: Path) -> SportPredictions:
    """Live ATP/WTA matches with prices, from The Odds API.

    Tennis has no free fixture feed, so unlike football and UFC this tab depends
    entirely on the odds API — the price *is* the fixture list here. Each active
    tennis competition costs one request, so the caller must cache.
    """
    key, source = resolve_api_key(root)
    if not key:
        return SportPredictions(
            "Tennis", False, pd.DataFrame(), {},
            "Aucune clé The Odds API trouvée. Le palier gratuit (500 requêtes/mois) "
            "suffit: définir ODDS_API_KEY, ou la placer dans les secrets Streamlit.",
        )

    catalogue = active_sports(root)
    if not catalogue.ok:
        return SportPredictions(
            "Tennis", False, pd.DataFrame(), {}, f"Catalogue indisponible: {catalogue.error}"
        )
    tennis_keys = [
        sport["key"] for sport in catalogue.events
        if str(sport.get("key", "")).startswith(TENNIS_SPORT_PREFIX) and sport.get("active")
    ]
    if not tennis_keys:
        return SportPredictions(
            "Tennis", False, pd.DataFrame(), {"key_source": source},
            "Aucune compétition de tennis active chez le fournisseur en ce moment. "
            "Les tournois n'apparaissent que pendant leur déroulement.",
        )

    ratings = _load_tennis_ratings(root)
    rows: list[dict[str, Any]] = []
    remaining = catalogue.remaining
    for sport_key in tennis_keys:
        response = fetch_h2h_odds(root, sport_key)
        remaining = response.remaining if response.remaining is not None else remaining
        if not response.ok:
            continue
        for event in response.events:
            prices = consensus_prices(event)
            if len(prices) != 2:
                continue
            probabilities = devig(prices)
            names = sorted(prices, key=lambda name: -probabilities.get(name, 0.0))
            favourite, outsider = names[0], names[1]
            tour = "wta" if "wta" in sport_key else "atp"
            model_probability, both_rated = _elo_probability(
                ratings, tour, favourite, outsider
            )
            market_probability = probabilities[favourite]
            history = _rating_history(ratings, tour, favourite, outsider)
            if pd.notna(model_probability):
                ev_favourite = model_probability * prices[favourite] - 1.0
                ev_outsider = (1.0 - model_probability) * prices[outsider] - 1.0
                if ev_favourite >= ev_outsider:
                    pick, expected, pick_odds = favourite, ev_favourite, prices[favourite]
                    pick_probability = model_probability
                else:
                    pick, expected, pick_odds = outsider, ev_outsider, prices[outsider]
                    pick_probability = 1.0 - model_probability
            else:
                pick, expected, pick_odds, pick_probability = "", np.nan, np.nan, np.nan
            rows.append({
                "début": str(event.get("commence_time", ""))[:16].replace("T", " "),
                "compétition": sport_key.replace(TENNIS_SPORT_PREFIX, "").replace("_", " "),
                "favori": favourite,
                "adversaire": outsider,
                "cote_favori": prices[favourite],
                "cote_adversaire": prices[outsider],
                "p_marché_favori": market_probability,
                "p_modèle_favori": model_probability,
                "écart": (
                    model_probability - market_probability
                    if pd.notna(model_probability) else np.nan
                ),
                "joueurs_notés": both_rated,
                "pari": pick,
                "cote_pari": pick_odds,
                "p_pari": pick_probability,
                "espérance": expected,
                "score": recommendation_score(
                    expected, history[0], history[1], both_rated,
                    disagreement=(
                        model_probability - market_probability
                        if pd.notna(model_probability) else None
                    ),
                ),
                "books": len(event.get("bookmakers") or []),
            })
    if not rows:
        return SportPredictions(
            "Tennis", False, pd.DataFrame(), {"key_source": source, "remaining": remaining},
            "Compétitions actives trouvées mais aucun match coté pour l'instant.",
        )

    frame = pd.DataFrame(rows).sort_values(
        ["score", "début"], ascending=[False, True]
    ).reset_index(drop=True)
    meta = {
        "key_source": source,
        "competitions": ", ".join(tennis_keys),
        "remaining_requests": remaining,
        "prices": "prix médian des bookmakers renvoyés, dévigé",
        "rated": int(frame["joueurs_notés"].sum()),
        "model": (
            "Probabilité Elo (global et surface dure, à parts égales), rapprochée par "
            "nom complet sans accent — le même format des deux côtés. Un joueur absent "
            "de la table reste vide plutôt que de recevoir une note moyenne."
        ),
    }
    return SportPredictions("Tennis", True, frame, meta)


def ufc_predictions(root: Path, events: int = 3) -> SportPredictions:
    """Scheduled cards with the descriptor model's opinion. No API key needed.

    Only the *price* requires a key; UFCStats publishes the fight list, and the
    package's client answers the site's proof-of-work challenge. Without a key
    there is no market to compare against, so no gap is shown — an opinion, and
    the page says so.
    """
    package = root / "predictor_ufc"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    try:
        from rigorous.upcoming import collect_upcoming
    except Exception as error:  # noqa: BLE001 - surfaced to the user
        return SportPredictions(
            "UFC", False, pd.DataFrame(), {},
            f"Module UFC indisponible: {error}",
        )

    try:
        result = collect_upcoming(package, limit=events)
    except Exception as error:  # noqa: BLE001 - network or parsing
        return SportPredictions(
            "UFC", False, pd.DataFrame(), {},
            f"UFCStats injoignable: {error}",
        )
    if not result.get("available"):
        return SportPredictions(
            "UFC", False, pd.DataFrame(), {}, result.get("reason", "indisponible")
        )

    frame = result["fights"]
    prices_note = "aucune clé The Odds API: pas de prix, donc aucun écart au marché"
    remaining = None
    key, key_source = resolve_api_key(root)
    if key:
        response = fetch_h2h_odds(root, MMA_SPORT_KEY)
        remaining = response.remaining
        if response.ok:
            frame = _attach_mma_prices(frame, response.events)
            prices_note = (
                f"prix médian des bookmakers, source de clé: {key_source}. "
                "Le flux MMA couvre plusieurs organisations; seuls les combats "
                "rapprochés par nom reçoivent un prix."
            )
        else:
            prices_note = f"cotes indisponibles: {response.error}"

    if "espérance" in frame.columns:
        # A fighter with a short record is exactly where the model is least
        # reliable, so his expected value is discounted the same way football's is.
        frame["score"] = [
            recommendation_score(
                row.espérance,
                getattr(row, "expérience_1", np.nan),
                getattr(row, "expérience_2", np.nan),
                bool(row.les_deux_connus),
                disagreement=getattr(row, "écart", None),
            )
            if pd.notna(getattr(row, "espérance", np.nan)) else 0.0
            for row in frame.itertuples(index=False)
        ]
        frame = frame.sort_values(
            ["score", "confiance"], ascending=[False, False]
        ).reset_index(drop=True)

    meta = {
        "source": result["source"],
        "fetched_at_utc": result["fetched_at_utc"],
        "events": ", ".join(event["name"] for event in result["events"]),
        "model_available": result.get("model_available", False),
        "model_note": result.get("model_note", ""),
        "prices": prices_note,
        "remaining_requests": remaining,
        "note": result["note"],
    }
    return SportPredictions("UFC", True, frame, meta)


def _attach_mma_prices(frame: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    """Match quoted MMA bouts to the scheduled card by fighter surname.

    The odds feed carries several promotions and spells names its own way, so the
    join is on the pair of surnames. A bout that does not match keeps no price
    rather than being attached to a plausible-looking neighbour.
    """
    def surnames(*names: str) -> frozenset[str]:
        return frozenset(
            re.sub(r"[^a-z]", "", part.lower())
            for name in names
            for part in str(name).split()[-1:]
        )

    quoted: dict[frozenset[str], dict[str, float]] = {}
    for event in events:
        prices = consensus_prices(event)
        if len(prices) == 2:
            quoted[surnames(*prices)] = prices

    columns = {"cote_1": [], "cote_2": [], "p_marché_1": [], "écart": []}
    for row in frame.itertuples(index=False):
        prices = quoted.get(surnames(row.combattant_1, row.combattant_2))
        if not prices:
            for key in columns:
                columns[key].append(np.nan)
            continue
        by_name = {surnames(name): (name, price) for name, price in prices.items()}
        first = by_name.get(surnames(row.combattant_1))
        second = by_name.get(surnames(row.combattant_2))
        if not first or not second:
            for key in columns:
                columns[key].append(np.nan)
            continue
        probabilities = devig(prices)
        market_1 = probabilities.get(first[0], np.nan)
        columns["cote_1"].append(first[1])
        columns["cote_2"].append(second[1])
        columns["p_marché_1"].append(market_1)
        model_1 = row.p_combattant_1
        columns["écart"].append(
            model_1 - market_1 if pd.notna(model_1) and pd.notna(market_1) else np.nan
        )
    for key, values in columns.items():
        frame[key] = values

    # Expected value on whichever fighter the model prefers at the quoted price.
    best_side, best_ev, best_odds, best_probability = [], [], [], []
    for row in frame.itertuples(index=False):
        model_1 = row.p_combattant_1
        if pd.isna(model_1) or pd.isna(row.cote_1) or pd.isna(row.cote_2):
            best_side.append("")
            best_ev.append(np.nan)
            best_odds.append(np.nan)
            best_probability.append(np.nan)
            continue
        ev_1 = model_1 * row.cote_1 - 1.0
        ev_2 = (1.0 - model_1) * row.cote_2 - 1.0
        if ev_1 >= ev_2:
            best_side.append(row.combattant_1)
            best_ev.append(ev_1)
            best_odds.append(row.cote_1)
            best_probability.append(model_1)
        else:
            best_side.append(row.combattant_2)
            best_ev.append(ev_2)
            best_odds.append(row.cote_2)
            best_probability.append(1.0 - model_1)
    frame["pari"] = best_side
    frame["espérance"] = best_ev
    frame["cote_pari"] = best_odds
    frame["p_pari"] = best_probability
    return frame


def all_predictions(root: Path) -> list[SportPredictions]:
    return [football_predictions(root), tennis_predictions(root), ufc_predictions(root)]


# Columns every sport must expose for the staking page to treat them alike.
CANDIDATE_COLUMNS = [
    "sport", "quand", "rencontre", "pari", "cote_pari", "p_pari",
    "espérance", "score",
]


def betting_candidates(block: SportPredictions) -> pd.DataFrame:
    """Normalise one sport's rows into the shape the staking page consumes.

    Each sport names its own columns after its own vocabulary — issue, combat,
    match — so a single staking rule needs one translation layer rather than
    three near-copies of the same sizing code.
    """
    empty = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if not block.available or block.rows.empty:
        return empty
    frame = block.rows
    if not {"pari", "cote_pari", "espérance", "score"}.issubset(frame.columns):
        return empty

    if block.sport == "Football":
        label = frame["domicile"] + " – " + frame["extérieur"] + "  (" + frame["division"] + ")"
        when = frame["date"].astype(str)
    elif block.sport == "UFC":
        label = frame["combattant_1"] + " – " + frame["combattant_2"]
        when = frame["date"].astype(str)
    else:
        label = frame["favori"] + " – " + frame["adversaire"]
        when = frame["début"].astype(str)

    normalised = pd.DataFrame({
        "sport": block.sport,
        "quand": when.to_numpy(),
        "rencontre": label.to_numpy(),
        "pari": frame["pari"].to_numpy(),
        "cote_pari": pd.to_numeric(frame["cote_pari"], errors="coerce").to_numpy(),
        "p_pari": pd.to_numeric(frame["p_pari"], errors="coerce").to_numpy(),
        "espérance": pd.to_numeric(frame["espérance"], errors="coerce").to_numpy(),
        "score": pd.to_numeric(frame["score"], errors="coerce").to_numpy(),
    })
    # A row the ranking already rejected is not a candidate for a stake.
    return normalised[normalised["score"] > 0].reset_index(drop=True)
