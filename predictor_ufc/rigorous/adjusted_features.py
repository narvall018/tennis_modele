"""Descripteurs UFC ajustés à l'adversaire, à l'usure et à l'âge.

Les variables de `features_v2.parquet` sont des moyennes brutes sur cinq
combats: `f1_sig_lnd_L5`, `f1_td_acc_L5`, `f1_ctrl_secs_L5`… Elles ne disent pas
contre *qui* ces coups ont été portés. Un combattant qui place 80 frappes
significatives contre une défense poreuse et un autre qui en place 50 contre le
meilleur bloqueur de la division sortent avec la même ligne, alors qu'ils n'ont
pas montré la même chose.

Trois familles s'ajoutent ici, toutes absentes du jeu existant:

**Ajustement à l'adversaire.** Chaque statistique est comparée à ce que
l'adversaire concède habituellement. Porter 80 frappes à quelqu'un qui en
encaisse 90 en moyenne est une sous-performance; en porter 50 à quelqu'un qui en
encaisse 30 est le contraire.

**Usure.** Le cumul de carrière des frappes encaissées et des knockdowns subis.
Un combattant de 35 ans ayant absorbé 1 500 frappes significatives n'est pas le
même athlète qu'un homonyme du même âge en ayant absorbé 300. Aucune variable
existante ne distingue les deux.

**Âge et activité.** L'âge au combat, son carré, l'écart au pic usuel, le nombre
de combats sur douze mois, et le changement de catégorie de poids.

Comme pour le tennis, l'état d'un combattant est lu avant le combat décrit et
n'est mis à jour qu'une fois tous les combats de la même soirée décrits.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Constantes déclarées, jamais ajustées sur un rendement.
ADJUSTED_HALFLIFE = 4.0          # demi-vie, en combats, des moyennes ajustées
PEAK_AGE = 29.0                  # pic usuel en MMA, valeur de la littérature
LONG_LAYOFF_DAYS = 540
ACTIVITY_WINDOW_DAYS = 365
_ALPHA = 1.0 - 0.5 ** (1.0 / ADJUSTED_HALFLIFE)

ADJUSTED_FEATURES: List[str] = [
    "adj_sig_lnd_diff",
    "adj_sig_absorbed_diff",
    "adj_td_lnd_diff",
    "adj_ctrl_diff",
    "adj_kd_diff",
    "opponent_quality_diff",
    "wear_sig_absorbed_diff",
    "wear_kd_suffered_diff",
    "wear_fights_diff",
    "age_diff_years",
    "age_gap_to_peak_diff",
    "fights_365d_diff",
    "long_layoff_diff",
    "weight_class_change_diff",
]


class _Ewma:
    __slots__ = ("_sum", "_weight")

    def __init__(self) -> None:
        self._sum = 0.0
        self._weight = 0.0

    def push(self, value: float) -> None:
        self._sum = _ALPHA * value + (1.0 - _ALPHA) * self._sum
        self._weight = _ALPHA + (1.0 - _ALPHA) * self._weight

    def value(self, default: float = 0.0) -> float:
        return default if self._weight <= 0.0 else self._sum / self._weight


@dataclass
class FighterState:
    """Ce qu'un combattant a montré avant le combat courant."""

    adj_sig_lnd: _Ewma = field(default_factory=_Ewma)
    adj_sig_absorbed: _Ewma = field(default_factory=_Ewma)
    adj_td_lnd: _Ewma = field(default_factory=_Ewma)
    adj_ctrl: _Ewma = field(default_factory=_Ewma)
    adj_kd: _Ewma = field(default_factory=_Ewma)
    opponent_elo: _Ewma = field(default_factory=_Ewma)

    raw_sig_lnd_total: float = 0.0
    raw_sig_absorbed_total: float = 0.0
    raw_td_lnd_total: float = 0.0
    raw_ctrl_total: float = 0.0
    raw_kd_total: float = 0.0
    kd_suffered_total: float = 0.0
    fights: float = 0.0

    fight_dates: deque = field(default_factory=deque)
    last_weight_class: Optional[str] = None
    weight_class_changes: float = 0.0

    # -- moyennes brutes servant de référence à l'adversaire -------------
    def mean_sig_lnd(self) -> Optional[float]:
        return None if self.fights == 0 else self.raw_sig_lnd_total / self.fights

    def mean_sig_absorbed(self) -> Optional[float]:
        return None if self.fights == 0 else self.raw_sig_absorbed_total / self.fights

    def mean_td_lnd(self) -> Optional[float]:
        return None if self.fights == 0 else self.raw_td_lnd_total / self.fights

    def mean_ctrl(self) -> Optional[float]:
        return None if self.fights == 0 else self.raw_ctrl_total / self.fights

    def mean_kd(self) -> Optional[float]:
        return None if self.fights == 0 else self.raw_kd_total / self.fights

    def fights_in_window(self, ref, days: int) -> float:
        cutoff = ref - timedelta(days=days)
        return float(sum(1 for day in self.fight_dates if day >= cutoff))

    def days_since_last(self, ref, cap: float = 1460.0) -> float:
        if not self.fight_dates:
            return cap
        return min(float((ref - self.fight_dates[-1]).days), cap)


def _adjusted(actual: float, opponent_reference: Optional[float]) -> Optional[float]:
    """Écart entre ce qui a été réalisé et ce que l'adversaire concède d'habitude.

    Sans référence — premier combat de l'adversaire — l'observation n'est pas
    ajustable et n'alimente pas la moyenne, plutôt que d'être comparée à zéro.
    """
    if opponent_reference is None:
        return None
    return float(actual) - float(opponent_reference)


_PER_FIGHTER_COLUMNS = [
    "adj_sig_lnd", "adj_sig_absorbed", "adj_td_lnd", "adj_ctrl", "adj_kd",
    "opponent_quality", "wear_sig_absorbed", "wear_kd_suffered", "wear_fights",
    "age_years", "age_gap_to_peak", "fights_365d", "long_layoff", "weight_class_change",
]


def build_adjusted_features(
    appearances: pd.DataFrame, bio: pd.DataFrame, progress=print
) -> pd.DataFrame:
    """Une ligne par (combat, combattant); rien n'est lu qui suive le combat.

    `appearances` doit porter les colonnes brutes d'UFCStats plus l'Elo
    pré-combat (`elo_global_pre`) déjà calculé par le pipeline existant.
    """
    required = {
        "fight_id", "fighter_id", "event_date", "weight_class",
        "kd", "sig_lnd", "td_lnd", "ctrl_secs", "elo_global_pre",
    }
    missing = sorted(required - set(appearances.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes: {missing}")

    frame = appearances.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.dropna(subset=["event_date", "fight_id", "fighter_id"])
    frame = frame.sort_values(["event_date", "fight_id"], kind="mergesort")

    dob = (
        bio.dropna(subset=["fighter_url", "dob"])
        .assign(dob=lambda d: pd.to_datetime(d["dob"], errors="coerce"))
        .set_index("fighter_url")["dob"]
        .to_dict()
        if {"fighter_url", "dob"} <= set(bio.columns)
        else {}
    )
    url_by_id = (
        frame.dropna(subset=["fighter_url"]).set_index("fighter_id")["fighter_url"].to_dict()
        if "fighter_url" in frame.columns
        else {}
    )

    states: Dict[str, FighterState] = defaultdict(FighterState)
    rows: List[dict] = []
    skipped_incomplete = 0

    for day, day_rows in frame.groupby("event_date", sort=True):
        day_date = day.date() if hasattr(day, "date") else day
        pending: List[tuple] = []

        for fight_id, pair in day_rows.groupby("fight_id", sort=True):
            if len(pair) != 2:
                skipped_incomplete += 1
                continue
            first, second = pair.iloc[0], pair.iloc[1]

            for me, opponent in ((first, second), (second, first)):
                state = states[me["fighter_id"]]
                birth = dob.get(url_by_id.get(me["fighter_id"]))
                age = (
                    float((day - birth).days) / 365.25
                    if isinstance(birth, pd.Timestamp) and pd.notna(birth)
                    else np.nan
                )
                last_class = state.last_weight_class
                rows.append(
                    {
                        "fight_id": fight_id,
                        "fighter_id": me["fighter_id"],
                        "adj_sig_lnd": state.adj_sig_lnd.value(),
                        "adj_sig_absorbed": state.adj_sig_absorbed.value(),
                        "adj_td_lnd": state.adj_td_lnd.value(),
                        "adj_ctrl": state.adj_ctrl.value(),
                        "adj_kd": state.adj_kd.value(),
                        "opponent_quality": state.opponent_elo.value(1500.0),
                        "wear_sig_absorbed": state.raw_sig_absorbed_total,
                        "wear_kd_suffered": state.kd_suffered_total,
                        "wear_fights": state.fights,
                        "age_years": age,
                        "age_gap_to_peak": abs(age - PEAK_AGE) if np.isfinite(age) else np.nan,
                        "fights_365d": state.fights_in_window(day_date, ACTIVITY_WINDOW_DAYS),
                        "long_layoff": float(
                            state.days_since_last(day_date) > LONG_LAYOFF_DAYS and state.fights > 0
                        ),
                        "weight_class_change": float(
                            last_class is not None and last_class != me["weight_class"]
                        ),
                    }
                )
            pending.append((first, second, day_date))

        for first, second, day_date in pending:
            _apply_fight(states, first, second, day_date)
            _apply_fight(states, second, first, day_date)

    out = pd.DataFrame(rows)
    progress(
        f"Descripteurs UFC ajustés: {out['fight_id'].nunique():,} combats, "
        f"{out['fighter_id'].nunique():,} combattants, {skipped_incomplete} combats incomplets écartés"
    )
    return out


def _apply_fight(states: Dict[str, FighterState], me, opponent, day_date) -> None:
    """Met à jour l'état d'un combattant après son combat.

    L'ajustement se fait contre l'état de l'adversaire *avant* le combat, donc
    contre ce qu'il concédait jusque-là et non contre ce qu'il vient de concéder.
    """
    state = states[me["fighter_id"]]
    other = states[opponent["fighter_id"]]

    sig_lnd = float(me.get("sig_lnd", 0.0) or 0.0)
    td_lnd = float(me.get("td_lnd", 0.0) or 0.0)
    ctrl = float(me.get("ctrl_secs", 0.0) or 0.0)
    kd = float(me.get("kd", 0.0) or 0.0)
    absorbed = float(opponent.get("sig_lnd", 0.0) or 0.0)
    kd_suffered = float(opponent.get("kd", 0.0) or 0.0)

    for accumulator, value, reference in (
        (state.adj_sig_lnd, sig_lnd, other.mean_sig_absorbed()),
        (state.adj_sig_absorbed, absorbed, other.mean_sig_lnd()),
        (state.adj_td_lnd, td_lnd, other.mean_td_lnd()),
        (state.adj_ctrl, ctrl, other.mean_ctrl()),
        (state.adj_kd, kd, other.mean_kd()),
    ):
        adjusted = _adjusted(value, reference)
        if adjusted is not None:
            accumulator.push(adjusted)

    opponent_elo = opponent.get("elo_global_pre")
    if opponent_elo is not None and np.isfinite(opponent_elo):
        state.opponent_elo.push(float(opponent_elo))

    state.raw_sig_lnd_total += sig_lnd
    state.raw_sig_absorbed_total += absorbed
    state.raw_td_lnd_total += td_lnd
    state.raw_ctrl_total += ctrl
    state.raw_kd_total += kd
    state.kd_suffered_total += kd_suffered
    state.fights += 1.0
    state.fight_dates.append(day_date)
    state.last_weight_class = me["weight_class"]


def orient_to_pairs(
    per_fighter: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Passe du format « une ligne par combattant » aux écarts signés 1 moins 2.

    `pairs` doit porter `fight_id`, `fighter_1_id` et `fighter_2_id`, c'est-à-dire
    exactement l'orientation déterministe déjà figée par le pipeline UFC.
    """
    indexed = per_fighter.set_index(["fight_id", "fighter_id"])
    left = indexed.reindex(
        pd.MultiIndex.from_arrays([pairs["fight_id"], pairs["fighter_1_id"]])
    ).reset_index(drop=True)
    right = indexed.reindex(
        pd.MultiIndex.from_arrays([pairs["fight_id"], pairs["fighter_2_id"]])
    ).reset_index(drop=True)

    out = pd.DataFrame({"fight_id": pairs["fight_id"].to_numpy()})
    out["adj_sig_lnd_diff"] = left["adj_sig_lnd"] - right["adj_sig_lnd"]
    out["adj_sig_absorbed_diff"] = left["adj_sig_absorbed"] - right["adj_sig_absorbed"]
    out["adj_td_lnd_diff"] = left["adj_td_lnd"] - right["adj_td_lnd"]
    out["adj_ctrl_diff"] = left["adj_ctrl"] - right["adj_ctrl"]
    out["adj_kd_diff"] = left["adj_kd"] - right["adj_kd"]
    out["opponent_quality_diff"] = left["opponent_quality"] - right["opponent_quality"]
    out["wear_sig_absorbed_diff"] = left["wear_sig_absorbed"] - right["wear_sig_absorbed"]
    out["wear_kd_suffered_diff"] = left["wear_kd_suffered"] - right["wear_kd_suffered"]
    out["wear_fights_diff"] = left["wear_fights"] - right["wear_fights"]
    out["age_diff_years"] = left["age_years"] - right["age_years"]
    out["age_gap_to_peak_diff"] = left["age_gap_to_peak"] - right["age_gap_to_peak"]
    out["fights_365d_diff"] = left["fights_365d"] - right["fights_365d"]
    out["long_layoff_diff"] = left["long_layoff"] - right["long_layoff"]
    out["weight_class_change_diff"] = left["weight_class_change"] - right["weight_class_change"]
    return out
