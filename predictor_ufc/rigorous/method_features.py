"""Pre-fight features for *how* a fight ends, not for who wins.

The moneyline studies in this package all model a difference between two
fighters, because that is what predicts a winner. Method of victory is a
different question with a different shape: whether a fight reaches the judges
depends on what the two men do *together*. Two heavy finishers rarely go the
distance; two cautious wrestlers usually do. So the features here are sums,
minima and levels rather than differences, and they are deliberately symmetric —
swapping the corners must not change the predicted probability of a decision.

Every value is built from fights strictly before the current event date, in one
chronological pass: a fighter's record is read before the current fight updates
it. Nothing here reads the method, duration, or result of the fight being
described.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


KO_METHODS = {"KO/TKO", "TKO - Doctor's Stoppage"}
SUB_METHODS = {"Submission", "SUB"}
DECISION_METHODS = {
    "Decision - Unanimous",
    "Decision - Split",
    "Decision - Majority",
    "U-DEC",
    "S-DEC",
    "M-DEC",
}

# Divisions where finishing power is structurally higher; kept as a single
# ordered feature rather than one column per class to limit the search space.
WEIGHT_ORDER = [
    "Women's Strawweight", "Women's Flyweight", "Flyweight", "Women's Bantamweight",
    "Bantamweight", "Women's Featherweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]


def method_category(method: Any) -> str | None:
    text = str(method or "")
    if text in KO_METHODS:
        return "ko"
    if text in SUB_METHODS:
        return "sub"
    if text in DECISION_METHODS:
        return "dec"
    return None


class _FighterState:
    """Running, pre-fight record of one fighter."""

    __slots__ = (
        "fights", "decisions", "ko_for", "ko_against", "sub_for", "sub_against",
        "seconds", "sig_landed", "sig_absorbed", "takedowns", "knockdowns",
        "sub_attempts", "control",
    )

    def __init__(self) -> None:
        self.fights = 0
        self.decisions = 0
        self.ko_for = 0
        self.ko_against = 0
        self.sub_for = 0
        self.sub_against = 0
        self.seconds = 0.0
        self.sig_landed = 0.0
        self.sig_absorbed = 0.0
        self.takedowns = 0.0
        self.knockdowns = 0.0
        self.sub_attempts = 0.0
        self.control = 0.0

    def snapshot(self) -> dict[str, float]:
        fights = max(self.fights, 1)
        minutes = max(self.seconds / 60.0, 1.0)
        return {
            "experience": float(self.fights),
            "decision_rate": self.decisions / fights,
            "finish_for_rate": (self.ko_for + self.sub_for) / fights,
            "finished_against_rate": (self.ko_against + self.sub_against) / fights,
            "ko_for_rate": self.ko_for / fights,
            "sub_for_rate": self.sub_for / fights,
            "mean_fight_minutes": self.seconds / 60.0 / fights,
            "sig_landed_pm": self.sig_landed / minutes,
            "sig_absorbed_pm": self.sig_absorbed / minutes,
            "takedowns_p15": 15.0 * self.takedowns / minutes,
            "knockdowns_p15": 15.0 * self.knockdowns / minutes,
            "sub_attempts_p15": 15.0 * self.sub_attempts / minutes,
            "control_share": self.control / max(self.seconds, 1.0),
        }

    def update(self, *, won: bool, category: str | None, seconds: float,
               landed: float, absorbed: float, takedowns: float,
               knockdowns: float, sub_attempts: float, control: float) -> None:
        self.fights += 1
        if category == "dec":
            self.decisions += 1
        elif category == "ko":
            self.ko_for += int(won)
            self.ko_against += int(not won)
        elif category == "sub":
            self.sub_for += int(won)
            self.sub_against += int(not won)
        self.seconds += seconds
        self.sig_landed += landed
        self.sig_absorbed += absorbed
        self.takedowns += takedowns
        self.knockdowns += knockdowns
        self.sub_attempts += sub_attempts
        self.control += control


SYMMETRIC_FEATURES = [
    "both_decision_rate",
    "min_decision_rate",
    "both_finish_for_rate",
    "max_finish_for_rate",
    "both_finished_against_rate",
    "max_finished_against_rate",
    "both_ko_for_rate",
    "both_sub_for_rate",
    "both_mean_fight_minutes",
    "min_mean_fight_minutes",
    "both_sig_landed_pm",
    "both_sig_absorbed_pm",
    "both_takedowns_p15",
    "both_knockdowns_p15",
    "max_knockdowns_p15",
    "both_sub_attempts_p15",
    "both_control_share",
    "min_experience",
    "both_experience",
    "both_age",
    "max_age",
    "scheduled_five_rounds",
    "weight_index",
]


def _pair(left: dict[str, float], right: dict[str, float], key: str) -> tuple[float, float, float]:
    """Mean, min and max of one statistic across the two fighters."""
    a, b = left[key], right[key]
    return (a + b) / 2.0, min(a, b), max(a, b)


def build_method_features(fights: pd.DataFrame) -> pd.DataFrame:
    """One chronological pass; a fighter's state is read before it is updated."""
    ordered = fights.sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)
    states: dict[str, _FighterState] = {}
    weight_index = {name: index for index, name in enumerate(WEIGHT_ORDER)}
    rows: list[dict[str, Any]] = []
    # Fights on one card must all read the same pre-card state.
    for _, card in ordered.groupby("event_id", sort=False):
        pending = []
        for fight in card.itertuples(index=False):
            left = states.setdefault(str(fight.fighter_1_id), _FighterState()).snapshot()
            right = states.setdefault(str(fight.fighter_2_id), _FighterState()).snapshot()
            title_bout = "title" in str(fight.weight_class).lower()
            division = str(fight.weight_class or "")
            for suffix in (" Title", "UFC "):
                division = division.replace(suffix, "").strip()

            row: dict[str, Any] = {
                "fight_id": fight.fight_id,
                "event_date": fight.event_date,
                "scheduled_five_rounds": float(title_bout),
                "weight_index": float(weight_index.get(division, len(WEIGHT_ORDER) // 2)),
            }
            for key, prefix in (
                ("decision_rate", "decision_rate"),
                ("finish_for_rate", "finish_for_rate"),
                ("finished_against_rate", "finished_against_rate"),
                ("ko_for_rate", "ko_for_rate"),
                ("sub_for_rate", "sub_for_rate"),
                ("mean_fight_minutes", "mean_fight_minutes"),
                ("sig_landed_pm", "sig_landed_pm"),
                ("sig_absorbed_pm", "sig_absorbed_pm"),
                ("takedowns_p15", "takedowns_p15"),
                ("knockdowns_p15", "knockdowns_p15"),
                ("sub_attempts_p15", "sub_attempts_p15"),
                ("control_share", "control_share"),
                ("experience", "experience"),
            ):
                mean, low, high = _pair(left, right, key)
                row[f"both_{prefix}"] = mean
                row[f"min_{prefix}"] = low
                row[f"max_{prefix}"] = high
            ages = [
                float(getattr(fight, "age_1", np.nan) or np.nan),
                float(getattr(fight, "age_2", np.nan) or np.nan),
            ]
            row["both_age"] = float(np.nanmean(ages)) if np.isfinite(ages).any() else np.nan
            row["max_age"] = float(np.nanmax(ages)) if np.isfinite(ages).any() else np.nan
            rows.append(row)

            category = method_category(fight.method)
            pending.append((fight, category))

        for fight, category in pending:
            seconds = float(fight.duration_secs or 0.0)
            won_1 = bool(fight.y == 1) if pd.notna(fight.y) else False
            states[str(fight.fighter_1_id)].update(
                won=won_1, category=category, seconds=seconds,
                landed=float(fight.p1_sig_lnd or 0.0), absorbed=float(fight.p2_sig_lnd or 0.0),
                takedowns=float(fight.p1_td_lnd or 0.0), knockdowns=float(fight.p1_kd or 0.0),
                sub_attempts=float(fight.p1_sub_att or 0.0), control=float(fight.p1_ctrl_secs or 0.0),
            )
            states[str(fight.fighter_2_id)].update(
                won=not won_1 and pd.notna(fight.y), category=category, seconds=seconds,
                landed=float(fight.p2_sig_lnd or 0.0), absorbed=float(fight.p1_sig_lnd or 0.0),
                takedowns=float(fight.p2_td_lnd or 0.0), knockdowns=float(fight.p2_kd or 0.0),
                sub_attempts=float(fight.p2_sub_att or 0.0), control=float(fight.p2_ctrl_secs or 0.0),
            )

    frame = pd.DataFrame(rows)
    labels = ordered[["fight_id", "method", "y"]].copy()
    labels["method_category"] = labels["method"].map(method_category)
    labels["goes_to_decision"] = (labels["method_category"] == "dec").astype("float")
    labels.loc[labels["method_category"].isna(), "goes_to_decision"] = np.nan
    return frame.merge(labels[["fight_id", "method_category", "goes_to_decision", "y"]], on="fight_id")
