"""Stake sizing, with the property that matters most here: zero edge, zero stake.

Kelly is the standard answer to "how much", and its standard failure is that it
sizes on a *believed* edge. Everything measured in this repository says the
believed edge is indistinguishable from zero, so the sizing below is deliberately
built to collapse to nothing when that is the case rather than to produce a
confident-looking number.

Three guards, all applied together:

* **Fractional Kelly.** Full Kelly is optimal only if the probability is exactly
  right. Ours is not, so a divisor is mandatory, never optional.
* **A hard cap per bet and per day.** Kelly on a mis-estimated edge can ask for
  an absurd fraction; the cap is what stops one wrong probability from mattering.
* **An edge floor.** Below it the stake is zero, because an edge smaller than the
  measurement error is not an edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Below this, the measured evidence cannot distinguish the edge from zero, so no
# stake is proposed. 2% is roughly the execution friction that
# RAPPORT_RENTABILITE.md shows is enough to erase the best edge ever measured.
MINIMUM_EDGE = 0.02


@dataclass(frozen=True)
class StakePlan:
    name: str
    kelly_divisor: float = 8.0
    max_fraction_per_bet: float = 0.005
    max_fraction_per_day: float = 0.02
    minimum_edge: float = MINIMUM_EDGE
    flat_fraction: float | None = None


PLANS = {
    "prudent": StakePlan("prudent", kelly_divisor=16.0, max_fraction_per_bet=0.0025),
    "standard": StakePlan("standard", kelly_divisor=8.0, max_fraction_per_bet=0.005),
    "plat": StakePlan("plat", flat_fraction=0.0025, max_fraction_per_bet=0.0025),
}


def kelly_fraction(probability: float, odds: float) -> float:
    """Full-Kelly fraction for a simple win/lose bet, floored at zero."""
    if not np.isfinite(probability) or not np.isfinite(odds) or odds <= 1.0:
        return 0.0
    net = odds - 1.0
    fraction = (probability * net - (1.0 - probability)) / net
    return float(max(fraction, 0.0))


def stake_for_bet(
    probability: float,
    odds: float,
    bankroll: float,
    plan: StakePlan,
) -> dict[str, Any]:
    """One bet's recommended stake, with the reason when it is zero."""
    if not np.isfinite(probability) or not np.isfinite(odds) or odds <= 1.0 or bankroll <= 0:
        return {"stake": 0.0, "fraction": 0.0, "reason": "prix ou probabilité inutilisable"}

    edge = probability * odds - 1.0
    if edge < plan.minimum_edge:
        return {
            "stake": 0.0,
            "fraction": 0.0,
            "edge": edge,
            "reason": (
                f"écart de {edge:+.2%} sous le plancher de {plan.minimum_edge:.0%}: "
                "indistinguable de zéro"
            ),
        }

    if plan.flat_fraction is not None:
        fraction = plan.flat_fraction
    else:
        fraction = kelly_fraction(probability, odds) / plan.kelly_divisor
    capped = min(fraction, plan.max_fraction_per_bet)
    return {
        "stake": round(capped * bankroll, 2),
        "fraction": capped,
        "edge": edge,
        "kelly_full": kelly_fraction(probability, odds),
        "capped": bool(capped < fraction),
        "reason": "",
    }


def apply_daily_cap(stakes: list[float], bankroll: float, plan: StakePlan) -> list[float]:
    """Scale a day's stakes down together if their total exceeds the cap.

    Same-day bets are scaled rather than truncated so that no single match is
    silently dropped because of its position in the list.
    """
    total = float(sum(stakes))
    ceiling = plan.max_fraction_per_day * bankroll
    if total <= ceiling or total <= 0:
        return [round(stake, 2) for stake in stakes]
    scale = ceiling / total
    return [round(stake * scale, 2) for stake in stakes]
