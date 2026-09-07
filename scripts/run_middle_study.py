#!/usr/bin/env python3
"""Does a one-game tennis middle actually pay? The answer is a property of tennis.

A middle at 38.5 / 39.5 wins both legs only when the match ends on exactly 39
games. With both legs at 1.92 the pair loses about 4% of the stake otherwise, so
it is worth taking if and only if

    P(exactly 39) x 0.92  >  (1 - P) x 0.04    ->    P > 4.17%

Three passes, each answering the objection to the one before:

1. *unconditional* — how often does a completed match land on a given total?
   Objection: a bookmaker does not pick the line blindly.
2. *conditional on a pre-match forecast* — predict the total out-of-fold, then
   ask how often the result equals the predicted line. Objection: my forecaster
   is worse than a bookmaker's, so this understates P.
3. *sensitivity* — degrade my own forecast on purpose. If blurring the line
   barely moves P(exact), the limit is the intrinsic spread of a tennis match,
   which no bookmaker can reduce, and sharpening past my forecast cannot rescue
   the bet either.
4. *repricing* — passes 1 to 3 all hold the prices at 1.92/1.92 while moving the
   line, and that is not a thing anyone can do: a book quoting a line eight
   games below its own forecast prices it at 1.18, not 1.92. Price each leg at
   what it would actually cost and the answer stops depending on the line at
   all. That is what closes the question.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Both legs at 1.92: the pair costs 4.17% of a unit to miss and returns 92% to hit.
BREAK_EVEN = 0.0417
GAIN_IF_HIT = 0.92
COST_IF_MISSED = 0.04
MINIMUM_CELL = 200


def total_games(score: str) -> float:
    """Games played in a completed match, read off its published score."""
    if not isinstance(score, str):
        return np.nan
    if any(token in score.upper() for token in ("RET", "W/O", "DEF", "WALKOVER")):
        return np.nan
    total = sets = 0
    for part in score.split():
        match = re.match(r"^(\d+)-(\d+)", part)
        if match:
            total += int(match.group(1)) + int(match.group(2))
            sets += 1
    return float(total) if sets >= 2 else np.nan


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Wilson score interval — honest at the small rates a middle deals in."""
    z = 1.959963984540054
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def expected_value(rate: float) -> float:
    return rate * GAIN_IF_HIT - (1 - rate) * COST_IF_MISSED


def load_matches(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        root / "data/processed/atp_matches_enriched.csv.gz", low_memory=False,
        usecols=["score", "best_of", "match_status", "surface", "tourney_level",
                 "player_1_rank", "player_2_rank", "player_1_odds", "player_2_odds"])
    frame = frame[(frame["match_status"] == "completed") & (frame["best_of"] == 5)].copy()
    frame["games"] = frame["score"].map(total_games)
    return frame.dropna(subset=["games"])


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    rank_1 = pd.to_numeric(frame["player_1_rank"], errors="coerce")
    rank_2 = pd.to_numeric(frame["player_2_rank"], errors="coerce")
    odds_1 = pd.to_numeric(frame["player_1_odds"], errors="coerce")
    odds_2 = pd.to_numeric(frame["player_2_odds"], errors="coerce")
    return pd.DataFrame({
        # A close match lasts longer; the ranking gap is the cheapest proxy.
        "log_rank_gap": np.log1p((rank_1 - rank_2).abs()),
        "log_best_rank": np.log1p(np.minimum(rank_1, rank_2)),
        "log_worst_rank": np.log1p(np.maximum(rank_1, rank_2)),
        "grass": (frame["surface"] == "Grass").astype(float),
        "clay": (frame["surface"] == "Clay").astype(float),
        "slam": (frame["tourney_level"] == "G").astype(float),
        # Market disagreement is what a bookmaker's own total would lean on.
        "price_gap": (1 / odds_1 - 1 / odds_2).abs(),
    })


def unconditional(frame: pd.DataFrame) -> None:
    print("=== 1. sans conditionnement ===")
    counts = frame["games"].value_counts(normalize=True)
    print(f"{len(frame):,} matchs best-of-5, médiane {frame['games'].median():.0f} jeux")
    for target in (35, 37, 39, 41):
        rate = float(counts.get(float(target), 0.0))
        print(f"  P(exactement {target}) = {rate:6.2%}   seuil {BREAK_EVEN:.2%}   "
              f"{'rentable' if rate > BREAK_EVEN else 'perdant'}")
    print(f"  meilleur total atteignable: {counts.max():.2%} à {counts.idxmax():.0f} jeux\n")


def conditional(predicted: np.ndarray, target: np.ndarray) -> list[tuple]:
    print("=== 2. conditionnellement à une ligne prévue d'avance ===")
    print(f"{'ligne':>6s} {'n':>6s} {'P(pile)':>9s} {'IC 95%':>18s} {'EV':>9s}")
    cells = []
    for line in range(31, 45, 2):
        window = (predicted >= line - 0.5) & (predicted < line + 0.5)
        if window.sum() < MINIMUM_CELL:
            continue
        n, hits = int(window.sum()), int((target[window] == line).sum())
        low, high = wilson(hits, n)
        cells.append((line, n, hits, low, high))
        print(f"{line:6d} {n:6d} {hits / n:9.2%} [{low:6.2%},{high:6.2%}] "
              f"{expected_value(hits / n):+9.2%}")

    total_n = sum(cell[1] for cell in cells)
    total_hits = sum(cell[2] for cell in cells)
    rate = total_hits / total_n
    low, high = wilson(total_hits, total_n)
    print(f"\ntoutes lignes: {total_hits}/{total_n} = {rate:.2%} "
          f"IC [{low:.2%}, {high:.2%}], seuil {BREAK_EVEN:.2%}")
    print(f"  -> EV {expected_value(rate):+.2%}")

    # Six cells were tested. Two came back above the threshold, which is what
    # noise looks like when you test six things — so check what the best of six
    # does in a world where nothing is there.
    rng = np.random.default_rng(0)
    draws = np.array([
        [rng.binomial(n, rate) / n for _, n, *_ in cells] for _ in range(20000)
    ])
    observed = max(cell[2] / cell[1] for cell in cells)
    beaten = float((draws.max(axis=1) >= observed).mean())
    print(f"  sous H0 (P={rate:.2%} partout), la meilleure des {len(cells)} cellules "
          f"atteint {observed:.2%} dans {beaten:.1%} des tirages\n")
    return cells


def sensitivity(predicted: np.ndarray, target: np.ndarray) -> None:
    """The objection that matters: would a bookmaker's sharper line pay?"""
    print("=== 3. la finesse de la ligne est-elle le facteur limitant ? ===")

    def hit_rate(forecast: np.ndarray) -> float:
        line = np.round((forecast - 1) / 2) * 2 + 1
        keep = (line >= 31) & (line <= 43)
        return float((target[keep] == line[keep]).mean())

    print(f"{'bruit ajouté':>13s} {'MAE':>7s} {'P(pile)':>9s} {'EV':>9s}")
    rng = np.random.default_rng(0)
    for noise in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        # Average several draws so the curve is not itself a noise reading.
        rates, errors = [], []
        for _ in range(12 if noise else 1):
            blurred = predicted + rng.normal(0, noise, size=len(predicted))
            rates.append(hit_rate(blurred))
            errors.append(float(np.abs(blurred - target).mean()))
        rate = float(np.mean(rates))
        print(f"{noise:13.1f} {np.mean(errors):7.2f} {rate:9.2%} "
              f"{expected_value(rate):+9.2%}")
    print("\nSi cette colonne est plate, une ligne plus fine ne sauve pas le middle:")
    print("ce qui limite P(pile) est la dispersion du match, pas la prévision.")


def repricing(predicted: np.ndarray, target: np.ndarray, margin: float) -> None:
    """The objection that settles it: the prices move with the line.

    A middle placed below the forecast lands more often — and pays less, because
    the Over leg is now a near-certainty priced accordingly. Charge each leg what
    a book would charge and the two effects cancel exactly, because it is the
    same bet seen from a different line.
    """
    print("\n=== 4. et si on facture chaque jambe à son vrai prix ? ===")
    base = np.round((predicted - 1) / 2) * 2 + 1
    keep = (base >= 31) & (base <= 43)
    base, actual = base[keep], target[keep]

    print(f"{'décalage':>9s} {'cote over':>10s} {'cote under':>11s} {'P(pile)':>9s} "
          f"{'seuil réel':>11s} {'EV':>9s}")
    for offset in range(-8, 9, 2):
        line = base + offset
        # Over (line - 0.5) at one book, Under (line + 0.5) at another.
        over_price = (1.0 / float((actual >= line).mean())) * (1.0 - margin)
        under_price = (1.0 / float((actual <= line).mean())) * (1.0 - margin)
        exact = float((actual == line).mean())

        single = 1.0 / (1.0 / over_price + 1.0 / under_price)
        gain, cost = 2.0 * single - 1.0, 1.0 - single
        print(f"{offset:+9d} {over_price:10.2f} {under_price:11.2f} {exact:9.2%} "
              f"{cost / (gain + cost):11.2%} "
              f"{exact * gain - (1 - exact) * cost:+9.2%}")

    print(f"\nL'EV ne bouge pas: elle vaut la marge, {-margin:.2%}, où qu'on place le")
    print("middle. Un middle n'a aucun avantage propre — sa seule valeur vient de")
    print("deux books en désaccord sur la ligne, c'est-à-dire d'un prix meilleur que")
    print("le vrai. C'est exactement l'arbitrage, avec les mêmes obstacles.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=0.045,
                        help="marge prise par chaque book sur un marché de totaux")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    frame = load_matches(root)
    unconditional(frame)

    features = build_features(frame)
    usable = features.notna().all(axis=1)
    features, target = features[usable], frame.loc[usable, "games"].to_numpy()
    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=250, random_state=0)
    # Out-of-fold: no match ever helps forecast its own total.
    predicted = cross_val_predict(model, features, target, cv=5)
    print(f"prévision hors échantillon sur {len(features):,} matchs, "
          f"erreur absolue moyenne {np.abs(predicted - target).mean():.2f} jeux\n")

    conditional(predicted, target)
    sensitivity(predicted, target)
    repricing(predicted, target, args.margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
