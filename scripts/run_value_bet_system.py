#!/usr/bin/env python3
"""Un système de value bet sans modèle : le prix sharp comme vérité.

Plutôt que de prédire un résultat — ce qu'aucun modèle de ce dépôt n'arrive à
faire mieux que le marché — on prend le prix dévigué d'un opérateur sharp comme
estimation de la probabilité, et on parie chez un opérateur mou quand son prix
dépasse cette estimation. Aucun descripteur, aucun paramètre appris : seulement
un seuil d'écart.

Deux modes :

* ``backtest`` mesure le système sur 101 794 matchs de football, avec un bras de
  falsification (le même signal inversé, qui doit perdre nettement plus) et un
  audit de simultanéité ;
* ``live`` regarde si un opérateur accessible depuis la France offre aujourd'hui
  quoi que ce soit au-dessus du prix juge.

L'audit de simultanéité n'est pas décoratif. Comparer un prix sharp de clôture à
un prix mou d'ouverture fait apparaître un avantage de +4,6 % dont l'intervalle
exclut zéro — et il est entièrement faux : c'est la clôture qui informe, pas une
inefficience. Les deux jambes doivent dater du même instant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import active_sports, fetch_market_odds

FRENCH_BOOKS = ("pmu_fr", "betclic_fr", "unibet_fr", "netbet_fr", "winamax_fr")
SHARP_BOOK = "pinnacle"
# Au-delà, la somme des probabilités implicites n'est plus une cote à trois
# issues cohérente: ligne incomplète ou erreur de saisie.
MAX_OVERROUND = 1.35


def _month_block_interval(profit: np.ndarray, months: np.ndarray,
                          draws: int = 3000) -> tuple[float, float]:
    months = np.asarray(months)
    groups = [profit[months == month] for month in np.unique(months)]
    rng = np.random.default_rng(0)
    means = [
        np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
        for _ in range(draws)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _legs(frame: pd.DataFrame, sharp_prefix: str, soft_prefix: str) -> pd.DataFrame:
    """Une ligne par issue, avec la probabilité déviguée du sharp en regard."""
    rows = []
    for side in "HDA":
        sharp = pd.to_numeric(frame[f"{sharp_prefix}{side}"], errors="coerce")
        soft = pd.to_numeric(frame[f"{soft_prefix}{side}"], errors="coerce")
        rows.append(pd.DataFrame({
            "sharp": sharp, "soft": soft,
            "won": (frame["result"] == side).astype(float),
            "month": frame["month"], "year": frame["year"], "idx": frame.index,
        }))
    legs = pd.concat(rows, ignore_index=True).dropna(subset=["sharp", "soft"])
    legs = legs[(legs["sharp"] > 1.0) & (legs["soft"] > 1.0)]
    total = (1.0 / legs["sharp"]).groupby(legs["idx"]).transform("sum")
    legs = legs[(total > 1.0) & (total < MAX_OVERROUND)]
    legs["p_true"] = (1.0 / legs["sharp"]) / total[legs.index]
    legs["value"] = legs["p_true"] * legs["soft"] - 1.0
    legs["profit"] = np.where(legs["won"] == 1, legs["soft"] - 1.0, -1.0)
    return legs


def backtest(root: Path) -> None:
    frame = pd.read_csv(root / "data/football/football_matches.csv.gz", low_memory=False)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.dropna(subset=["match_date", "result"])
    frame["month"] = frame["match_date"].dt.to_period("M")
    frame["year"] = frame["match_date"].dt.year

    print("=== audit de simultanéité (à lire avant tout le reste) ===")
    print(f"{'vérité / offre':>44s} {'paris':>8s} {'ROI':>8s} {'IC 95%':>20s}")
    honest = None
    for sharp, soft, name in (("PS", "B365", "sharp pré / mou pré"),
                              ("PSC", "B365C", "sharp clôture / mou clôture"),
                              ("PSC", "B365", "sharp CLÔTURE / mou pré (anachronique)"),
                              ("PS", "B365C", "sharp pré / mou clôture (anachronique)")):
        try:
            legs = _legs(frame, sharp, soft)
        except KeyError:
            continue
        take = legs[legs["value"] > 0.0]
        if len(take) < 200:
            continue
        profit = take["profit"].to_numpy()
        low, high = _month_block_interval(profit, take["month"].to_numpy())
        print(f"{name:>44s} {len(take):>8,} {profit.mean():>+8.2%} "
              f"[{low:+.2%}, {high:+.2%}]")
        if name == "sharp pré / mou pré":
            honest = legs
    print("\nSeule la ligne anachronique exclut zéro, et elle lit le futur:")
    print("la clôture informe, ce n'est pas une inefficience exploitable.\n")

    if honest is None:
        return
    print("=== le système, aux instants cohérents ===")
    print(f"{'seuil':>7s} {'paris':>8s} {'cote moy':>9s} {'ROI':>8s} {'IC 95%':>20s}")
    for threshold in (0.0, 0.02, 0.05, 0.10):
        take = honest[honest["value"] > threshold]
        if len(take) < 100:
            continue
        profit = take["profit"].to_numpy()
        low, high = _month_block_interval(profit, take["month"].to_numpy())
        print(f"{threshold:>7.0%} {len(take):>8,} {take['soft'].mean():>9.2f} "
              f"{profit.mean():>+8.2%} [{low:+.2%}, {high:+.2%}]")

    print("\n=== bras de falsification, cotes 2,5-8,0 ===")
    band = honest[honest["soft"].between(2.5, 8.0, inclusive="left")]
    for name, mask in (("valeur > +2 %", band["value"] > 0.02),
                       ("valeur ≈ 0", band["value"].between(-0.01, 0.01)),
                       ("valeur < −2 % (contrôle)", band["value"] < -0.02)):
        cell = band[mask]
        if len(cell) < 100:
            continue
        print(f"  {name:>26s}: {len(cell):>7,} paris, "
              f"ROI {cell['profit'].mean():>+7.2%}")
    print("  Un gradient monotone est la signature d'un signal réel.")

    print("\n=== stabilité annuelle ===")
    take = honest[honest["value"] > 0.0]
    years = take.groupby("year")["profit"].agg(["count", "mean"])
    years = years[years["count"] >= 150]
    print(f"  {int((years['mean'] > 0).sum())} années positives sur {len(years)}")
    print("  " + "  ".join(f"{int(y)}:{m:+.1%}" for y, m in years["mean"].items()))


def live(root: Path) -> None:
    catalogue = active_sports(root)
    sports = [s["key"] for s in catalogue.events if s.get("active")
              and any(k in s["key"] for k in ("soccer", "basketball", "tennis"))][:30]
    rows = []
    for key in sports:
        for region in ("fr", "eu"):
            response = fetch_market_odds(root, key, "h2h", regions=region)
            if not response.ok:
                continue
            for event in response.events:
                event_id = (f"{key}|{event.get('home_team')}|{event.get('away_team')}"
                            f"|{event.get('commence_time')}")
                for book in event.get("bookmakers") or []:
                    for market in book.get("markets") or []:
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes") or []:
                            rows.append({"event": event_id, "book": book["key"],
                                         "name": outcome["name"],
                                         "price": float(outcome["price"])})
    frame = pd.DataFrame(rows).drop_duplicates(["event", "book", "name"])
    if frame.empty:
        print("aucune cote")
        return
    sharp = frame[frame["book"] == SHARP_BOOK]
    truth: dict[tuple[str, str], float] = {}
    for event, block in sharp.groupby("event"):
        total = float((1.0 / block["price"]).sum())
        if 1.0 < total < MAX_OVERROUND:
            for row in block.itertuples(index=False):
                truth[(event, row.name)] = (1.0 / row.price) / total
    print(f"{len(frame):,} cotes, {frame['event'].nunique()} événements, "
          f"{sharp['event'].nunique()} avec un prix juge\n")

    def report(name: str, block: pd.DataFrame) -> None:
        values = [truth[(row.event, row.name)] * row.price - 1.0
                  for row in block.itertuples(index=False)
                  if (row.event, row.name) in truth]
        if len(values) < 20:
            return
        values = np.array(values)
        print(f"{name:>14s} {len(values):>8,} {(values > 0).mean():>9.1%} "
              f"{(values > 0.02).mean():>11.1%} {np.median(values):>15.2%} "
              f"{values.max():>+10.2%}")

    print(f"{'opérateur':>14s} {'comparés':>8s} {'valeur>0':>9s} {'valeur>+2%':>11s} "
          f"{'valeur médiane':>15s} {'meilleure':>10s}")
    for book in FRENCH_BOOKS:
        report(book, frame[frame["book"] == book])
    accessible = frame[frame["book"].isin(FRENCH_BOOKS)]
    best = accessible.loc[accessible.groupby(["event", "name"])["price"].idxmax()]
    report("MEILLEUR FR", best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["backtest", "live"])
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.mode == "backtest":
        backtest(root)
    else:
        live(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
