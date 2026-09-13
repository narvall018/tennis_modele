#!/usr/bin/env python3
"""Combien vaut la qualité d'exécution, et à quelle distance est le seuil ?

Toutes les pistes de prédiction ont échoué parce qu'il faut battre 5 à 7 % de
marge. Mais le biais favori-outsider ne répartit pas cette marge : il en pose
l'essentiel sur les outsiders. Sur les gros favoris, le prélèvement effectif
tombe à 1-2 %, et la barre devient franchissable — non par un meilleur modèle,
mais par un meilleur prix.

Ce script mesure donc la seule chose qui reste : **le rendement d'un gros favori
en fonction de la surmarge du book qui le cote**. Trois sources historiques
donnent chacune un couple (surmarge, rendement), la relation se lit directement,
et elle dit à quelle surmarge un gros favori cesse de perdre.

Le prix français est ensuite mesuré en direct chez les cinq opérateurs agréés
ANJ de l'API, ce qui situe ce que l'on peut réellement atteindre sur cette
échelle. C'est la formulation la plus précise du mur juridictionnel : il ne
s'agit plus de dire que l'arbitrage est hors de portée, mais de chiffrer de
combien de points de surmarge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import fetch_market_odds

# La tranche où le biais favori-outsider laisse le moins de marge à franchir.
FAVOURITE_BAND = (1.10, 1.30)
MAJOR_LEAGUES = ("soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
                 "soccer_germany_bundesliga", "soccer_france_ligue_one",
                 "soccer_uefa_champs_league")
HISTORICAL_SOURCES = {"Bet365": ("B365H", "B365D", "B365A"),
                      "Pinnacle": ("PSH", "PSD", "PSA"),
                      "Max panel": ("MaxH", "MaxD", "MaxA")}


def _legs(frame: pd.DataFrame, columns: tuple[str, str, str]) -> pd.DataFrame:
    rows = []
    for side, column in zip("HDA", columns):
        odds = pd.to_numeric(frame[column], errors="coerce")
        keep = odds.notna() & odds.gt(1.0)
        rows.append(pd.DataFrame({
            "odds": odds[keep],
            "won": (frame["result"] == side).astype(float)[keep],
            "month": frame.loc[keep, "month"],
        }))
    legs = pd.concat(rows, ignore_index=True)
    legs["gain"] = legs["won"] * legs["odds"]
    return legs


def _overround(frame: pd.DataFrame, columns: tuple[str, str, str]) -> float:
    prices = frame[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    total = (1.0 / prices).sum(axis=1)
    return float(total[(total > 1.0) & (total < 1.6)].median())


def _month_block_interval(values: np.ndarray, months: np.ndarray,
                          draws: int = 3000) -> tuple[float, float]:
    months = np.asarray(months)
    groups = [values[months == month] for month in np.unique(months)]
    rng = np.random.default_rng(0)
    means = [
        np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
        for _ in range(draws)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def historical_ladder(root: Path) -> list[dict[str, float]]:
    """Pour chaque source: sa surmarge, et ce que rend un gros favori chez elle."""
    frame = pd.read_csv(root / "data/football/football_matches.csv.gz", low_memory=False)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.dropna(subset=["match_date", "result"])
    frame["month"] = frame["match_date"].dt.to_period("M")

    rungs = []
    print(f"{len(frame):,} matchs de football, tranche favori "
          f"{FAVOURITE_BAND[0]}-{FAVOURITE_BAND[1]}\n")
    print(f"{'source':>12s} {'surmarge':>9s} {'n':>7s} {'r':>8s} {'IC 95%':>20s}")
    for name, columns in HISTORICAL_SOURCES.items():
        legs = _legs(frame, columns)
        cell = legs[legs["odds"].between(*FAVOURITE_BAND, inclusive="left")]
        gain = cell["gain"].to_numpy()
        low, high = _month_block_interval(gain, cell["month"].to_numpy())
        overround = _overround(frame, columns) - 1.0
        rungs.append({"source": name, "overround": overround, "r": float(gain.mean())})
        print(f"{name:>12s} {overround:>9.2%} {len(cell):>7,} {gain.mean():>8.4f} "
              f"[{low:.4f}, {high:.4f}]")
    return rungs


def french_overrounds(root: Path) -> dict[str, float]:
    """Surmarge réellement pratiquée par les opérateurs agréés, en direct."""
    rows = []
    for league in MAJOR_LEAGUES:
        for region in ("fr", "eu"):
            response = fetch_market_odds(root, league, "h2h", regions=region)
            if not response.ok:
                continue
            for event in response.events:
                for book in event.get("bookmakers") or []:
                    for market in book.get("markets") or []:
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes") or []:
                            rows.append({
                                "region": region,
                                "event": f"{event.get('home_team')}|{event.get('away_team')}",
                                "book": book["key"], "name": outcome["name"],
                                "price": float(outcome["price"]),
                            })
    frame = pd.DataFrame(rows).drop_duplicates(["event", "book", "name"])
    if frame.empty:
        return {}

    def median_overround(block: pd.DataFrame) -> float | None:
        totals = []
        for _, event in block.groupby("event"):
            if event["name"].nunique() != len(event):
                continue
            total = float((1.0 / event["price"]).sum())
            if 1.0 < total < 1.6:
                totals.append(total)
        return float(np.median(totals)) - 1.0 if len(totals) >= 5 else None

    measured: dict[str, float] = {}
    french = frame[frame["region"] == "fr"]
    for book, block in french.groupby("book"):
        value = median_overround(block)
        if value is not None:
            measured[book] = value
    best = french.loc[french.groupby(["event", "name"])["price"].idxmax()]
    value = median_overround(best)
    if value is not None:
        measured["MEILLEUR DES 5 FRANÇAIS"] = value
    european = frame[frame["region"] == "eu"]
    for book in ("pinnacle",):
        value = median_overround(european[european["book"] == book])
        if value is not None:
            measured[f"{book} (hors France)"] = value
    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--skip-live", action="store_true",
                        help="ne pas interroger l'API (économise du quota)")
    args = parser.parse_args()
    root = args.project_root.resolve()

    rungs = historical_ladder(root)
    overrounds = np.array([rung["overround"] for rung in rungs])
    returns = np.array([rung["r"] for rung in rungs])
    slope, intercept = np.polyfit(overrounds, returns, 1)
    break_even = float((1.0 - intercept) / slope)
    print(f"\npente {slope:+.3f}: le favori ne porte qu'environ {-slope*100:.0f} % "
          f"de la marge, le reste tombe sur l'outsider")
    print(f"seuil de rentabilité: surmarge de {break_even:.2%}")

    if args.skip_live:
        return 0
    measured = french_overrounds(root)
    if not measured:
        print("\naucune cote en direct: échelle non située")
        return 0
    print(f"\n{'accès':>28s} {'surmarge':>9s} {'r attendu':>10s} {'par pari':>12s}")
    for name, overround in sorted(measured.items(), key=lambda kv: -kv[1]):
        expected = intercept + slope * overround
        verdict = "RENTABLE" if expected > 1.0 else f"{(expected - 1) * 100:+.2f} %"
        print(f"{name:>28s} {overround:>9.2%} {expected:>10.4f} {verdict:>12s}")

    reachable = measured.get("MEILLEUR DES 5 FRANÇAIS")
    if reachable is not None:
        print(f"\nIl manque {reachable - break_even:.2%} de surmarge au meilleur "
              f"des cinq opérateurs français.")
    output = root / "models" / "execution_ladder.json"
    output.write_text(json.dumps({
        "band": FAVOURITE_BAND, "historical": rungs,
        "slope": float(slope), "intercept": float(intercept),
        "break_even_overround": break_even, "measured_overrounds": measured,
        "evidence": "EXPLORATORY; live overrounds are a single snapshot",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"détail écrit dans {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
