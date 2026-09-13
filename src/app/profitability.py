"""À partir de quelle cote un pari cesse-t-il de perdre, chez quel opérateur.

Aucun modèle de ce dépôt ne bat le marché : c'est le résultat de treize études,
et il ne bouge pas. Mais il reste une quantité mesurable et utile — **le
rendement d'un pari selon sa cote et selon la marge de l'opérateur**.

Deux faits, mesurés et non supposés :

1. le biais favori-outsider fait que la marge n'est pas répartie. Sur 191 458
   matchs de football chez Bet365, une jambe à 1,15 rend 0,985 et une jambe à
   10,00 rend 0,755. La marge effective va de 1,5 % à 24 % selon la cote ;
2. la marge de l'opérateur décale toute la courbe. Un point de surmarge en plus
   coûte environ 0,53 point de rendement au football, 0,62 à l'ATP, 0,72 à la
   WTA — le favori portant d'autant plus de marge que le marché a moins d'issues.

De là découle un seuil exploitable : chez un opérateur donné, il existe une cote
au-dessus de laquelle plus rien n'est jouable. Ce module la calcule, et rien
d'autre. Il ne prédit aucun résultat.

Les courbes sont figées dans `models/profitability_curves.json` plutôt que
recalculées : les rejouer à chaque affichage coûterait 191 458 lignes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CURVES_PATH = Path("models/profitability_curves.json")
# En dessous, une mise de Kelly n'a plus de sens: l'avantage est dans le bruit
# de mesure de la courbe elle-même.
MINIMUM_EDGE = 0.002


@dataclass(frozen=True)
class Assessment:
    """Ce qu'on peut dire d'une cote précise chez un opérateur précis."""
    sport: str
    odds: float
    overround: float
    expected_return: float
    kelly_fraction: float

    @property
    def edge(self) -> float:
        return self.expected_return - 1.0

    @property
    def playable(self) -> bool:
        return self.edge > MINIMUM_EDGE


@lru_cache(maxsize=1)
def load_curves(root_text: str) -> dict:
    path = Path(root_text) / CURVES_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _curve_for(curves: dict, sport: str) -> dict | None:
    """Le tennis a deux circuits mesurés; tout le reste retombe sur le football."""
    key = sport.lower()
    if key in curves:
        return curves[key]
    if key.startswith("tennis_wta") or "wta" in key:
        return curves.get("wta")
    if key.startswith("tennis") or "atp" in key:
        return curves.get("atp")
    if key.startswith("soccer") or key.startswith("football"):
        return curves.get("football")
    return None


def reference_return(curve: dict, odds: float) -> float:
    """Rendement mesuré à cette cote, interpolé sur le logarithme de la cote."""
    grid = curve.get("grid") or []
    if not grid:
        return float("nan")
    points = np.log([row["odds"] for row in grid])
    values = np.array([row["r"] for row in grid])
    # Hors de la plage mesurée, on tient la valeur du bord plutôt que d'extrapoler
    # une droite: le rendement s'effondre vite sur les très grosses cotes et une
    # extrapolation linéaire y donnerait des chiffres absurdes.
    return float(np.interp(np.log(max(odds, 1.0001)), points, values))


def assess(root: Path, sport: str, odds: float, overround: float) -> Assessment | None:
    """Rendement attendu et mise de Kelly pour une cote chez un opérateur donné."""
    curves = load_curves(str(root))
    curve = _curve_for(curves, sport)
    if curve is None or odds <= 1.0:
        return None
    shifted = reference_return(curve, odds) - curve["slope"] * (
        overround - curve["reference_overround"])
    edge = shifted - 1.0
    fraction = edge / (odds - 1.0) if odds > 1.0 and edge > 0 else 0.0
    return Assessment(sport=sport, odds=float(odds), overround=float(overround),
                      expected_return=float(shifted), kelly_fraction=float(fraction))


def break_even_odds(root: Path, sport: str, overround: float) -> float | None:
    """La cote au-delà de laquelle plus rien n'est jouable chez cet opérateur.

    Renvoie None si même la cote la plus courte mesurée perd — le cas de tous les
    opérateurs français — car il n'existe alors aucun seuil à annoncer.
    """
    curves = load_curves(str(root))
    curve = _curve_for(curves, sport)
    if curve is None:
        return None
    grid = sorted(curve.get("grid") or [], key=lambda row: row["odds"])
    if not grid:
        return None
    shift = curve["slope"] * (overround - curve["reference_overround"])
    best = None
    for row in grid:
        if row["r"] - shift - 1.0 > MINIMUM_EDGE:
            best = row["odds"]
    return best


def market_overround(prices: list[float]) -> float | None:
    """Surmarge d'un marché complet; None si les cotes ne forment pas un marché."""
    usable = [p for p in prices if isinstance(p, (int, float)) and p > 1.0]
    if len(usable) < 2:
        return None
    total = sum(1.0 / price for price in usable)
    if not 0.8 < total < 1.8:
        return None
    return total - 1.0
