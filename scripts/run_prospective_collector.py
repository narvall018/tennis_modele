#!/usr/bin/env python3
"""Enregistrer des prix horodatés, pour qu'il existe un jour une preuve neuve.

Tout ce dépôt rejoue le même historique, déjà exploré, et chaque conclusion porte
la même réserve : les années ont servi. La seule façon d'en sortir est de
constituer un jeu de données que personne n'a encore vu — des prix relevés avant
les matchs, horodatés, puis les résultats après coup.

Deux besoins couverts par la même collecte :

* **la valeur de clôture** — le prix pris valait-il mieux que le prix final ?
  C'est le seul instrument qui tranche en centaines de paris plutôt qu'en
  dizaines de milliers ;
* **les marchés jamais testés faute d'historique** — totaux de jeux au tennis,
  handicaps, dont aucune archive n'existe dans ce dépôt.

Le palier gratuit donne 500 requêtes par mois. Le script tient donc un budget :
il refuse de dépasser le quota quotidien qu'on lui fixe, et il journalise ce
qu'il consomme, plutôt que de découvrir un 401 au milieu d'une analyse.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import active_sports, fetch_market_odds

LOG_PATH = Path("models/prospective/price_snapshots.jsonl")
BUDGET_PATH = Path("models/prospective/quota_ledger.json")
# Un relevé n'a d'intérêt que s'il précède le match: après le coup d'envoi le
# prix suit le score et ne dit plus rien sur ce que le marché savait d'avance.
MINIMUM_LEAD_MINUTES = 10.0


def _load_budget(root: Path) -> dict:
    path = root / BUDGET_PATH
    if not path.exists():
        return {"days": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"days": {}}


def _save_budget(root: Path, ledger: dict) -> None:
    path = root / BUDGET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")


def _spent_today(ledger: dict) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return int(ledger.get("days", {}).get(today, 0))


def _charge(ledger: dict, count: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledger.setdefault("days", {})
    ledger["days"][today] = int(ledger["days"].get(today, 0)) + count


def _rows(response, sport: str, market: str, now: pd.Timestamp) -> list[dict]:
    """Aplatir une réponse en lignes, en gardant l'instant du relevé."""
    rows: list[dict] = []
    for event in response.events:
        starts = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(starts):
            continue
        lead = (starts - now).total_seconds() / 60.0
        if lead < MINIMUM_LEAD_MINUTES:
            continue
        for book in event.get("bookmakers") or []:
            updated = book.get("last_update")
            for block in book.get("markets") or []:
                if block.get("key") != market:
                    continue
                for outcome in block.get("outcomes") or []:
                    price = outcome.get("price")
                    if not isinstance(price, (int, float)) or price <= 1.0:
                        continue
                    rows.append({
                        "observed_at_utc": now.isoformat(),
                        "sport": sport, "market": market,
                        "event_id": (f"{sport}|{event.get('home_team')}"
                                     f"|{event.get('away_team')}|{event.get('commence_time')}"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "commence_time": event.get("commence_time"),
                        "lead_minutes": round(lead, 1),
                        "book": book.get("key"),
                        "book_last_update": updated,
                        "outcome": outcome.get("name"),
                        "point": outcome.get("point"),
                        "price": float(price),
                    })
    return rows


def collect(root: Path, sports: list[str], markets: list[str], regions: str,
            daily_budget: int) -> int:
    ledger = _load_budget(root)
    spent = _spent_today(ledger)
    if spent >= daily_budget:
        print(f"budget du jour déjà consommé ({spent}/{daily_budget})")
        return 0

    now = pd.Timestamp.now("UTC")
    rows: list[dict] = []
    calls = 0
    remaining = None
    for sport in sports:
        for market in markets:
            if spent + calls >= daily_budget:
                print(f"budget atteint après {calls} appels")
                break
            response = fetch_market_odds(root, sport, market, regions=regions)
            calls += 1
            if not response.ok:
                print(f"  {sport} {market}: {response.error}")
                continue
            remaining = response.remaining if response.remaining is not None else remaining
            found = _rows(response, sport, market, now)
            rows += found
            print(f"  {sport:38s} {market:8s} {len(found):>5d} lignes")

    _charge(ledger, calls)
    ledger["last_remaining"] = remaining
    ledger["last_run_utc"] = now.isoformat()
    _save_budget(root, ledger)

    if rows:
        path = root / LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(rows):,} lignes ajoutées, {calls} requêtes "
          f"({spent + calls}/{daily_budget} aujourd'hui)")
    if remaining is not None:
        print(f"quota mensuel restant: {remaining}")
    return len(rows)


def summarise(root: Path) -> None:
    path = root / LOG_PATH
    if not path.exists():
        print("aucun relevé encore enregistré")
        return
    frame = pd.read_json(path, lines=True)
    print(f"{len(frame):,} lignes, {frame['event_id'].nunique():,} événements")
    print(f"depuis {frame['observed_at_utc'].min()} jusqu'à {frame['observed_at_utc'].max()}")
    print(f"\n{'marché':>10s} {'lignes':>8s} {'événements':>11s} {'books':>6s}")
    for market, block in frame.groupby("market"):
        print(f"{market:>10s} {len(block):>8,} {block['event_id'].nunique():>11,} "
              f"{block['book'].nunique():>6d}")
    # Un événement relevé plusieurs fois est ce qui rend la clôture mesurable.
    counts = frame.groupby("event_id")["observed_at_utc"].nunique()
    print(f"\névénements avec au moins deux relevés: "
          f"{int((counts >= 2).sum()):,} sur {len(counts):,}")
    print("Ce sont eux qui permettront de mesurer la valeur de clôture.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "summary"])
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--markets", nargs="*", default=["h2h"])
    parser.add_argument("--regions", default="fr,eu")
    parser.add_argument("--sports", nargs="*", default=None)
    parser.add_argument("--daily-budget", type=int, default=12,
                        help="requêtes autorisées par jour (500/mois au palier gratuit)")
    args = parser.parse_args()
    root = args.project_root.resolve()

    if args.action == "summary":
        summarise(root)
        return 0

    sports = args.sports
    if not sports:
        catalogue = active_sports(root)
        if not catalogue.ok:
            print(f"catalogue indisponible: {catalogue.error}")
            return 1
        sports = [s["key"] for s in catalogue.events
                  if s.get("active") and str(s["key"]).startswith("tennis_")][:4]
    collect(root, sports, args.markets, args.regions, args.daily_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
