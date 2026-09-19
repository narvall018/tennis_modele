#!/usr/bin/env python3
"""Une seule sortie : la meilleure opportunité du moment, tous sports confondus.

Interroge une fois chaque sport demandé, fait répondre les stratégies qui ont le
droit de sélectionner, et classe le tout selon la règle figée dans
`src/app/best_opportunity.py` : un arbitrage exploitable passe avant une espérance
de modèle, quel que soit le pourcentage affiché.

Chaque sport coûte une requête du quota mensuel (500 en offre gratuite), donc la
liste est explicite et rien n'est interrogé deux fois. Aucune mise n'est envoyée.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import best_opportunity as ranking
from src.app import wta_kernel_strategy as wta
from src.app.odds_api import MMA_SPORT_KEY, TENNIS_SPORT_PREFIX, active_sports, fetch_h2h_odds


def parse_surfaces(values):
    surfaces = {}
    for item in values or []:
        name, separator, surface = item.partition("=")
        surface = surface.strip().title()
        if not separator or not name.strip() or surface not in ranking.SURFACES:
            raise SystemExit(
                f"--wta-surface attend « Tournoi=Hard|Clay|Grass », reçu : {item!r}"
            )
        surfaces[name.strip()] = surface
    return surfaces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--regions", default="eu,uk,fr")
    parser.add_argument("--sports", nargs="*", default=None,
                        help="clés de sport ; par défaut tennis actifs + MMA")
    parser.add_argument("--wta-surface", nargs="*", default=None, metavar="TOURNOI=SURFACE",
                        help="confirme surface et simple tableau principal pour ce tournoi WTA")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="bankroll de simulation en euros, pour la mise suggérée")
    args = parser.parse_args()
    root = args.project_root.resolve()
    surfaces = parse_surfaces(args.wta_surface)

    sports = args.sports
    if not sports:
        catalogue = active_sports(root)
        if not catalogue.ok:
            print(f"catalogue indisponible : {catalogue.error}")
            return 1
        sports = [
            sport["key"] for sport in catalogue.events
            if sport.get("active") and (
                str(sport["key"]).startswith(TENNIS_SPORT_PREFIX)
                or sport["key"] == MMA_SPORT_KEY
            )
        ]
    print(f"sports interrogés : {', '.join(sports)}\n")

    now = pd.Timestamp.now("UTC")
    events_by_sport, remaining = {}, None
    for sport in sports:
        response = fetch_h2h_odds(root, sport, regions=args.regions)
        remaining = response.remaining if response.remaining is not None else remaining
        if not response.ok:
            print(f"  {sport} : {response.error}")
            continue
        events_by_sport[sport] = response.events
        print(f"  {sport} : {len(response.events)} marchés")

    rows = ranking.arbitrage_rows(events_by_sport, now)
    tennis_events = [event for events in events_by_sport.values() for event in events]
    rows += ranking.atp_rows(tennis_events, now)

    wta_notes = {"surfaces_a_confirmer": [], "ecartes": []}
    if any(str(event.get("sport_key", "")).startswith("tennis_wta_") for event in tennis_events):
        try:
            bundle = wta.load_bundle(root)
            wta_found, wta_notes = ranking.wta_rows(bundle, tennis_events, surfaces, now)
            rows += wta_found
        except (ValueError, OSError) as error:
            print(f"\nWTA hors service : {error}")

    notes = ranking.models_without_selection_rights(root)
    print("\nSans droit de sélection, couverts seulement par l'arbitrage :")
    for sport, note in sorted(notes.items()):
        print(f"  {sport} : {note}")
    if wta_notes["surfaces_a_confirmer"]:
        print("\nTournois WTA sans surface confirmée (non classés, jamais devinés) : "
              + ", ".join(wta_notes["surfaces_a_confirmer"]))
        print("  les ajouter avec --wta-surface \"Nom du tournoi=Hard\"")
    for skipped in wta_notes["ecartes"]:
        print(f"\nWTA écarté — {skipped['event']} ({skipped['bookmaker']}) : {skipped['raison']}")

    ordered = ranking.rank(rows)
    if not rows:
        print("\nAucune opportunité, d'aucun palier.")
    else:
        frame = pd.DataFrame([{k: row[k] for k in
                               ["tier_label", "sport", "event", "pick", "books", "odds",
                                "metric", "metric_kind", "exploitable", "blocked_reason"]}
                              for row in rows])
        pd.set_option("display.width", 220)
        print("\n" + frame.to_string(index=False))

    if ordered:
        top = ordered[0]
        plan = ranking.stake_plan(top, int(round(args.bankroll*100)))
        print("\n=== meilleure opportunité ===")
        print(f"palier      : {top['tier_label']} ({top['strategy_id']})")
        print(f"sport       : {top['sport']} · {top['competition']}")
        print(f"rencontre   : {top['event']} — début {top['start']}")
        print(f"sélection   : {top['pick']} chez {top['books']} à {top['odds']}")
        print(f"mesure      : {top['metric']:+.2%} ({top['metric_kind']})")
        for leg in plan["legs"]:
            print(f"mise        : {leg['stake_cents']/100:.2f} € sur {leg['outcome']} "
                  f"chez {leg['bookmaker']} à {leg['odds']}")
        print(f"plafond jour: {plan['cap_cents']/100:.2f} € pour une bankroll de {args.bankroll:.2f} €")
        if top["tier"] == ranking.MODEL_TIER:
            print("réserve     : espérance estimée par une règle dont le filtre de validation "
                  "a échoué ; ce n'est pas un gain démontré.")
        else:
            print("réserve     : gain arithmétique seulement si les deux jambes sont prises "
                  "aux prix relevés, avant que l'un des books ne bouge.")
        print("argent réel : non autorisé par ce dépôt ; simulation uniquement.")
    else:
        print("\nAucune opportunité exploitable : ne rien miser est la sortie normale de ce scan.")

    output = root/"models"/"best_opportunity_scan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "scanned_at_utc": now.isoformat(), "regions": args.regions, "sports": sports,
        "remaining_requests": remaining, "wta_surfaces_confirmed": surfaces,
        "wta_notes": wta_notes, "models_without_selection_rights": notes,
        "candidates": rows, "best": ordered[0] if ordered else None,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nquota restant : {remaining} · rapport : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
