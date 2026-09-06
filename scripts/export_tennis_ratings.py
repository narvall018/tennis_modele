#!/usr/bin/env python3
"""Export a player-name → rating table so live tennis matches can be scored.

The odds feed identifies players by full name ("Alex Michelsen"), and so do the
enriched tables built from TennisMyLife — while the legacy odds tables abbreviate
("Michelsen A."). Running the ratings over the *enriched* tables therefore gives
a key that joins directly to the live feed, with no fuzzy matching and no chance
of attaching a rating to the wrong player.

ATP and WTA are exported separately. Their Elo scales are not comparable, and a
shared table would let a men's rating answer a women's match.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.elo_system import TennisEloEngine

TOURS = {
    "atp": "data/processed/atp_matches_enriched.csv.gz",
    "wta": "data/processed/wta_matches_enriched.csv.gz",
}

ROUND_TO_LEGACY = {
    "R128": "1st Round", "R64": "1st Round", "R32": "2nd Round", "R16": "3rd Round",
    "QF": "Quarterfinals", "SF": "Semifinals", "F": "The Final", "RR": "Round Robin",
}


def normalise_player(name: object) -> str:
    """Accent- and case-free key, so 'Iva Jović' and 'Iva Jovic' are one player."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().replace("-", " ").split())


def export_tour(root: Path, tour: str, relative: str) -> dict:
    frame = pd.read_csv(root / relative, low_memory=False)
    stream = pd.DataFrame({
        "Date": pd.to_datetime(frame["match_date"], errors="coerce"),
        "Player_1": frame["player_1_name"].astype(str).str.strip(),
        "Player_2": frame["player_2_name"].astype(str).str.strip(),
        "Winner": frame["winner_name"].astype(str).str.strip(),
        "Tournament": frame["tourney_name"].astype(str),
        "Surface": frame["surface"].fillna("Hard").astype(str),
        "Round": frame["round"].map(ROUND_TO_LEGACY).fillna("1st Round"),
        "Status": frame["match_status"].astype(str),
    }).dropna(subset=["Date"])
    stream = stream[stream["Player_1"].ne("") & stream["Player_2"].ne("")]
    stream = stream.sort_values("Date").reset_index(drop=True)
    print(f"{tour.upper()}: {len(stream):,} matchs", flush=True)

    engine = TennisEloEngine()
    engine.fit(stream, progress_callback=lambda message: print(f"  {message}", flush=True))

    rows = []
    for name, state in engine.get_all_ratings().items():
        rows.append({
            "tour": tour,
            "player": name,
            "player_key": normalise_player(name),
            "elo": state.global_elo,
            "elo_hard": state.surface_elo.get("Hard"),
            "elo_clay": state.surface_elo.get("Clay"),
            "elo_grass": state.surface_elo.get("Grass"),
            "matches": state.matches_played,
            "form_5": state.recent_form(5),
            "last_date": state.last_match_date,
        })
    table = pd.DataFrame(rows)
    # A player who has not appeared in years is not a live opinion.
    table["last_date"] = pd.to_datetime(table["last_date"], errors="coerce")
    output = root / "models" / "tennis"
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / f"{tour}_player_ratings.parquet", index=False)
    print(f"  {len(table):,} joueurs exportés")
    return {
        "tour": tour,
        "players": int(len(table)),
        "matches": int(len(stream)),
        "last_match": str(stream["Date"].max().date()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    summary = [export_tour(root, tour, relative) for tour, relative in TOURS.items()]
    metadata = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "tours": summary,
        "key": "player_key = nom complet, sans accent ni casse",
        "honest_note": (
            "Ratings Elo purs, sans cote. Ils fournissent une opinion affichable "
            "à côté du prix; ils ne le battent pas."
        ),
    }
    (root / "models" / "tennis" / "ratings_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
