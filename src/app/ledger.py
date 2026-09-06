"""Record a recommended bet into the app's existing ledger.

The ledger already knows how to hold events, bets, bankrolls and settlement; what
it did not have was a way in from the prediction pages, and it refused football
outright. This module supplies both.

Two design points worth stating:

* **A bet is written with the price and probability seen at the moment it was
  taken**, not recomputed later. A ledger whose odds drift with the market can no
  longer answer the only question it exists for — did this decision pay?
* **The same match is never duplicated.** Recording twice from the same row
  reuses the event, so a bankroll cannot be double-counted by an impatient click.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPORTS = ("tennis", "ufc", "football")
_ALLOWED = ", ".join(f"'{sport}'" for sport in SPORTS)


def migrate_sports(conn: sqlite3.Connection) -> bool:
    """Widen the sport constraint to include football.

    SQLite cannot alter a CHECK in place, so the tables are rebuilt. The rebuild
    is skipped when the constraint already mentions football, which makes this
    safe to call on every start-up.
    """
    changed = False
    for table in ("events", "bets"):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row or not row[0] or "football" in row[0]:
            continue
        original = row[0]
        widened = original.replace(
            "sport TEXT NOT NULL CHECK (sport IN ('tennis', 'ufc'))",
            f"sport TEXT NOT NULL CHECK (sport IN ({_ALLOWED}))",
        )
        # SQLite drops "IF NOT EXISTS" from the SQL it stores, so the rename has
        # to tolerate either spelling.
        rebuilt = re.sub(
            rf"^CREATE TABLE (IF NOT EXISTS )?[\"'`\[]?{table}[\"'`\]]?",
            f"CREATE TABLE {table}__migrated",
            widened,
            count=1,
        )
        if "__migrated" not in rebuilt or widened == original:
            continue
        columns = [
            item[1] for item in conn.execute(f"PRAGMA table_info({table})").fetchall()
        ]
        joined = ", ".join(columns)
        conn.execute(rebuilt)
        conn.execute(f"INSERT INTO {table}__migrated ({joined}) SELECT {joined} FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}__migrated RENAME TO {table}")
        changed = True
    if changed:
        conn.commit()
    return changed


def _sport_code(sport_label: str) -> str:
    lowered = sport_label.strip().lower()
    if lowered.startswith("foot"):
        return "football"
    if lowered.startswith("ufc") or lowered.startswith("mma"):
        return "ufc"
    return "tennis"


def find_existing_event(
    conn: sqlite3.Connection, sport: str, participant_a: str, participant_b: str
) -> int | None:
    """An event already recorded for the same pairing, whichever way round."""
    row = conn.execute(
        """
        SELECT id FROM events
        WHERE sport = ?
          AND status = 'upcoming'
          AND ((participant_a = ? AND participant_b = ?)
            OR (participant_a = ? AND participant_b = ?))
        ORDER BY id DESC LIMIT 1
        """,
        (sport, participant_a, participant_b, participant_b, participant_a),
    ).fetchone()
    return int(row[0]) if row else None


def parse_matchup(label: str) -> tuple[str, str]:
    """Split a displayed fixture back into its two participants."""
    cleaned = label.split("  (")[0]
    for separator in (" – ", " - ", " vs ", " — "):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            return left.strip(), right.strip()
    return cleaned.strip(), ""


def record_recommendation(
    db_path: Path,
    user_id: int,
    row: dict[str, Any],
    stake: float,
) -> tuple[bool, str]:
    """Create the event if needed and place the bet, in one transaction."""
    if stake <= 0:
        return False, "Mise nulle: rien n'est enregistré."
    sport = _sport_code(str(row.get("sport", "")))
    participant_a, participant_b = parse_matchup(str(row.get("rencontre", "")))
    if not participant_a or not participant_b:
        return False, f"Rencontre illisible: {row.get('rencontre')!r}"

    pick = str(row.get("pari", "")).strip()
    odds = float(row.get("cote_pari") or 0.0)
    probability = float(row.get("p_pari") or 0.0)
    if odds <= 1.0:
        return False, "Cote invalide."

    # Football picks are outcome labels, not participant names.
    if sport == "football":
        side = "a" if pick.lower().startswith("dom") else "b"
        if pick.lower().startswith("nul"):
            return False, (
                "Le carnet ne gère que deux issues; un pari sur le nul ne peut pas "
                "y être enregistré tel quel."
            )
    else:
        side = "a" if pick == participant_a else "b" if pick == participant_b else ""
        if not side:
            return False, f"Le pari « {pick} » ne correspond à aucun des deux participants."

    odds_a = odds if side == "a" else None
    odds_b = odds if side == "b" else None
    probability_a = probability if side == "a" else 1.0 - probability
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        event_id = find_existing_event(conn, sport, participant_a, participant_b)
        if event_id is None:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    sport, title, participant_a, participant_b, event_datetime,
                    odds_a, odds_b, predicted_prob_a, predicted_prob_b,
                    stats_a, stats_b, status, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', 'upcoming', ?, ?, ?)
                """,
                (
                    sport, str(row.get("rencontre", "")), participant_a, participant_b,
                    str(row.get("quand", "")), odds_a, odds_b,
                    probability_a, 1.0 - probability_a, user_id, now, now,
                ),
            )
            event_id = int(cursor.lastrowid)
        else:
            # Refresh the quoted price on the side actually taken.
            conn.execute(
                f"UPDATE events SET odds_{side} = ?, updated_at = ? WHERE id = ?",
                (odds, now, event_id),
            )
        conn.execute(
            """
            INSERT INTO bets (
                user_id, sport, event_id, pick_side, odds, stake,
                model_probability, edge, ev, status, source, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', 'recommandation', ?, ?)
            """,
            (
                user_id, sport, event_id, side, odds, stake,
                probability,
                # `edge` records the disagreement with the price, `ev` the
                # expectation at the odds taken — both frozen at decision time.
                float(row.get("écart") or 0.0),
                float(row.get("espérance") or 0.0),
                f"score {float(row.get('score') or 0.0):.3f} · {row.get('quand', '')}",
                now,
            ),
        )
        conn.commit()
    except sqlite3.Error as error:
        conn.rollback()
        return False, f"Enregistrement refusé: {error}"
    finally:
        conn.close()
    return True, f"{stake:.2f} € sur {pick} à {odds:.2f} enregistré."
