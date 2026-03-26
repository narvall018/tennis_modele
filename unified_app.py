import hashlib
import importlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


APP_TITLE = "djoudjou_predictor"
ADMIN_USERNAME = "narvall018"
DEFAULT_ADMIN_PASSWORD = os.getenv("NARVALL018_DEFAULT_PASSWORD", "Jumanji_75")
DEFAULT_INITIAL_BANKROLL = float(os.getenv("DEFAULT_INITIAL_BANKROLL", "1000"))

DB_PATH = Path("bets") / "unified_app.db"
LEGACY_TENNIS_BETS = Path("bets") / "bets.csv"
LEGACY_UFC_BETS = Path("predictor_ufc") / "bets" / "bets.csv"
LEGACY_TENNIS_BANKROLL = Path("bets") / "bankroll.json"
LEGACY_UFC_BANKROLL = Path("predictor_ufc") / "bets" / "bankroll.csv"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def safe_username(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    return cleaned or "anonymous"


def tennis_user_dir(username: str) -> Path:
    p = Path("bets") / "users" / safe_username(username) / "tennis"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ufc_user_dir(username: str) -> Path:
    p = Path("bets") / "users" / safe_username(username) / "ufc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_tennis_user_files(
    username: str,
    is_admin: bool,
    initial_bankroll: Optional[float] = None,
) -> None:
    user_dir = tennis_user_dir(username)
    bets_file = user_dir / "bets.csv"
    bankroll_file = user_dir / "bankroll.json"
    migration_marker = user_dir / ".migrated_from_legacy"

    if is_admin and not migration_marker.exists():
        if LEGACY_TENNIS_BETS.exists():
            shutil.copy2(LEGACY_TENNIS_BETS, bets_file)
        if LEGACY_TENNIS_BANKROLL.exists():
            shutil.copy2(LEGACY_TENNIS_BANKROLL, bankroll_file)
        migration_marker.write_text(now_iso(), encoding="utf-8")

    if not bets_file.exists():
        cols = [
            "bet_id",
            "date",
            "tournament",
            "round",
            "player_1",
            "player_2",
            "pick",
            "odds",
            "stake",
            "model_prob",
            "edge",
            "ev",
            "status",
            "result",
            "profit",
        ]
        pd.DataFrame(columns=cols).to_csv(bets_file, index=False)

    if not bankroll_file.exists():
        initial_value = float(
            DEFAULT_INITIAL_BANKROLL if initial_bankroll is None else max(0.0, float(initial_bankroll))
        )
        bankroll_file.write_text(
            json.dumps(
                {
                    "initial_bankroll": initial_value,
                    "updated": now_iso(),
                }
            ),
            encoding="utf-8",
        )


def ensure_ufc_user_files(
    username: str,
    is_admin: bool,
    initial_bankroll: Optional[float] = None,
) -> None:
    user_dir = ufc_user_dir(username)
    bets_file = user_dir / "bets.csv"
    bankroll_file = user_dir / "bankroll.csv"
    migration_marker = user_dir / ".migrated_from_legacy"

    if is_admin and not migration_marker.exists():
        if LEGACY_UFC_BETS.exists():
            shutil.copy2(LEGACY_UFC_BETS, bets_file)
        if LEGACY_UFC_BANKROLL.exists():
            shutil.copy2(LEGACY_UFC_BANKROLL, bankroll_file)
        migration_marker.write_text(now_iso(), encoding="utf-8")

    if not bets_file.exists():
        cols = [
            "bet_id",
            "date",
            "event",
            "fighter_red",
            "fighter_blue",
            "pick",
            "odds",
            "stake",
            "model_probability",
            "kelly_fraction",
            "edge",
            "ev",
            "status",
            "result",
            "profit",
            "roi",
        ]
        pd.DataFrame(columns=cols).to_csv(bets_file, index=False)

    if not bankroll_file.exists():
        initial_value = float(
            DEFAULT_INITIAL_BANKROLL if initial_bankroll is None else max(0.0, float(initial_bankroll))
        )
        pd.DataFrame(
            {
                "date": [datetime.now().strftime("%Y-%m-%d")],
                "amount": [initial_value],
                "action": ["initial"],
                "note": ["Bankroll initiale"],
            }
        ).to_csv(bankroll_file, index=False)


def ensure_user_legacy_files(
    username: str,
    is_admin: bool,
    initial_bankroll: Optional[float] = None,
) -> None:
    # Evite les checks/copies repetes a chaque rerun Streamlit.
    cache_key = "_legacy_files_ready_users"
    try:
        ready_users = set(st.session_state.get(cache_key, []))
    except Exception:
        ready_users = set()

    if username in ready_users:
        return

    ensure_tennis_user_files(
        username,
        is_admin=is_admin,
        initial_bankroll=initial_bankroll,
    )
    ensure_ufc_user_files(
        username,
        is_admin=is_admin,
        initial_bankroll=initial_bankroll,
    )

    ready_users.add(username)
    try:
        st.session_state[cache_key] = sorted(ready_users)
    except Exception:
        pass


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    return text


def tokenize_name(value: object) -> list[str]:
    text = normalize_text(value)
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def detect_pick_side(pick: object, participant_a: str, participant_b: str) -> Optional[str]:
    p_tokens = tokenize_name(pick)
    a_tokens = tokenize_name(participant_a)
    b_tokens = tokenize_name(participant_b)
    if not p_tokens:
        return None

    overlap_a = len(set(p_tokens) & set(a_tokens))
    overlap_b = len(set(p_tokens) & set(b_tokens))

    if overlap_a > overlap_b and overlap_a > 0:
        return "a"
    if overlap_b > overlap_a and overlap_b > 0:
        return "b"

    p_last = p_tokens[-1]
    if p_last in a_tokens and p_last not in b_tokens:
        return "a"
    if p_last in b_tokens and p_last not in a_tokens:
        return "b"
    return None


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_datetime_str(value: object) -> str:
    if value is None:
        return now_iso()
    text = str(value).strip()
    if not text:
        return now_iso()

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).isoformat(timespec="seconds")
    except ValueError:
        return now_iso()


def format_datetime(value: Optional[str]) -> str:
    if not value:
        return "Date non renseignee"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return value


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False

    if stored_hash.startswith("pbkdf2_sha256$"):
        parts = stored_hash.split("$")
        if len(parts) != 3:
            return False
        _, salt, expected = parts
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
        return secrets.compare_digest(digest, expected)

    # Compatibilite legacy SHA256 (ancien app UFC)
    if re.fullmatch(r"[a-f0-9]{64}", stored_hash):
        digest = hashlib.sha256(password.encode()).hexdigest()
        return secrets.compare_digest(digest, stored_hash)

    return False


@contextmanager
def db_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO app_meta(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def recompute_user_bankroll(conn: sqlite3.Connection, user_id: int) -> float:
    user = conn.execute(
        "SELECT initial_bankroll FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    if not user:
        return 0.0

    initial_bankroll = float(user["initial_bankroll"])
    closed_profit = conn.execute(
        """
        SELECT COALESCE(SUM(profit), 0.0) AS total
        FROM bets
        WHERE user_id = ? AND status = 'resolved'
        """,
        (user_id,),
    ).fetchone()["total"]
    open_stake = conn.execute(
        """
        SELECT COALESCE(SUM(stake), 0.0) AS total
        FROM bets
        WHERE user_id = ? AND status = 'open'
        """,
        (user_id,),
    ).fetchone()["total"]

    amount = float(initial_bankroll) + float(closed_profit) - float(open_stake)
    conn.execute(
        "UPDATE users SET current_bankroll = ? WHERE id = ?",
        (round(amount, 2), user_id),
    )
    return round(amount, 2)


def recompute_all_bankrolls(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id FROM users").fetchall()
    for row in rows:
        recompute_user_bankroll(conn, int(row["id"]))


def infer_combined_admin_initial_bankroll() -> float:
    parts: list[float] = []

    if LEGACY_TENNIS_BANKROLL.exists():
        try:
            data = json.loads(LEGACY_TENNIS_BANKROLL.read_text(encoding="utf-8"))
            value = to_float(data.get("initial_bankroll"))
            if value is not None and value > 0:
                parts.append(value)
        except Exception:
            pass

    if LEGACY_UFC_BANKROLL.exists():
        try:
            df = pd.read_csv(LEGACY_UFC_BANKROLL)
            if "action" in df.columns and "amount" in df.columns:
                initial_rows = df[df["action"].astype(str).str.lower() == "initial"]
                if not initial_rows.empty:
                    value = to_float(initial_rows.iloc[0]["amount"])
                    if value is not None and value > 0:
                        parts.append(value)
                elif not df.empty:
                    value = to_float(df.iloc[0]["amount"])
                    if value is not None and value > 0:
                        parts.append(value)
        except Exception:
            pass

    if parts:
        return float(sum(parts))
    return float(DEFAULT_INITIAL_BANKROLL)


def ensure_admin_user(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT id, is_admin FROM users WHERE username = ?",
        (ADMIN_USERNAME,),
    ).fetchone()
    if row:
        if int(row["is_admin"]) != 1:
            conn.execute(
                "UPDATE users SET is_admin = 1 WHERE id = ?",
                (int(row["id"]),),
            )
        ensure_user_legacy_files(ADMIN_USERNAME, is_admin=True)
        return int(row["id"])

    initial_bankroll = infer_combined_admin_initial_bankroll()
    password_hash = hash_password(DEFAULT_ADMIN_PASSWORD)
    now = now_iso()
    conn.execute(
        """
        INSERT INTO users(username, password_hash, is_admin, initial_bankroll, current_bankroll, created_at)
        VALUES (?, ?, 1, ?, ?, ?)
        """,
        (ADMIN_USERNAME, password_hash, initial_bankroll, initial_bankroll, now),
    )
    ensure_user_legacy_files(ADMIN_USERNAME, is_admin=True)
    return int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])


def init_database() -> None:
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                initial_bankroll REAL NOT NULL DEFAULT 1000,
                current_bankroll REAL NOT NULL DEFAULT 1000,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL CHECK (sport IN ('tennis', 'ufc')),
                title TEXT,
                participant_a TEXT NOT NULL,
                participant_b TEXT NOT NULL,
                event_datetime TEXT,
                odds_a REAL,
                odds_b REAL,
                predicted_prob_a REAL,
                predicted_prob_b REAL,
                stats_a TEXT,
                stats_b TEXT,
                status TEXT NOT NULL DEFAULT 'upcoming' CHECK (status IN ('upcoming', 'completed')),
                winner_side TEXT CHECK (winner_side IN ('a', 'b', 'void')),
                created_by INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sport TEXT NOT NULL CHECK (sport IN ('tennis', 'ufc')),
                event_id INTEGER NOT NULL,
                pick_side TEXT NOT NULL CHECK (pick_side IN ('a', 'b')),
                odds REAL NOT NULL,
                stake REAL NOT NULL CHECK (stake > 0),
                model_probability REAL,
                edge REAL,
                ev REAL,
                status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
                result TEXT CHECK (result IN ('win', 'loss', 'void')),
                profit REAL,
                source TEXT NOT NULL DEFAULT 'manual',
                notes TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_sport_status ON events(sport, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_user ON bets(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_event ON bets(event_id)")
        admin_id = ensure_admin_user(conn)
        try:
            maybe_migrate_legacy_data(conn, admin_id)
        except Exception as exc:
            set_meta(conn, "legacy_migration_v1_error", f"{type(exc).__name__}: {exc}")
        conn.commit()


def get_or_create_event(
    conn: sqlite3.Connection,
    *,
    sport: str,
    title: str,
    participant_a: str,
    participant_b: str,
    event_datetime: str,
    created_by: int,
    odds_a: Optional[float] = None,
    odds_b: Optional[float] = None,
    predicted_prob_a: Optional[float] = None,
    predicted_prob_b: Optional[float] = None,
    stats_a: str = "",
    stats_b: str = "",
) -> int:
    row = conn.execute(
        """
        SELECT id FROM events
        WHERE sport = ? AND COALESCE(title, '') = ? AND participant_a = ? AND participant_b = ?
          AND COALESCE(event_datetime, '') = ?
        """,
        (sport, title or "", participant_a, participant_b, event_datetime or ""),
    ).fetchone()
    if row:
        event_id = int(row["id"])
        conn.execute(
            """
            UPDATE events
            SET odds_a = COALESCE(odds_a, ?),
                odds_b = COALESCE(odds_b, ?),
                predicted_prob_a = COALESCE(predicted_prob_a, ?),
                predicted_prob_b = COALESCE(predicted_prob_b, ?),
                stats_a = CASE WHEN (stats_a IS NULL OR stats_a = '') THEN ? ELSE stats_a END,
                stats_b = CASE WHEN (stats_b IS NULL OR stats_b = '') THEN ? ELSE stats_b END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                odds_a,
                odds_b,
                predicted_prob_a,
                predicted_prob_b,
                stats_a,
                stats_b,
                now_iso(),
                event_id,
            ),
        )
        return event_id

    now = now_iso()
    conn.execute(
        """
        INSERT INTO events(
            sport, title, participant_a, participant_b, event_datetime,
            odds_a, odds_b, predicted_prob_a, predicted_prob_b,
            stats_a, stats_b, status, winner_side, created_by, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'upcoming', NULL, ?, ?, ?)
        """,
        (
            sport,
            title,
            participant_a,
            participant_b,
            event_datetime,
            odds_a,
            odds_b,
            predicted_prob_a,
            predicted_prob_b,
            stats_a,
            stats_b,
            created_by,
            now,
            now,
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])


def normalize_result(raw: object) -> Optional[str]:
    value = normalize_text(raw)
    if value in {"win", "won", "gagne"}:
        return "win"
    if value in {"loss", "lose", "perdu"}:
        return "loss"
    if value in {"void", "annule", "cancel"}:
        return "void"
    return None


def import_tennis_bets(conn: sqlite3.Connection, admin_id: int) -> int:
    if not LEGACY_TENNIS_BETS.exists():
        return 0

    df = pd.read_csv(LEGACY_TENNIS_BETS)
    imported = 0

    for row in df.to_dict(orient="records"):
        participant_a = str(row.get("player_1", "")).strip()
        participant_b = str(row.get("player_2", "")).strip()
        if not participant_a or not participant_b:
            continue

        tournament = str(row.get("tournament", "")).strip() or "Tennis"
        round_name = str(row.get("round", "")).strip()
        title = tournament if not round_name else f"{tournament} - {round_name}"
        event_dt = parse_datetime_str(row.get("date"))
        pick_label = str(row.get("pick", "")).strip()
        pick_side = detect_pick_side(pick_label, participant_a, participant_b) or "a"

        odds = to_float(row.get("odds"), 1.01) or 1.01
        stake = to_float(row.get("stake"), 0.0) or 0.0
        if stake <= 0:
            continue

        result = normalize_result(row.get("result"))
        status_raw = normalize_text(row.get("status"))
        status = "open" if status_raw == "open" else "resolved"
        if status == "open":
            result = None

        profit = to_float(row.get("profit"))
        if status == "resolved" and profit is None:
            if result == "win":
                profit = stake * (odds - 1.0)
            elif result == "loss":
                profit = -stake
            else:
                profit = 0.0

        odds_a = odds if pick_side == "a" else None
        odds_b = odds if pick_side == "b" else None

        event_id = get_or_create_event(
            conn,
            sport="tennis",
            title=title,
            participant_a=participant_a,
            participant_b=participant_b,
            event_datetime=event_dt,
            created_by=admin_id,
            odds_a=odds_a,
            odds_b=odds_b,
        )

        model_probability = to_float(row.get("model_prob"))
        edge = to_float(row.get("edge"))
        ev = to_float(row.get("ev"))

        conn.execute(
            """
            INSERT INTO bets(
                user_id, sport, event_id, pick_side, odds, stake,
                model_probability, edge, ev, status, result, profit,
                source, notes, created_at, resolved_at
            )
            VALUES (?, 'tennis', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'import_tennis', ?, ?, ?)
            """,
            (
                admin_id,
                event_id,
                pick_side,
                odds,
                stake,
                model_probability,
                edge,
                ev,
                status,
                result,
                profit,
                f"legacy_bet_id={row.get('bet_id')}|pick={pick_label}",
                event_dt,
                event_dt if status == "resolved" else None,
            ),
        )
        imported += 1

    return imported


def import_ufc_bets(conn: sqlite3.Connection, admin_id: int) -> int:
    if not LEGACY_UFC_BETS.exists():
        return 0

    df = pd.read_csv(LEGACY_UFC_BETS)
    imported = 0

    for row in df.to_dict(orient="records"):
        participant_a = str(row.get("fighter_red", "")).strip()
        participant_b = str(row.get("fighter_blue", "")).strip()
        if not participant_a or not participant_b:
            continue

        title = str(row.get("event", "")).strip() or "UFC"
        event_dt = parse_datetime_str(row.get("date"))
        pick_label = str(row.get("pick", "")).strip()
        pick_side = detect_pick_side(pick_label, participant_a, participant_b) or "a"

        odds = to_float(row.get("odds"), 1.01) or 1.01
        stake = to_float(row.get("stake"), 0.0) or 0.0
        if stake <= 0:
            continue

        result = normalize_result(row.get("result"))
        status_raw = normalize_text(row.get("status"))
        status = "open" if status_raw == "open" else "resolved"
        if status == "open":
            result = None

        profit = to_float(row.get("profit"))
        if status == "resolved" and profit is None:
            if result == "win":
                profit = stake * (odds - 1.0)
            elif result == "loss":
                profit = -stake
            else:
                profit = 0.0

        odds_a = odds if pick_side == "a" else None
        odds_b = odds if pick_side == "b" else None

        event_id = get_or_create_event(
            conn,
            sport="ufc",
            title=title,
            participant_a=participant_a,
            participant_b=participant_b,
            event_datetime=event_dt,
            created_by=admin_id,
            odds_a=odds_a,
            odds_b=odds_b,
        )

        model_probability = to_float(row.get("model_probability"))
        edge = to_float(row.get("edge"))
        ev = to_float(row.get("ev"))

        conn.execute(
            """
            INSERT INTO bets(
                user_id, sport, event_id, pick_side, odds, stake,
                model_probability, edge, ev, status, result, profit,
                source, notes, created_at, resolved_at
            )
            VALUES (?, 'ufc', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'import_ufc', ?, ?, ?)
            """,
            (
                admin_id,
                event_id,
                pick_side,
                odds,
                stake,
                model_probability,
                edge,
                ev,
                status,
                result,
                profit,
                f"legacy_bet_id={row.get('bet_id')}|pick={pick_label}",
                event_dt,
                event_dt if status == "resolved" else None,
            ),
        )
        imported += 1

    return imported


def refresh_event_status_from_bets(conn: sqlite3.Connection) -> None:
    event_ids = conn.execute("SELECT id FROM events").fetchall()
    for event_row in event_ids:
        event_id = int(event_row["id"])
        bets = conn.execute(
            "SELECT pick_side, status, result FROM bets WHERE event_id = ?",
            (event_id,),
        ).fetchall()
        if not bets:
            continue

        has_open = any(b["status"] == "open" for b in bets)
        if has_open:
            conn.execute(
                "UPDATE events SET status = 'upcoming', winner_side = NULL, updated_at = ? WHERE id = ?",
                (now_iso(), event_id),
            )
            continue

        winners = set()
        only_void = True
        for bet in bets:
            result = bet["result"]
            pick = bet["pick_side"]
            if result == "win":
                winners.add(pick)
                only_void = False
            elif result == "loss":
                winners.add("b" if pick == "a" else "a")
                only_void = False
            elif result == "void":
                continue
            else:
                only_void = False

        winner_side: Optional[str]
        if only_void:
            winner_side = "void"
        elif len(winners) == 1:
            winner_side = next(iter(winners))
        else:
            winner_side = None

        conn.execute(
            "UPDATE events SET status = 'completed', winner_side = ?, updated_at = ? WHERE id = ?",
            (winner_side, now_iso(), event_id),
        )


def maybe_migrate_legacy_data(conn: sqlite3.Connection, admin_id: int) -> None:
    if get_meta(conn, "legacy_migration_v1") == "done":
        return

    tennis_count = import_tennis_bets(conn, admin_id)
    ufc_count = import_ufc_bets(conn, admin_id)

    # Bankroll initiale admin combinee depuis les deux apps legacy
    combined_initial = infer_combined_admin_initial_bankroll()
    conn.execute(
        "UPDATE users SET initial_bankroll = ? WHERE id = ?",
        (combined_initial, admin_id),
    )

    refresh_event_status_from_bets(conn)
    recompute_user_bankroll(conn, admin_id)

    summary = json.dumps(
        {
            "migrated_at": now_iso(),
            "tennis_bets": tennis_count,
            "ufc_bets": ufc_count,
            "admin_initial_bankroll": combined_initial,
        }
    )
    set_meta(conn, "legacy_migration_v1", "done")
    set_meta(conn, "legacy_migration_v1_summary", summary)


def validate_username(username: str) -> Optional[str]:
    value = username.strip()
    if len(value) < 3 or len(value) > 30:
        return "Le pseudo doit contenir entre 3 et 30 caracteres."
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", value):
        return "Le pseudo doit utiliser uniquement lettres, chiffres, ., _ ou -."
    return None


def create_user(
    username: str,
    password: str,
    initial_bankroll: Optional[float] = None,
) -> tuple[bool, str]:
    username = username.strip()
    user_error = validate_username(username)
    if user_error:
        return False, user_error
    if len(password) < 8:
        return False, "Le mot de passe doit contenir au moins 8 caracteres."
    initial_value = float(
        DEFAULT_INITIAL_BANKROLL if initial_bankroll is None else max(0.0, float(initial_bankroll))
    )
    if username == ADMIN_USERNAME and get_user_by_username(username) is not None:
        return False, "Le compte administrateur existe deja."

    with db_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if exists:
            return False, "Ce pseudo existe deja."

        now = now_iso()
        pwd_hash = hash_password(password)
        conn.execute(
            """
            INSERT INTO users(username, password_hash, is_admin, initial_bankroll, current_bankroll, created_at)
            VALUES (?, ?, 0, ?, ?, ?)
            """,
            (username, pwd_hash, initial_value, initial_value, now),
        )
        conn.commit()
    ensure_user_legacy_files(username, is_admin=False, initial_bankroll=initial_value)
    return True, "Compte cree. Vous pouvez maintenant vous connecter."


def authenticate(username: str, password: str) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if not user:
            return None
        if verify_password(password, user["password_hash"]):
            ensure_user_legacy_files(str(user["username"]), is_admin=int(user["is_admin"]) == 1)
            return user
        return None


def current_user() -> Optional[sqlite3.Row]:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    user = get_user_by_id(int(user_id))
    if user:
        ensure_user_legacy_files(str(user["username"]), is_admin=int(user["is_admin"]) == 1)
    return user


def fetch_events(sport: str, include_completed: bool = True) -> list[dict]:
    with db_conn() as conn:
        query = """
            SELECT *
            FROM events
            WHERE sport = ?
        """
        params: list[object] = [sport]
        if not include_completed:
            query += " AND status = 'upcoming'"
        query += """
            ORDER BY
                CASE status WHEN 'upcoming' THEN 0 ELSE 1 END,
                COALESCE(event_datetime, '') ASC,
                id DESC
        """
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def create_event(
    *,
    sport: str,
    title: str,
    participant_a: str,
    participant_b: str,
    event_datetime: str,
    odds_a: Optional[float],
    odds_b: Optional[float],
    predicted_prob_a: Optional[float],
    predicted_prob_b: Optional[float],
    stats_a: str,
    stats_b: str,
    created_by: int,
) -> tuple[bool, str]:
    participant_a = participant_a.strip()
    participant_b = participant_b.strip()
    title = title.strip()
    if not participant_a or not participant_b:
        return False, "Les deux participants sont obligatoires."
    if normalize_text(participant_a) == normalize_text(participant_b):
        return False, "Les deux participants doivent etre differents."
    if odds_a is None or odds_b is None or odds_a <= 1.0 or odds_b <= 1.0:
        return False, "Les cotes doivent etre > 1.0."
    if predicted_prob_a is not None and not (0 <= predicted_prob_a <= 1):
        return False, "Probabilite A invalide."
    if predicted_prob_b is not None and not (0 <= predicted_prob_b <= 1):
        return False, "Probabilite B invalide."

    with db_conn() as conn:
        get_or_create_event(
            conn,
            sport=sport,
            title=title,
            participant_a=participant_a,
            participant_b=participant_b,
            event_datetime=event_datetime,
            created_by=created_by,
            odds_a=odds_a,
            odds_b=odds_b,
            predicted_prob_a=predicted_prob_a,
            predicted_prob_b=predicted_prob_b,
            stats_a=stats_a.strip(),
            stats_b=stats_b.strip(),
        )
        conn.commit()
    return True, "Evenement ajoute."


def update_event(
    event_id: int,
    *,
    title: str,
    participant_a: str,
    participant_b: str,
    event_datetime: str,
    odds_a: Optional[float],
    odds_b: Optional[float],
    predicted_prob_a: Optional[float],
    predicted_prob_b: Optional[float],
    stats_a: str,
    stats_b: str,
) -> tuple[bool, str]:
    participant_a = participant_a.strip()
    participant_b = participant_b.strip()
    title = title.strip()
    if not participant_a or not participant_b:
        return False, "Les deux participants sont obligatoires."
    if normalize_text(participant_a) == normalize_text(participant_b):
        return False, "Les deux participants doivent etre differents."
    if odds_a is None or odds_b is None or odds_a <= 1.0 or odds_b <= 1.0:
        return False, "Les cotes doivent etre > 1.0."

    with db_conn() as conn:
        row = conn.execute("SELECT id, status FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return False, "Evenement introuvable."
        if row["status"] != "upcoming":
            return False, "Seuls les evenements a venir peuvent etre modifies."

        conn.execute(
            """
            UPDATE events
            SET title = ?, participant_a = ?, participant_b = ?, event_datetime = ?,
                odds_a = ?, odds_b = ?, predicted_prob_a = ?, predicted_prob_b = ?,
                stats_a = ?, stats_b = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                participant_a,
                participant_b,
                event_datetime,
                odds_a,
                odds_b,
                predicted_prob_a,
                predicted_prob_b,
                stats_a.strip(),
                stats_b.strip(),
                now_iso(),
                event_id,
            ),
        )
        conn.commit()
    return True, "Evenement mis a jour."


def delete_event(event_id: int) -> tuple[bool, str]:
    with db_conn() as conn:
        row = conn.execute("SELECT status FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return False, "Evenement introuvable."
        if row["status"] != "upcoming":
            return False, "Seuls les evenements a venir peuvent etre supprimes."

        linked_bets = conn.execute(
            "SELECT COUNT(*) AS n FROM bets WHERE event_id = ?",
            (event_id,),
        ).fetchone()["n"]
        if int(linked_bets) > 0:
            return False, "Impossible de supprimer: des paris existent deja sur cet evenement."

        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
    return True, "Evenement supprime."


def place_bet(user_id: int, event_id: int, pick_side: str, stake: float) -> tuple[bool, str]:
    if stake <= 0:
        return False, "La mise doit etre positive."

    with db_conn() as conn:
        user = conn.execute(
            "SELECT id, current_bankroll FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not user:
            return False, "Utilisateur introuvable."

        bankroll = float(user["current_bankroll"])
        if stake > bankroll:
            return False, "Bankroll insuffisante."

        event = conn.execute(
            "SELECT * FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not event:
            return False, "Evenement introuvable."
        if event["status"] != "upcoming":
            return False, "Les paris sont fermes pour cet evenement."

        odds = to_float(event["odds_a"] if pick_side == "a" else event["odds_b"])
        if odds is None or odds <= 1.0:
            return False, "Cote invalide pour ce choix."
        odds = float(odds)

        model_probability = event["predicted_prob_a"] if pick_side == "a" else event["predicted_prob_b"]
        edge = None
        ev = None
        if model_probability is not None:
            model_probability = float(model_probability)
            edge = model_probability - (1.0 / odds)
            ev = model_probability * odds - 1.0

        conn.execute(
            """
            INSERT INTO bets(
                user_id, sport, event_id, pick_side, odds, stake,
                model_probability, edge, ev, status, result, profit,
                source, notes, created_at, resolved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, 'manual', NULL, ?, NULL)
            """,
            (
                user_id,
                event["sport"],
                event_id,
                pick_side,
                odds,
                stake,
                model_probability,
                edge,
                ev,
                now_iso(),
            ),
        )
        recompute_user_bankroll(conn, user_id)
        conn.commit()

    return True, "Pari enregistre."


def apply_event_result(event_id: int, winner_side: Optional[str]) -> tuple[bool, str]:
    if winner_side not in {"a", "b", "void", None}:
        return False, "Resultat invalide."

    with db_conn() as conn:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if not event:
            return False, "Evenement introuvable."

        bets = conn.execute(
            "SELECT id, user_id, pick_side, stake, odds FROM bets WHERE event_id = ?",
            (event_id,),
        ).fetchall()

        impacted_users: set[int] = set()
        for bet in bets:
            bet_id = int(bet["id"])
            user_id = int(bet["user_id"])
            impacted_users.add(user_id)
            stake = float(bet["stake"])
            odds = float(bet["odds"])
            pick_side = bet["pick_side"]

            if winner_side is None:
                conn.execute(
                    """
                    UPDATE bets
                    SET status = 'open', result = NULL, profit = NULL, resolved_at = NULL
                    WHERE id = ?
                    """,
                    (bet_id,),
                )
                continue

            if winner_side == "void":
                result = "void"
                profit = 0.0
            elif pick_side == winner_side:
                result = "win"
                profit = stake * (odds - 1.0)
            else:
                result = "loss"
                profit = -stake

            conn.execute(
                """
                UPDATE bets
                SET status = 'resolved', result = ?, profit = ?, resolved_at = ?
                WHERE id = ?
                """,
                (result, profit, now_iso(), bet_id),
            )

        event_status = "upcoming" if winner_side is None else "completed"
        conn.execute(
            """
            UPDATE events
            SET status = ?, winner_side = ?, updated_at = ?
            WHERE id = ?
            """,
            (event_status, winner_side, now_iso(), event_id),
        )

        for user_id in impacted_users:
            recompute_user_bankroll(conn, user_id)
        conn.commit()

    return True, f"Resultat applique ({len(bets)} paris mis a jour)."


def fetch_user_bets(user_id: int, sport: Optional[str] = None) -> pd.DataFrame:
    with db_conn() as conn:
        query = """
            SELECT
                b.id AS bet_id,
                b.created_at,
                b.sport,
                e.title,
                e.participant_a,
                e.participant_b,
                b.pick_side,
                b.odds,
                b.stake,
                b.model_probability,
                b.edge,
                b.ev,
                b.status,
                b.result,
                b.profit
            FROM bets b
            JOIN events e ON e.id = b.event_id
            WHERE b.user_id = ?
        """
        params: list[object] = [user_id]
        if sport:
            query += " AND b.sport = ?"
            params.append(sport)
        query += " ORDER BY b.created_at DESC, b.id DESC"

        rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["pick"] = df.apply(
        lambda r: r["participant_a"] if r["pick_side"] == "a" else r["participant_b"],
        axis=1,
    )
    df["match_combat"] = df.apply(
        lambda r: f"{r['participant_a']} vs {r['participant_b']}",
        axis=1,
    )
    cols = [
        "bet_id",
        "created_at",
        "sport",
        "title",
        "match_combat",
        "pick",
        "odds",
        "stake",
        "model_probability",
        "edge",
        "ev",
        "status",
        "result",
        "profit",
    ]
    return df[cols]


def fetch_user_stats(user_id: int) -> dict[str, float]:
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_bets,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_bets,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN result = 'void' THEN 1 ELSE 0 END) AS voids,
                COALESCE(SUM(CASE WHEN status = 'resolved' THEN profit ELSE 0 END), 0.0) AS profit_total,
                COALESCE(SUM(CASE WHEN status = 'resolved' AND result != 'void' THEN stake ELSE 0 END), 0.0) AS resolved_stake
            FROM bets
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    data = dict(row)
    resolved = float(data.get("resolved_bets") or 0)
    wins = float(data.get("wins") or 0)
    resolved_stake = float(data.get("resolved_stake") or 0.0)
    profit_total = float(data.get("profit_total") or 0.0)
    data["win_rate"] = (wins / resolved * 100.0) if resolved > 0 else 0.0
    data["roi"] = (profit_total / resolved_stake * 100.0) if resolved_stake > 0 else 0.0
    return data


def admin_events_for_select(sport: str, include_completed: bool = True) -> list[dict]:
    return fetch_events(sport=sport, include_completed=include_completed)


def event_option_label(event: dict) -> str:
    title = event["title"] or ("Tennis" if event["sport"] == "tennis" else "UFC")
    return (
        f"#{event['id']} - {title} - {event['participant_a']} vs {event['participant_b']} "
        f"- {format_datetime(event['event_datetime'])} [{event['status']}]"
    )


def split_datetime_for_form(value: Optional[str]) -> tuple[date, time]:
    if value:
        try:
            dt = datetime.fromisoformat(value)
            return dt.date(), dt.time().replace(microsecond=0)
        except ValueError:
            pass
    now = datetime.now().replace(second=0, microsecond=0)
    return now.date(), now.time()


def render_auth_page() -> None:
    st.title(APP_TITLE)
    st.markdown("Connexion utilisateur requise.")

    login_tab, register_tab = st.tabs(["Connexion", "Creer un compte"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Pseudo", key="login_username")
            password = st.text_input("Mot de passe", type="password", key="login_password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
        if submit:
            user = authenticate(username, password)
            if user:
                st.session_state.user_id = int(user["id"])
                st.session_state.section = "Accueil"
                st.rerun()
            st.error("Pseudo ou mot de passe incorrect.")

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            username = st.text_input("Pseudo", key="register_username")
            password = st.text_input("Mot de passe", type="password", key="register_password")
            password_confirm = st.text_input("Confirmer le mot de passe", type="password", key="register_password_confirm")
            initial_bankroll = st.number_input(
                "Bankroll initiale (appliquee a Tennis et UFC)",
                min_value=0.0,
                value=float(DEFAULT_INITIAL_BANKROLL),
                step=50.0,
                key="register_initial_bankroll",
            )
            submit = st.form_submit_button("Creer mon compte", use_container_width=True)
        if submit:
            if password != password_confirm:
                st.error("Les mots de passe ne correspondent pas.")
            else:
                ok, message = create_user(username, password, initial_bankroll=float(initial_bankroll))
                if ok:
                    st.success(message)
                else:
                    st.error(message)

    st.info(
        f"Compte admin disponible: `{ADMIN_USERNAME}`. "
        "Le mot de passe initial peut etre configure via `NARVALL018_DEFAULT_PASSWORD`."
    )


def render_home(user: sqlite3.Row) -> None:
    st.title("Accueil")
    st.markdown("Selectionnez une section.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Section Tennis")
        st.write("Matchs a venir, paris, historique tennis.")
        if st.button("Ouvrir Tennis", use_container_width=True):
            st.session_state.section = "Tennis"
            st.rerun()
    with c2:
        st.subheader("Section UFC")
        st.write("Combats a venir, stats combattants, paris UFC.")
        if st.button("Ouvrir UFC", use_container_width=True):
            st.session_state.section = "UFC"
            st.rerun()

    stats = fetch_user_stats(int(user["id"]))
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Paris total", int(stats["total_bets"] or 0))
    k2.metric("Paris resolus", int(stats["resolved_bets"] or 0))
    k3.metric("Profit total", f"{float(stats['profit_total']):+.2f} EUR")
    k4.metric("ROI", f"{float(stats['roi']):.1f}%")


def render_events_list_for_sport(user: sqlite3.Row, sport: str) -> None:
    sport_label = "Tennis" if sport == "tennis" else "UFC"
    st.title(f"Section {sport_label}")

    events = fetch_events(sport=sport, include_completed=True)
    upcoming = [e for e in events if e["status"] == "upcoming"]
    completed = [e for e in events if e["status"] == "completed"]

    st.caption(f"Evenements a venir: {len(upcoming)} | Evenements termines: {len(completed)}")
    if not upcoming:
        st.info("Aucun evenement a venir. Ajoutez des evenements depuis le panneau admin.")

    bankroll = float(user["current_bankroll"])
    for event in upcoming:
        with st.container(border=True):
            header_title = event["title"] or sport_label
            st.markdown(f"### {header_title}")
            st.write(f"**{event['participant_a']}** vs **{event['participant_b']}**")
            st.caption(f"Date: {format_datetime(event['event_datetime'])}")

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"Cote {event['participant_a']}: {float(event['odds_a'] or 0):.2f}")
                prob_a = event["predicted_prob_a"]
                if prob_a is not None and float(prob_a) > 0:
                    pred_odd = 1 / float(prob_a)
                    st.caption(f"Cote predite: {pred_odd:.2f} ({float(prob_a):.1%})")
            with c2:
                st.write(f"Cote {event['participant_b']}: {float(event['odds_b'] or 0):.2f}")
                prob_b = event["predicted_prob_b"]
                if prob_b is not None and float(prob_b) > 0:
                    pred_odd = 1 / float(prob_b)
                    st.caption(f"Cote predite: {pred_odd:.2f} ({float(prob_b):.1%})")

            if sport == "ufc":
                stats_a = (event["stats_a"] or "").strip()
                stats_b = (event["stats_b"] or "").strip()
                if stats_a or stats_b:
                    s1, s2 = st.columns(2)
                    with s1:
                        st.markdown(f"**Stats {event['participant_a']}**")
                        st.caption(stats_a if stats_a else "Aucune statistique")
                    with s2:
                        st.markdown(f"**Stats {event['participant_b']}**")
                        st.caption(stats_b if stats_b else "Aucune statistique")

            if bankroll < 1.0:
                st.warning("Bankroll insuffisante pour placer un pari.")
                continue

            with st.form(f"bet_form_{sport}_{event['id']}"):
                pick = st.radio(
                    "Selection du vainqueur",
                    options=[
                        (event["participant_a"], "a"),
                        (event["participant_b"], "b"),
                    ],
                    format_func=lambda x: x[0],
                    horizontal=True,
                    key=f"pick_{sport}_{event['id']}",
                )
                max_value = max(1.0, bankroll)
                stake = st.number_input(
                    "Mise (EUR)",
                    min_value=1.0,
                    max_value=float(max_value),
                    value=min(10.0, float(max_value)),
                    step=1.0,
                    key=f"stake_{sport}_{event['id']}",
                )
                submit = st.form_submit_button("Placer le pari", use_container_width=True)

            if submit:
                ok, msg = place_bet(int(user["id"]), int(event["id"]), pick[1], float(stake))
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with st.expander("Historique des evenements resolus"):
        if not completed:
            st.write("Aucun evenement resolu.")
        for event in completed:
            winner = event["winner_side"]
            if winner == "a":
                winner_label = event["participant_a"]
            elif winner == "b":
                winner_label = event["participant_b"]
            elif winner == "void":
                winner_label = "Annule / void"
            else:
                winner_label = "Non renseigne"
            st.write(
                f"#{event['id']} - {event['title'] or sport_label}: "
                f"{event['participant_a']} vs {event['participant_b']} "
                f"| Vainqueur: {winner_label}"
            )

    st.markdown("---")
    st.subheader("Mes paris sur cette section")
    user_bets = fetch_user_bets(int(user["id"]), sport=sport)
    if user_bets.empty:
        st.info("Aucun pari enregistre pour cette section.")
    else:
        st.dataframe(user_bets, use_container_width=True, hide_index=True)


def render_my_account(user: sqlite3.Row) -> None:
    st.title("Mon compte")

    stats = fetch_user_stats(int(user["id"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bankroll", f"{float(user['current_bankroll']):.2f} EUR")
    c2.metric("Profit total", f"{float(stats['profit_total']):+.2f} EUR")
    c3.metric("Taux de reussite", f"{float(stats['win_rate']):.1f}%")
    c4.metric("ROI", f"{float(stats['roi']):.1f}%")

    st.markdown(
        f"Bankroll initiale: **{float(user['initial_bankroll']):.2f} EUR** | "
        f"Paris ouverts: **{int(stats['open_bets'] or 0)}**"
    )

    df = fetch_user_bets(int(user["id"]))
    if df.empty:
        st.info("Aucun pari enregistre.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_admin_events_editor(user: sqlite3.Row, sport: str) -> None:
    sport_label = "Tennis" if sport == "tennis" else "UFC"
    st.subheader(f"Gestion des evenements {sport_label}")

    with st.expander(f"Ajouter un evenement {sport_label}", expanded=False):
        with st.form(f"add_event_{sport}"):
            title = st.text_input("Tournoi / Event", key=f"add_title_{sport}")
            p1_label = "Joueur 1" if sport == "tennis" else "Combattant 1"
            p2_label = "Joueur 2" if sport == "tennis" else "Combattant 2"
            participant_a = st.text_input(p1_label, key=f"add_p1_{sport}")
            participant_b = st.text_input(p2_label, key=f"add_p2_{sport}")
            c1, c2 = st.columns(2)
            with c1:
                d = st.date_input("Date", key=f"add_date_{sport}")
            with c2:
                t = st.time_input("Heure", value=time(20, 0), key=f"add_time_{sport}")
            o1, o2 = st.columns(2)
            with o1:
                odds_a = st.number_input("Cote participant 1", min_value=1.01, value=1.80, step=0.01, key=f"add_odds_a_{sport}")
            with o2:
                odds_b = st.number_input("Cote participant 2", min_value=1.01, value=1.80, step=0.01, key=f"add_odds_b_{sport}")
            p1, p2 = st.columns(2)
            with p1:
                predicted_prob_a = st.number_input(
                    "Probabilite predite participant 1",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.01,
                    key=f"add_prob_a_{sport}",
                )
            with p2:
                predicted_prob_b = st.number_input(
                    "Probabilite predite participant 2",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.01,
                    key=f"add_prob_b_{sport}",
                )
            stats_a = ""
            stats_b = ""
            if sport == "ufc":
                s1, s2 = st.columns(2)
                with s1:
                    stats_a = st.text_area("Stats combattant 1", key=f"add_stats_a_{sport}")
                with s2:
                    stats_b = st.text_area("Stats combattant 2", key=f"add_stats_b_{sport}")
            submit = st.form_submit_button("Ajouter", use_container_width=True)
        if submit:
            event_dt = datetime.combine(d, t).isoformat(timespec="seconds")
            ok, msg = create_event(
                sport=sport,
                title=title,
                participant_a=participant_a,
                participant_b=participant_b,
                event_datetime=event_dt,
                odds_a=float(odds_a),
                odds_b=float(odds_b),
                predicted_prob_a=float(predicted_prob_a),
                predicted_prob_b=float(predicted_prob_b),
                stats_a=stats_a,
                stats_b=stats_b,
                created_by=int(user["id"]),
            )
            if ok:
                st.success(msg)
                st.rerun()
            st.error(msg)

    upcoming_events = [e for e in admin_events_for_select(sport, include_completed=False) if e["status"] == "upcoming"]
    if not upcoming_events:
        st.info("Aucun evenement a venir a modifier.")
        return

    selected = st.selectbox(
        "Evenement a modifier",
        options=upcoming_events,
        format_func=event_option_label,
        key=f"edit_select_{sport}",
    )
    date_default, time_default = split_datetime_for_form(selected["event_datetime"])

    with st.form(f"edit_event_form_{sport}"):
        title = st.text_input("Tournoi / Event", value=selected["title"] or "", key=f"edit_title_{sport}")
        p1_label = "Joueur 1" if sport == "tennis" else "Combattant 1"
        p2_label = "Joueur 2" if sport == "tennis" else "Combattant 2"
        participant_a = st.text_input(p1_label, value=selected["participant_a"], key=f"edit_p1_{sport}")
        participant_b = st.text_input(p2_label, value=selected["participant_b"], key=f"edit_p2_{sport}")
        c1, c2 = st.columns(2)
        with c1:
            d = st.date_input("Date", value=date_default, key=f"edit_date_{sport}")
        with c2:
            t = st.time_input("Heure", value=time_default, key=f"edit_time_{sport}")
        o1, o2 = st.columns(2)
        with o1:
            odds_a = st.number_input(
                "Cote participant 1",
                min_value=1.01,
                value=float(selected["odds_a"] or 1.8),
                step=0.01,
                key=f"edit_odds_a_{sport}",
            )
        with o2:
            odds_b = st.number_input(
                "Cote participant 2",
                min_value=1.01,
                value=float(selected["odds_b"] or 1.8),
                step=0.01,
                key=f"edit_odds_b_{sport}",
            )
        pr1, pr2 = st.columns(2)
        with pr1:
            predicted_prob_a = st.number_input(
                "Probabilite predite participant 1",
                min_value=0.0,
                max_value=1.0,
                value=float(selected["predicted_prob_a"] if selected["predicted_prob_a"] is not None else 0.5),
                step=0.01,
                key=f"edit_prob_a_{sport}",
            )
        with pr2:
            predicted_prob_b = st.number_input(
                "Probabilite predite participant 2",
                min_value=0.0,
                max_value=1.0,
                value=float(selected["predicted_prob_b"] if selected["predicted_prob_b"] is not None else 0.5),
                step=0.01,
                key=f"edit_prob_b_{sport}",
            )
        stats_a = ""
        stats_b = ""
        if sport == "ufc":
            s1, s2 = st.columns(2)
            with s1:
                stats_a = st.text_area("Stats combattant 1", value=selected["stats_a"] or "", key=f"edit_stats_a_{sport}")
            with s2:
                stats_b = st.text_area("Stats combattant 2", value=selected["stats_b"] or "", key=f"edit_stats_b_{sport}")
        submit_update = st.form_submit_button("Enregistrer les modifications", use_container_width=True)

    if submit_update:
        event_dt = datetime.combine(d, t).isoformat(timespec="seconds")
        ok, msg = update_event(
            int(selected["id"]),
            title=title,
            participant_a=participant_a,
            participant_b=participant_b,
            event_datetime=event_dt,
            odds_a=float(odds_a),
            odds_b=float(odds_b),
            predicted_prob_a=float(predicted_prob_a),
            predicted_prob_b=float(predicted_prob_b),
            stats_a=stats_a,
            stats_b=stats_b,
        )
        if ok:
            st.success(msg)
            st.rerun()
        st.error(msg)

    with st.form(f"delete_event_form_{sport}"):
        confirm = st.checkbox(
            "Je confirme la suppression de cet evenement",
            key=f"confirm_delete_{sport}",
        )
        submit_delete = st.form_submit_button("Supprimer l'evenement", use_container_width=True)
    if submit_delete:
        if not confirm:
            st.error("Veuillez confirmer la suppression.")
        else:
            ok, msg = delete_event(int(selected["id"]))
            if ok:
                st.success(msg)
                st.rerun()
            st.error(msg)


def render_admin_results_panel() -> None:
    st.subheader("Mise a jour des resultats")

    sport = st.selectbox(
        "Sport",
        options=["tennis", "ufc"],
        format_func=lambda s: "Tennis" if s == "tennis" else "UFC",
        key="admin_result_sport",
    )

    events = admin_events_for_select(sport, include_completed=True)
    if not events:
        st.info("Aucun evenement disponible.")
        return

    event = st.selectbox(
        "Evenement",
        options=events,
        format_func=event_option_label,
        key="admin_result_event",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        open_bets = 0
        resolved_bets = 0
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_n,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_n
                FROM bets
                WHERE event_id = ?
                """,
                (int(event["id"]),),
            ).fetchone()
            open_bets = int(row["open_n"] or 0)
            resolved_bets = int(row["resolved_n"] or 0)
        st.metric("Paris ouverts", open_bets)
    with c2:
        st.metric("Paris resolus", resolved_bets)
    with c3:
        st.metric("Statut event", event["status"])

    options = {
        f"Vainqueur: {event['participant_a']}": "a",
        f"Vainqueur: {event['participant_b']}": "b",
        "Annule / Void (remboursement)": "void",
        "Aucun resultat (reouvrir les paris)": None,
    }
    selected_label = st.radio("Resultat officiel", list(options.keys()), key="admin_result_choice")

    if st.button("Appliquer le resultat", type="primary", use_container_width=True):
        ok, msg = apply_event_result(int(event["id"]), options[selected_label])
        if ok:
            st.success(msg)
            st.rerun()
        st.error(msg)


def render_admin_panel(user: sqlite3.Row) -> None:
    st.title("Panneau d'administration")
    st.caption("Acces reserve a narvall018.")

    tab_tennis, tab_ufc, tab_results = st.tabs(
        ["Evenements Tennis", "Evenements UFC", "Resultats & Resolution"]
    )

    with tab_tennis:
        render_admin_events_editor(user, "tennis")
    with tab_ufc:
        render_admin_events_editor(user, "ufc")
    with tab_results:
        render_admin_results_panel()


def _set_unified_session_context(user: sqlite3.Row) -> None:
    username = str(user["username"])
    st.session_state["unified_mode"] = True
    st.session_state["unified_username"] = username


def _read_tennis_bankroll(username: str) -> float:
    user_dir = tennis_user_dir(username)
    bankroll_file = user_dir / "bankroll.json"
    bets_file = user_dir / "bets.csv"
    initial = float(DEFAULT_INITIAL_BANKROLL)

    if bankroll_file.exists():
        try:
            data = json.loads(bankroll_file.read_text(encoding="utf-8"))
            initial = float(data.get("initial_bankroll", initial))
        except Exception:
            pass

    if not bets_file.exists():
        return round(initial, 2)
    try:
        df = pd.read_csv(bets_file)
    except Exception:
        return round(initial, 2)
    if df.empty:
        return round(initial, 2)

    status_col = df.get("status")
    if status_col is None:
        return round(initial, 2)
    status = status_col.astype(str).str.lower()
    closed_profit = pd.to_numeric(df.loc[status == "closed", "profit"], errors="coerce").fillna(0).sum()
    open_stake = pd.to_numeric(df.loc[status == "open", "stake"], errors="coerce").fillna(0).sum()
    return round(float(initial) + float(closed_profit) - float(open_stake), 2)


def _read_ufc_bankroll(username: str) -> float:
    bankroll_file = ufc_user_dir(username) / "bankroll.csv"
    if not bankroll_file.exists():
        return round(float(DEFAULT_INITIAL_BANKROLL), 2)
    try:
        df = pd.read_csv(bankroll_file)
        if df.empty:
            return round(float(DEFAULT_INITIAL_BANKROLL), 2)
        amount = pd.to_numeric(df["amount"], errors="coerce").dropna()
        if amount.empty:
            return round(float(DEFAULT_INITIAL_BANKROLL), 2)
        return round(float(amount.iloc[-1]), 2)
    except Exception:
        return round(float(DEFAULT_INITIAL_BANKROLL), 2)


def _load_tennis_module():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "main") and hasattr(main_mod, "show_home_page"):
        return main_mod
    os.environ["TENNIS_EMBED_MODE"] = "1"
    return importlib.import_module("app")


def _load_ufc_module():
    os.environ["UFC_EMBED_MODE"] = "1"
    return importlib.import_module("predictor_ufc.app")


def _render_legacy_tennis(user: sqlite3.Row) -> None:
    _set_unified_session_context(user)
    tennis_app = _load_tennis_module()
    tennis_app.main()


def _render_legacy_ufc(user: sqlite3.Row) -> None:
    _set_unified_session_context(user)
    username = str(user["username"])
    is_admin = int(user["is_admin"]) == 1
    ufc_bets_folder = ufc_user_dir(username)
    st.session_state["unified_ufc_bets_folder"] = str(ufc_bets_folder)

    ufc_app = _load_ufc_module()
    ufc_app.USER_PROFILES[username] = {
        "password_hash": str(user["password_hash"]),
        "display_name": f"👤 {username}",
        "is_admin": is_admin,
        "bets_folder": str(ufc_bets_folder),
        "can_bet": True,
        "can_view_bankroll": True,
    }
    st.session_state["logged_in_user"] = username

    def _logout_bridge():
        st.session_state.pop("logged_in_user", None)
        st.session_state.pop("user_id", None)
        st.session_state.pop("unified_ufc_bets_folder", None)
        st.session_state["_goto_section"] = "Accueil"

    ufc_app.logout_user = _logout_bridge
    ufc_app.main()


def _render_unified_home(user: sqlite3.Row) -> None:
    username = str(user["username"])
    ensure_user_legacy_files(username, is_admin=int(user["is_admin"]) == 1)
    bankroll_tennis = _read_tennis_bankroll(username)
    bankroll_ufc = _read_ufc_bankroll(username)

    st.title(APP_TITLE)
    st.markdown("Application unifiee avec bankroll distincte par sport.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Utilisateur", username)
    m2.metric("Bankroll Tennis", f"{bankroll_tennis:.2f} EUR")
    m3.metric("Bankroll UFC", f"{bankroll_ufc:.2f} EUR")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Section Tennis")
        st.write("Matchs a venir, conseils de paris, classement Elo, stats et mise a jour.")
        if st.button("Ouvrir l'app Tennis", use_container_width=True):
            st.session_state["_goto_section"] = "Tennis"
            st.rerun()
    with c2:
        st.subheader("Section UFC")
        st.write("Evenements a venir, recommandations, classement Elo et mise a jour.")
        if st.button("Ouvrir l'app UFC", use_container_width=True):
            st.session_state["_goto_section"] = "UFC"
            st.rerun()


def main() -> None:
    try:
        st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    except Exception:
        pass

    if not st.session_state.get("_unified_db_initialized"):
        init_database()
        st.session_state["_unified_db_initialized"] = True

    user = current_user()
    if not user:
        render_auth_page()
        return

    user = get_user_by_id(int(user["id"]))
    if not user:
        st.session_state.pop("user_id", None)
        st.rerun()
        return

    username = str(user["username"])
    is_admin = int(user["is_admin"]) == 1
    ensure_user_legacy_files(username, is_admin=is_admin)
    _set_unified_session_context(user)

    with st.sidebar:
        st.markdown("### Session")
        st.write(f"Connecte: **{username}**")
        st.caption(f"Role: {'Admin' if is_admin else 'Utilisateur'}")
        if st.button("Se deconnecter", use_container_width=True):
            st.session_state.pop("user_id", None)
            st.session_state.pop("logged_in_user", None)
            st.session_state.pop("unified_ufc_bets_folder", None)
            st.session_state["_goto_section"] = "Accueil"
            st.rerun()
        sections = ["Accueil", "Tennis", "UFC"]
        if is_admin and username == ADMIN_USERNAME:
            sections.append("Administration")
        pending_section = st.session_state.pop("_goto_section", None)
        if pending_section in sections:
            st.session_state["section"] = pending_section
        if "section" not in st.session_state or st.session_state["section"] not in sections:
            st.session_state["section"] = "Accueil"
        st.radio("Navigation", sections, key="section")

    section = st.session_state["section"]
    if section == "Accueil":
        _render_unified_home(user)
    elif section == "Tennis":
        _render_legacy_tennis(user)
    elif section == "UFC":
        _render_legacy_ufc(user)
    elif section == "Administration" and is_admin and username == ADMIN_USERNAME:
        render_admin_panel(user)
    else:
        st.error("Section invalide.")


if __name__ == "__main__":
    main()
