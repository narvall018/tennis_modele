"""Isolated per-sport paper accounts; football draws are first-class outcomes."""
import json
import sqlite3

from src.app import tennis_strategy_ledger as shared
from src.app.tennis_strategy_ledger import initialise, state, proposed_stake, settle
from src.app.tennis_strategy import utc
from src.app.value_methods import STRATEGIES, verify_candidate


def database(root, sport):
    return root / 'bets' / f'value_{sport}.sqlite3'


def record(root, sport, owner, candidate, now=None):
    now = utc(now)
    current = verify_candidate(root, sport, candidate, now)
    path = database(root, sport)
    conn = shared.connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        summary = shared._state(conn, owner, now)
        stake = proposed_stake(summary)
        if stake <= 0:
            raise ValueError('Budget quotidien ou capital disponible épuisé.')
        conn.execute('''INSERT INTO atp_paper_bets
            (owner,strategy,event_key,decision_day,created_at,start_at,pick,odds,probability,stake_cents,snapshot)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (str(owner), STRATEGIES[sport], current['event_key'], summary['decision_day'], now.isoformat(),
             current['fixture']['start'], current['pick'], current['odds'], current['probability'], stake,
             json.dumps(current, ensure_ascii=False, allow_nan=False)))
        conn.commit()
        return stake / 100
    except sqlite3.IntegrityError as error:
        conn.rollback()
        raise ValueError('Cette rencontre figure déjà dans ton carnet pour cette stratégie.') from error
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def export_backup(path, owner, sport, now=None):
    return shared.export_backup(path, owner, now, strategy_id=STRATEGIES[sport], backup_format=f'value-{sport}-paper-v1')


def restore_backup(path, owner, text, sport, now=None):
    return shared.restore_backup(path, owner, text, now, strategy_id=STRATEGIES[sport], backup_format=f'value-{sport}-paper-v1')
