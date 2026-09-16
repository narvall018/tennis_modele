"""Isolated, transactional, per-user paper bankroll. Never sends a bet to a book."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from src.app.tennis_strategy import STRATEGY_ID, utc, validate_fixture


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS atp_paper_accounts (
        owner TEXT PRIMARY KEY, initial_cents INTEGER NOT NULL CHECK(initial_cents>0), created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS atp_paper_bets (
        id INTEGER PRIMARY KEY, owner TEXT NOT NULL, strategy TEXT NOT NULL,
        event_key TEXT NOT NULL, decision_day TEXT NOT NULL, created_at TEXT NOT NULL,
        start_at TEXT NOT NULL, pick TEXT NOT NULL, odds REAL NOT NULL,
        probability REAL NOT NULL, stake_cents INTEGER NOT NULL CHECK(stake_cents>0),
        status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','won','lost','void')),
        profit_cents INTEGER NOT NULL DEFAULT 0, settled_at TEXT, snapshot TEXT NOT NULL,
        UNIQUE(owner,strategy,event_key));
    ''')
    return conn


def initialise(path, owner, initial, now=None):
    if not owner or not math.isfinite(initial) or not 10 <= initial <= 1_000_000:
        raise ValueError('Capital initial requis entre 10 et 1 000 000 €.')
    conn = connect(path)
    try:
        conn.execute('INSERT INTO atp_paper_accounts VALUES (?,?,?)',
                     (str(owner), round(initial * 100), utc(now).isoformat()))
    except sqlite3.IntegrityError as error:
        raise ValueError('Cette bankroll existe déjà ; son capital initial est figé.') from error
    finally:
        conn.close()


def _state(conn, owner, now):
    account = conn.execute('SELECT * FROM atp_paper_accounts WHERE owner=?', (str(owner),)).fetchone()
    if account is None:
        raise ValueError('Initialiser la bankroll de simulation.')
    bets = [dict(row) for row in conn.execute('SELECT * FROM atp_paper_bets WHERE owner=? ORDER BY id', (str(owner),))]
    day = now.tz_convert('Europe/Paris').date().isoformat()
    midnight = now.tz_convert('Europe/Paris').normalize().tz_convert('UTC')
    initial = account['initial_cents']
    profit = sum(b['profit_cents'] for b in bets)
    held = sum(b['stake_cents'] for b in bets if b['status'] == 'pending')
    # Settlement timestamps, not event dates: late results cannot rewrite a past day's capital.
    base = initial + sum(b['profit_cents'] for b in bets
                         if b['settled_at'] and utc(b['settled_at']) < midnight)
    used = sum(b['stake_cents'] for b in bets if b['decision_day'] == day)
    settled_stakes = sum(b['stake_cents'] for b in bets if b['status'] != 'pending' and b['status'] != 'void')
    return {'initial_cents': initial, 'balance_cents': initial + profit,
            'available_cents': max(0, initial + profit - held), 'reserved_cents': held,
            'profit_cents': profit, 'day_base_cents': max(0, base), 'day_used_cents': used,
            'day_remaining_cents': max(0, math.floor(max(0, base) * .02) - used),
            'roi': profit / settled_stakes if settled_stakes else None,
            'decision_day': day, 'bets': bets}


def state(path, owner, now=None):
    conn = connect(path)
    try:
        return _state(conn, owner, utc(now))
    finally:
        conn.close()


def proposed_stake(summary):
    # Floor, never round up through a risk cap; stakes remain integral cents.
    return min(math.floor(summary['day_base_cents'] * .0025),
               summary['day_remaining_cents'], summary['available_cents'])


def record(path, owner, candidate, now=None, *, strategy_id=STRATEGY_ID, validator=validate_fixture):
    now = utc(now)
    if candidate.get('strategy_id') != strategy_id or not candidate.get('eligible'):
        raise ValueError('Sélection absente ou stratégie incompatible.')
    validator(candidate['fixture'], now)
    computed = utc(candidate['computed_at'])
    if computed > now or now - computed > pd.Timedelta(minutes=15):
        raise ValueError('Calcul trop ancien : recalculer avec des cotes fraîches.')
    p, odds = float(candidate['probability']), float(candidate['odds'])
    if not math.isfinite(p) or not 0 < p < 1 or not 1.3 <= odds <= 5 or p * (1 + (odds - 1) * .98) - 1 < .02:
        raise ValueError('La sélection ne respecte plus la règle figée.')
    side = candidate['selected_side']
    if side not in (0, 1) or candidate['pick'] != candidate['fixture'][f'player_{side+1}']:
        raise ValueError('Côté sélectionné incohérent.')
    if odds != float(candidate['fixture'][f'odds_{side+1}']):
        raise ValueError('Cote modifiée depuis le calcul.')
    conn = connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        summary = _state(conn, owner, now)
        stake = proposed_stake(summary)
        if stake <= 0:
            raise ValueError('Budget quotidien ou capital disponible épuisé.')
        conn.execute('''INSERT INTO atp_paper_bets
            (owner,strategy,event_key,decision_day,created_at,start_at,pick,odds,probability,stake_cents,snapshot)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (str(owner), strategy_id, candidate['event_key'], summary['decision_day'], now.isoformat(),
             utc(candidate['fixture']['start']).isoformat(), candidate['pick'], odds, p, stake,
             json.dumps(candidate, ensure_ascii=False, allow_nan=False)))
        conn.commit()
        return stake / 100
    except sqlite3.IntegrityError as error:
        conn.rollback()
        raise ValueError('Ce match figure déjà dans ton carnet pour cette stratégie.') from error
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def settle(path, owner, bet_id, result, now=None):
    now = utc(now)
    if result not in {'won', 'lost', 'void'}:
        raise ValueError('Résultat invalide.')
    conn = connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        bet = conn.execute('SELECT * FROM atp_paper_bets WHERE owner=? AND id=?', (str(owner), bet_id)).fetchone()
        if bet is None or bet['status'] != 'pending':
            raise ValueError('Pari absent, appartenant à un autre compte ou déjà réglé.')
        if utc(bet['start_at']) > now:
            raise ValueError('Attendre le début prévu avant de saisir le résultat.')
        profit = (round(bet['stake_cents'] * (bet['odds'] - 1) * .98) if result == 'won'
                  else -bet['stake_cents'] if result == 'lost' else 0)
        conn.execute('UPDATE atp_paper_bets SET status=?,profit_cents=?,settled_at=? WHERE owner=? AND id=?',
                     (result, profit, now.isoformat(), str(owner), bet_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def export_backup(path, owner, now=None, *, strategy_id=STRATEGY_ID, backup_format='atp-paper-v1'):
    summary = state(path, owner, now)
    # Only this user's paper account, never the authentication DB or other users.
    return json.dumps({'format': backup_format, 'strategy': strategy_id,
                       'initial_cents': summary['initial_cents'], 'bets': summary['bets']},
                      ensure_ascii=False, indent=2)


def restore_backup(path, owner, text, now=None, *, strategy_id=STRATEGY_ID, backup_format='atp-paper-v1'):
    """Restore into an empty account only. No overwrite or merge ambiguity."""
    now = utc(now)
    if len(text) > 10_000_000:
        raise ValueError('Sauvegarde trop volumineuse.')
    data = json.loads(text)
    if data.get('format') != backup_format or data.get('strategy') != strategy_id:
        raise ValueError('Format de sauvegarde incompatible.')
    initial = data['initial_cents']
    if type(initial) is not int or not 1000 <= initial <= 100_000_000:
        raise ValueError('Capital de sauvegarde invalide.')
    allowed = ['strategy','event_key','decision_day','created_at','start_at','pick','odds','probability',
               'stake_cents','status','profit_cents','settled_at','snapshot']
    conn = connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM atp_paper_accounts WHERE owner=?', (str(owner),)).fetchone():
            raise ValueError('Restauration réservée à un compte sans bankroll ; aucune donnée existante écrasée.')
        conn.execute('INSERT INTO atp_paper_accounts VALUES (?,?,?)', (str(owner), initial, utc(now).isoformat()))
        for b in data['bets']:
            stake, odds = b['stake_cents'], float(b['odds'])
            if (b['strategy'] != strategy_id or type(stake) is not int or stake <= 0
                    or not math.isfinite(odds) or not 1.3 <= odds <= 5
                    or not math.isfinite(float(b['probability'])) or not 0 < float(b['probability']) < 1
                    or b['status'] not in {'pending', 'won', 'lost', 'void'}):
                raise ValueError('Pari invalide dans la sauvegarde.')
            for field in ['created_at', 'start_at']:
                utc(b[field])
            created, start = utc(b['created_at']), utc(b['start_at'])
            if created > now or created >= start or b['decision_day'] != created.tz_convert('Europe/Paris').date().isoformat():
                raise ValueError('Chronologie de décision invalide dans la sauvegarde.')
            if b['settled_at']:
                if not start <= utc(b['settled_at']) <= now:
                    raise ValueError('Chronologie de règlement invalide dans la sauvegarde.')
            expected = round(stake * (odds - 1) * .98) if b['status'] == 'won' else -stake if b['status'] == 'lost' else 0
            if b['profit_cents'] != expected or (b['status'] == 'pending') != (b['settled_at'] is None):
                raise ValueError('Règlement incohérent dans la sauvegarde.')
            snapshot = json.loads(b['snapshot'])
            if snapshot.get('strategy_id') != strategy_id:
                raise ValueError('Snapshot incompatible.')
            conn.execute(f"INSERT INTO atp_paper_bets (owner,{','.join(allowed)}) VALUES ({','.join(['?'] * (len(allowed)+1))})",
                         [str(owner), *[b[k] for k in allowed]])
        restored = _state(conn, owner, now)
        if restored['balance_cents'] < 0 or restored['reserved_cents'] > restored['balance_cents']:
            raise ValueError('Capital ou réservations incohérents dans la sauvegarde.')
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
