"""Isolated football paper ledger; recompute before accepting any candidate.

Three-way markets do not fit the two-sided tennis recorder, so `record` is
written here. Everything else -- accounts, state, staking, settlement -- is the
shared machinery, pointed at its own database file.
"""
from functools import partial
import json
import sqlite3

from src.app import football_exchange_strategy as engine
from src.app import tennis_strategy_ledger as shared
from src.app.tennis_strategy_ledger import initialise, state, proposed_stake, settle  # noqa: F401


def record(path, owner, candidate, now=None, *, root):
    """Re-score the fixture and refuse anything that moved since it was shown."""
    now = engine.utc(now)
    if candidate.get('strategy_id') != engine.STRATEGY_ID or not candidate.get('eligible'):
        raise ValueError('Sélection non éligible pour cette stratégie.')
    fresh = engine.score_fixture(engine.load_bundle(root), candidate['fixture'], now)
    for field in ['strategy_id', 'event_key', 'rule_sha256', 'outcome', 'eligible',
                  'selected_side', 'pick', 'odds', 'probability', 'expected_return']:
        if candidate.get(field) != fresh[field]:
            raise ValueError('Calcul ou règle modifié : recalculer la sélection.')
    summary = shared.state(path, owner, now)
    stake_cents = shared.proposed_stake(summary)   # already in cents, do not scale again
    if stake_cents <= 0:
        raise ValueError('Budget quotidien épuisé ou bankroll insuffisante.')
    conn = shared.connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'INSERT INTO atp_paper_bets (owner,strategy,event_key,decision_day,created_at,'
            'start_at,pick,odds,probability,stake_cents,snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (str(owner), engine.STRATEGY_ID, fresh['event_key'], summary['decision_day'],
             now.isoformat(), engine.utc(fresh['fixture']['start']).isoformat(), fresh['pick'],
             float(fresh['odds']), float(fresh['probability']), int(stake_cents),
             json.dumps(fresh, ensure_ascii=False, allow_nan=False)))
        conn.execute('COMMIT')
    except sqlite3.IntegrityError as error:
        conn.execute('ROLLBACK')
        raise ValueError('Cette rencontre est déjà enregistrée dans le carnet.') from error
    finally:
        conn.close()
    return stake_cents/100


export_backup = partial(shared.export_backup, strategy_id=engine.STRATEGY_ID,
                        backup_format='football-exchange-paper-v1')
restore_backup = partial(shared.restore_backup, strategy_id=engine.STRATEGY_ID,
                         backup_format='football-exchange-paper-v1')
