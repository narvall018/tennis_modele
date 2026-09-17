"""ATP-only runtime of the frozen reference-price hypothesis (paper only).

Deliberately independent of research scripts, trained bundles and WTA data.
The reference score is not a calibrated probability or confidence bound.
"""
from copy import deepcopy
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd

from src.app.tennis_strategy import BOOKMAKERS, utc

STRATEGY_ID = 'atp_reference_price_paper_v1'
RULE = {'minimum_ev_after_haircut': .03, 'profit_haircut': .02,
        'odds_min': 1.3, 'odds_max': 5., 'probability_buffer': .005,
        'max_reference_overround': 1.08, 'max_execution_overround': 1.20,
        'max_quote_age_seconds': 300, 'max_pair_gap_seconds': 180,
        'min_minutes_before_start': 10, 'max_hours_before_start': 48,
        'score_method': 'minimum_proportional_power_minus_buffer_v1'}
RULE_SHA256 = hashlib.sha256(json.dumps(RULE, sort_keys=True).encode()).hexdigest()


def reference_scores(prices):
    odds = np.asarray(prices, dtype=float)
    if odds.shape != (2,) or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Référence Pinnacle incomplète ou invalide.')
    inv = 1/odds
    margin = inv.sum()
    if not 1-1e-12 <= margin <= RULE['max_reference_overround']:
        raise ValueError('Marge Pinnacle hors limites (0–8 %).')
    proportional = inv/margin
    low, high = 1., 64.
    for _ in range(60):
        exponent = (low+high)/2
        if (inv**exponent).sum() > 1:
            low = exponent
        else:
            high = exponent
    power = inv**((low+high)/2)
    return np.maximum(0, np.minimum(proportional, power)-RULE['probability_buffer'])


def _pair(book, names, now):
    markets = [m for m in book.get('markets', []) if m.get('key') == 'h2h']
    if len(markets) != 1:
        raise ValueError('Marché vainqueur absent ou dupliqué.')
    market = markets[0]
    outcomes = market.get('outcomes') or []
    if len(outcomes) != 2 or {o.get('name') for o in outcomes} != set(names):
        raise ValueError('Deux issues exactes du même match requises ; aucun nul supprimé.')
    stamp_value = market.get('last_update') or book.get('last_update')
    if not stamp_value:
        raise ValueError('Horodatage de cote absent.')
    stamp = utc(stamp_value)
    if not pd.Timedelta(0) <= now-stamp <= pd.Timedelta(seconds=RULE['max_quote_age_seconds']):
        raise ValueError('Cotes périmées (> 5 minutes) ou datées du futur : relancer le scan.')
    by_name = {o['name']: float(o['price']) for o in outcomes}
    prices = [by_name[n] for n in names]
    if not np.isfinite(prices).all() or min(prices) <= 1:
        raise ValueError('Cote non finie ou ≤ 1.')
    return {'prices': prices, 'quote_at': stamp.isoformat()}


def _event_key(names, start):
    normalised = [' '.join(unicodedata.normalize('NFKC', n).casefold().split()) for n in names]
    identity = {'tour': 'ATP', 'day': start.tz_convert('Europe/Paris').date().isoformat(),
                'players': sorted(normalised)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def analyse_event(event, now=None):
    now = utc(now)
    base = {'event_id': event.get('id', ''), 'match': f"{event.get('home_team', '?')} — {event.get('away_team', '?')}",
            'competition': event.get('sport_title', ''), 'candidate': None, 'checked_books': 0}
    try:
        if not str(event.get('sport_key', '')).startswith('tennis_atp_'):
            raise ValueError('ATP uniquement : WTA, football et MMA exclus de cette section.')
        names = [event['home_team'], event['away_team']]
        if not event.get('id') or any(not isinstance(n, str) or not n.strip() or '/' in n for n in names) or names[0] == names[1]:
            raise ValueError('Identifiants absents, ambigus ou double non pris en charge.')
        start = utc(event['commence_time'])
        if not pd.Timedelta(minutes=RULE['min_minutes_before_start']) <= start-now <= pd.Timedelta(hours=RULE['max_hours_before_start']):
            raise ValueError('Hors fenêtre : match requis entre 10 minutes et 48 heures avant le début.')
        books, invalid = {}, []
        seen = set()
        for book in event.get('bookmakers') or []:
            key = book.get('key')
            if key not in {*BOOKMAKERS, 'pinnacle'}:
                continue
            if key in seen:
                raise ValueError('Bookmaker dupliqué : relevé ambigu, relancer le scan.')
            seen.add(key)
            try:
                books[key] = _pair(book, names, now)
            except (ValueError, KeyError, TypeError) as error:
                invalid.append(f'{key} : {error}')
        if 'pinnacle' not in books:
            raise ValueError('Référence Pinnacle complète et récente absente. ' + ' ; '.join(invalid))
        reference = books['pinnacle']
        scores = reference_scores(reference['prices'])
        candidates = []
        for book, pair in sorted(books.items()):
            if book == 'pinnacle':
                continue
            if abs(utc(pair['quote_at'])-utc(reference['quote_at'])) > pd.Timedelta(seconds=RULE['max_pair_gap_seconds']):
                invalid.append(f'{book} : références espacées de plus de trois minutes.'); continue
            odds = np.asarray(pair['prices'])
            if not 1-1e-12 <= (1/odds).sum() <= RULE['max_execution_overround']:
                invalid.append(f'{book} : marge hors limites (0–20 %).'); continue
            base['checked_books'] += 1
            ev = scores*(1+(odds-1)*(1-RULE['profit_haircut']))-1
            eligible = (odds >= RULE['odds_min']) & (odds <= RULE['odds_max']) & (ev >= RULE['minimum_ev_after_haircut'])
            if not eligible.any():
                continue
            side = int(np.argmax(np.where(eligible, ev, -np.inf)))
            fixture = {'tour': 'ATP', 'player_1': names[0], 'player_2': names[1],
                       'tournament': event.get('sport_title', ''), 'start': start.isoformat(), 'bookmaker': book,
                       'odds_1': float(odds[0]), 'odds_2': float(odds[1]), 'quote_at': pair['quote_at'],
                       'reference_pair': reference, 'source_event': deepcopy(event)}
            candidates.append({'strategy_id': STRATEGY_ID, 'rule_sha256': RULE_SHA256,
                'fixture': fixture, 'event_key': _event_key(names, start), 'computed_at': now.isoformat(),
                'selected_side': side, 'pick': names[side], 'odds': float(odds[side]),
                # Generic ledger's historical column name; UI explicitly calls it a score.
                'probability': float(scores[side]), 'reference_scores': scores.tolist(),
                'expected_returns': ev.tolist(), 'eligible': True, 'real_money_authorised': False})
        return {**base, 'status': 'candidate' if candidates else 'no_signal' if base['checked_books'] else 'blocked',
                'reason': 'Critères théoriques satisfaits — simulation uniquement.' if candidates else
                          'Aucune cote analysable ne passe la règle.' if base['checked_books'] else
                          'Aucune paire française récente comparable à Pinnacle.',
                'blocked_books': invalid,
                'candidate': max(candidates, key=lambda c: c['expected_returns'][c['selected_side']]) if candidates else None}
    except (ValueError, TypeError, KeyError) as error:
        return {**base, 'status': 'blocked', 'reason': str(error)}


def validate_fixture(fixture, now=None):
    result = analyse_event(fixture.get('source_event', {}), now)
    current = result.get('candidate')
    if not current or current['fixture'] != fixture:
        raise ValueError('Sélection absente, périmée ou modifiée : relancer le scan. ' + result['reason'])
    return np.asarray([fixture['odds_1'], fixture['odds_2']])


def verify_candidate(candidate, now=None):
    now = utc(now)
    if candidate.get('strategy_id') != STRATEGY_ID or candidate.get('rule_sha256') != RULE_SHA256:
        raise ValueError('Ancienne stratégie ou règle modifiée : recalcul obligatoire.')
    computed = utc(candidate['computed_at'])
    if not pd.Timedelta(0) <= now-computed <= pd.Timedelta(seconds=RULE['max_quote_age_seconds']):
        raise ValueError('Calcul périmé : relancer le scan.')
    current = analyse_event(candidate.get('fixture', {}).get('source_event', {}), now).get('candidate')
    if not current:
        raise ValueError('Les cotes ne respectent plus les contrôles de fraîcheur : relancer le scan.')
    for key in current:
        if key != 'computed_at' and current[key] != candidate.get(key):
            raise ValueError('Sélection modifiée depuis le calcul : relancer le scan.')
    return current


def allocation(results, summary):
    """One event per strategy; deterministic start order; total suggested stake capped."""
    used = {b['event_key'] for b in summary['bets']}
    best = {}
    for r in results:
        c = r.get('candidate')
        if not c or c['event_key'] in used:
            continue
        key = c['event_key']
        if key not in best or c['expected_returns'][c['selected_side']] > best[key]['expected_returns'][best[key]['selected_side']]:
            best[key] = c
    from src.app.tennis_strategy_ledger import proposed_stake
    remaining = dict(summary)
    planned = []
    for c in sorted(best.values(), key=lambda c: (utc(c['fixture']['start']), c['event_key'])):
        amount = proposed_stake(remaining)
        planned.append({'candidate': c, 'stake_cents': amount})
        remaining['day_remaining_cents'] -= amount
        remaining['available_cents'] -= amount
    return planned
