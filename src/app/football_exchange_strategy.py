"""Portable football favourite-bias inference on exchange prices. Paper only.

The rule exploits one measured fact and nothing else: on 89 987 matches of the
2012-2023 development seasons, the devigged sharp price understates a strong
favourite. The bias is monotone across 270 000 outcomes, but its bootstrap lower
bound only clears zero between 0.70 and 0.85, so the rule refuses to operate
anywhere else. There is no model file and no fitted parameter: the correction is
a frozen table of *lower bounds*, not point estimates.

Development data was already research-exposed, so this cannot authorise money. It
accumulates prospective evidence, which is the only thing that could.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_ID = 'football_favourite_bias_exchange_paper_v1'
FOLDER = 'models/football_exchange_strategy'
# An exchange quotes offers of unknown depth and charges commission on winnings;
# a bookmaker quotes a price it must honour. Only exchanges carry this rule,
# because no reachable bookmaker's margin fits under the measured bias.
EXCHANGES = {'betfair_ex_eu': 'Betfair', 'betfair_ex_uk': 'Betfair UK',
             'matchbook': 'Matchbook', 'smarkets': 'Smarkets', 'betdaq': 'Betdaq'}
OUTCOMES = ('home', 'draw', 'away')


def utc(value=None):
    stamp = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None: raise ValueError('Horodatage avec fuseau obligatoire.')
    return stamp.tz_convert('UTC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rule_digest(rule):
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()


def load_bundle(root):
    """The frozen rule and its bias table, with the hash that pins them."""
    meta = json.loads((Path(root)/FOLDER/'metadata.json').read_text())
    if meta.get('strategy_id') != STRATEGY_ID: raise ValueError('Paquet football incompatible.')
    if rule_digest(meta['rule']) != meta['rule_sha256']: raise ValueError('Règle football modifiée.')
    bands = meta['rule']['bias_bands']
    if not bands or any(b['lower_bound_points'] <= 0 for b in bands):
        raise ValueError('Une bande sans biais prouvé ne peut pas figurer dans la règle.')
    edges = [(b['from'], b['to']) for b in bands]
    if any(a >= b for a, b in edges) or any(edges[i][1] > edges[i+1][0] for i in range(len(edges)-1)):
        raise ValueError('Bandes de biais mal ordonnées ou chevauchantes.')
    if meta.get('real_money_authorised') is not False:
        raise ValueError('Ce paquet est une simulation ; real_money_authorised doit rester faux.')
    return meta


def reference_probabilities(prices):
    """Proportional devig of a three-way exchange price."""
    odds = np.asarray(prices, dtype=float)
    if odds.shape != (3,) or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Trois cotes réelles du même opérateur sont requises.')
    inverse = 1/odds
    overround = float(inverse.sum())
    return inverse/overround, overround


def bias_for(meta, probability):
    """The frozen lower bound for this probability, or None outside the rule."""
    for band in meta['rule']['bias_bands']:
        if band['from'] <= probability < band['to']:
            return band['lower_bound_points']/100.
    return None


def validate_fixture(f, now=None):
    now = utc(now)
    rule = f.get('_rule') or {}
    if f.get('sport') != 'football' or not f.get('competition'):
        raise ValueError('Confirmer une rencontre de football et sa compétition.')
    teams = [f.get('home_team'), f.get('away_team')]
    if not all(teams) or teams[0] == teams[1]: raise ValueError('Deux équipes distinctes sont requises.')
    if f.get('bookmaker') not in EXCHANGES:
        raise ValueError('Cette règle ne vaut que sur un exchange : aucun bookmaker joignable '
                         'ne facture assez peu pour le biais mesuré.')
    start = utc(f['start'])
    lo = pd.Timedelta(minutes=rule.get('min_minutes_before_start', 10))
    hi = pd.Timedelta(days=rule.get('max_days_before_start', 7))
    if not now+lo < start <= now+hi:
        raise ValueError('Match commencé, trop proche ou trop lointain.')
    age = pd.Timedelta(seconds=rule.get('max_quote_age_seconds', 300))
    if not f.get('quote_at') or not pd.Timedelta(0) <= now-utc(f['quote_at']) <= age:
        raise ValueError('Cotes expirées ou futures : relever à nouveau les trois prix.')
    probabilities, overround = reference_probabilities([f['odds_home'], f['odds_draw'], f['odds_away']])
    ceiling = rule.get('max_reference_overround', 1.02)
    if overround > ceiling:
        raise ValueError(f'Marge {100*(overround-1):.2f} % : au-delà de '
                         f'{100*(ceiling-1):.2f} %, ce n\'est pas un prix d\'exchange.')
    api = f.get('api_quote')
    if api:
        for key in ['home_team', 'away_team', 'bookmaker', 'start', 'quote_at',
                    'odds_home', 'odds_draw', 'odds_away']:
            if api.get(key) != f.get(key): raise ValueError('Relevé API modifié : consulter à nouveau.')
    return probabilities, overround


def score_fixture(meta, f, now=None):
    """Apply the frozen rule to one quoted three-way market."""
    now = utc(now)
    rule = meta['rule']
    probabilities, overround = validate_fixture({**f, '_rule': rule}, now)
    side = int(np.argmax(probabilities))
    reference = float(probabilities[side])
    price = float([f['odds_home'], f['odds_draw'], f['odds_away']][side])
    # The draw is never the strongest outcome in the rule's band, but refuse it explicitly.
    blocked = 'Le nul ne porte pas le biais mesuré.' if OUTCOMES[side] == 'draw' else ''
    bias = bias_for(meta, reference)
    commission = float(rule['commission'])
    corrected = min(reference+(bias or 0.), 1.)
    expected = corrected*(1+(price-1)*(1-commission))-1
    if blocked:
        reason, eligible = blocked, False
    elif bias is None:
        reason = (f'Probabilité {reference:.3f} hors de la bande prouvée '
                  f'[{rule["bias_bands"][0]["from"]:.2f} ; {rule["bias_bands"][-1]["to"]:.2f}].')
        eligible = False
    elif not rule['odds_min'] <= price <= rule['odds_max']:
        reason, eligible = f'Cote {price:.2f} hors bornes de la règle.', False
    elif expected < rule['min_expected_return']:
        reason = (f'Espérance {100*expected:+.2f} % sous le seuil de '
                  f'{100*rule["min_expected_return"]:.2f} % après commission.')
        eligible = False
    else:
        reason, eligible = 'Signal expérimental — simulation uniquement.', True
    event = {'sport': 'football', 'day': str(utc(f['start']).tz_convert('Europe/Paris').date()),
             'teams': sorted([str(f['home_team']), str(f['away_team'])])}
    return {'strategy_id': STRATEGY_ID, 'fixture': dict(f), 'computed_at': now.isoformat(),
            'event_key': hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),
            'rule_sha256': meta['rule_sha256'], 'outcome': OUTCOMES[side],
            'reference_probability': reference, 'bias_applied': bias,
            'corrected_probability': corrected if bias is not None else None,
            'overround': overround, 'expected_return': float(expected),
            'eligible': eligible, 'selected_side': side if eligible else None,
            'pick': f['home_team'] if side == 0 else (f['away_team'] if side == 2 else 'Nul'),
            'odds': price, 'probability': corrected if bias is not None else None,
            'reason': reason, 'real_money_authorised': False}


def best_quotes(rows):
    """One quote per match: several exchanges list the same game, keep the best price.

    Comparison is on the favourite's price at each venue, since that is the only
    side the rule can ever back.
    """
    best = {}
    for row in rows:
        try:
            probabilities, _ = reference_probabilities(
                [row['odds_home'], row['odds_draw'], row['odds_away']])
        except ValueError:
            continue
        side = int(np.argmax(probabilities))
        price = float([row['odds_home'], row['odds_draw'], row['odds_away']][side])
        key = (str(row.get('event_id')), OUTCOMES[side])
        if key not in best or price > best[key][0]:
            best[key] = (price, row)
    return [row for _, row in sorted(best.values(), key=lambda item: str(item[1].get('start')))]


def quotes(events, now=None):
    """Three-way exchange prices from an odds-API payload, already guarded."""
    result = []
    for event in events or []:
        if not str(event.get('sport_key', '')).startswith('soccer_') or not event.get('id'): continue
        home, away = event.get('home_team'), event.get('away_team')
        if not home or not away or home == away: continue
        for book in event.get('bookmakers') or []:
            if book.get('key') not in EXCHANGES: continue
            for market in book.get('markets') or []:
                outcomes = market.get('outcomes') or []
                if market.get('key') != 'h2h' or len(outcomes) != 3: continue
                names = {o.get('name') for o in outcomes}
                if not {home, away} <= names or len(names) != 3: continue
                draw = next(iter(names-{home, away}))
                try:
                    price = {o['name']: float(o['price']) for o in outcomes}
                    stamp = market.get('last_update') or book.get('last_update')
                    result.append({'event_id': event['id'], 'sport': 'football',
                        'competition': event.get('sport_title', event['sport_key']),
                        'sport_key': event['sport_key'], 'home_team': home, 'away_team': away,
                        'draw_label': draw, 'start': utc(event['commence_time']).isoformat(),
                        'quote_at': utc(stamp).isoformat() if stamp else '',
                        'bookmaker': book['key'], 'odds_home': price[home],
                        'odds_draw': price[draw], 'odds_away': price[away]})
                except (ValueError, TypeError, KeyError): continue
    return result
