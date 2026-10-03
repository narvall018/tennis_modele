"""Two distinct, frozen experimental strategies for the application."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd

from src.app.tennis_strategy import utc
from src.app.atp_reference_strategy import allocation
from src.app.ufc_markov import score_fight

BOOKMAKERS = {'betclic_fr': 'Betclic', 'netbet_fr': 'NetBet', 'pmu_fr': 'PMU',
              'unibet_fr': 'Unibet', 'winamax_fr': 'Winamax'}
FOOTBALL_SPORTS = {'soccer_france_ligue_one': 'Ligue 1', 'soccer_france_ligue_two': 'Ligue 2',
    'soccer_epl': 'Premier League', 'soccer_germany_bundesliga': 'Bundesliga',
    'soccer_italy_serie_a': 'Serie A', 'soccer_spain_la_liga': 'Liga',
    'soccer_uefa_champs_league': 'Ligue des champions', 'soccer_uefa_europa_league': 'Ligue Europa',
    'soccer_uefa_europa_conference_league': 'Ligue Conférence',
    'soccer_netherlands_eredivisie': 'Eredivisie', 'soccer_portugal_primeira_liga': 'Portugal'}
MMA_KEY = 'mma_mixed_martial_arts'
STRATEGIES = {'football': 'football_price_gaps_paper_v1', 'ufc': 'ufc_markov_paper_v1'}
COMMON_RULE = {'haircut': .02, 'odds_min': 1.3, 'odds_max': 5., 'quote_age_seconds': 300,
               'quote_gap_seconds': 180, 'minimum_minutes_to_start': 10, 'horizon_days': 14,
               'execution_margin_max': 1.20}
RULES = {'football': {**COMMON_RULE, 'minimum_ev': .03, 'probability_buffer': .005,
                     'reference_margin_max': 1.08, 'score': 'min_proportional_power_minus_buffer'},
         'ufc': {**COMMON_RULE, 'minimum_ev': .26, 'selection': 'most_likely_winner',
                 'duration_policy': 'same_favourite_and_edge_in_both_3_and_5_rounds'}}
RULE_HASHES = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in RULES.items()}


def normalise(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).casefold() if c.isalnum())


def reference_scores(prices):
    odds = np.asarray(prices, float)
    if odds.shape != (3,) or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Référence 1X2 complète requise, nul compris.')
    inverse = 1 / odds
    margin = inverse.sum()
    if not 1 - 1e-12 <= margin <= RULES['football']['reference_margin_max']:
        raise ValueError('Marge de référence hors limites (0–8 %).')
    low, high = 1., 64.
    for _ in range(60):
        middle = (low + high) / 2
        if (inverse ** middle).sum() > 1:
            low = middle
        else:
            high = middle
    return np.maximum(0., np.minimum(inverse / margin, inverse ** ((low + high) / 2)) - .005)


def complete_market(book, outcomes, now):
    markets = [m for m in book.get('markets', []) if m.get('key') == 'h2h']
    if len(markets) != 1:
        raise ValueError('Marché vainqueur absent ou dupliqué.')
    market = markets[0]
    rows = market.get('outcomes') or []
    if len(rows) != len(outcomes) or {r.get('name') for r in rows} != set(outcomes):
        raise ValueError('Issues incomplètes ou contrat différent ; aucune issue supprimée.')
    stamp = market.get('last_update') or book.get('last_update')
    if not stamp:
        raise ValueError('Horodatage de cote absent.')
    stamp = utc(stamp)
    if not pd.Timedelta(0) <= utc(now) - stamp <= pd.Timedelta(seconds=300):
        raise ValueError('Cotes périmées (> 5 minutes) ou futures.')
    by_name = {r['name']: float(r['price']) for r in rows}
    prices = np.array([by_name[o] for o in outcomes])
    if not np.isfinite(prices).all() or min(prices) <= 1:
        raise ValueError('Cotes invalides.')
    return {'prices': prices.tolist(), 'quote_at': stamp.isoformat()}


def event_key(sport, event, start):
    identity = {'sport': sport, 'league': event['sport_key'], 'day': start.normalize().isoformat(),
                'participants': sorted([normalise(event['home_team']), normalise(event['away_team'])])}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def match_official(event, cards, now):
    checked = utc(cards['checked_at'])
    if not pd.Timedelta(0) <= utc(now) - checked <= pd.Timedelta(hours=48):
        raise ValueError('Programme officiel UFC périmé : actualiser les données.')
    names = {normalise(event['home_team']), normalise(event['away_team'])}
    day = utc(event['commence_time']).normalize().tz_localize(None)
    matched = [r for r in cards['fights'] if {normalise(r['fighter_1']), normalise(r['fighter_2'])} == names
               and 0 <= (day - pd.Timestamp(r['event_date'])).days <= 1]
    if len(matched) != 1:
        raise ValueError('Combat absent ou ambigu dans le programme officiel UFC ; autre MMA exclu.')
    official = deepcopy(matched[0])
    if normalise(official['fighter_1']) != normalise(event['home_team']):
        for suffix in ['', '_id']:
            official[f'fighter_1{suffix}'], official[f'fighter_2{suffix}'] = official[f'fighter_2{suffix}'], official[f'fighter_1{suffix}']
    return official


def analyse_event(sport, event, now=None, bundle=None, cards=None):
    now, rule = utc(now), RULES[sport]
    base = {'event_id': event.get('id', ''), 'match': f"{event.get('home_team', '?')} — {event.get('away_team', '?')}",
            'competition': FOOTBALL_SPORTS.get(event.get('sport_key'), event.get('sport_title', 'UFC')),
            'candidate': None, 'checked_books': 0, 'blocked_books': []}
    try:
        if (sport == 'football' and event.get('sport_key') not in FOOTBALL_SPORTS) or (sport == 'ufc' and event.get('sport_key') != MMA_KEY):
            raise ValueError('Événement hors du sport et des compétitions de cette section.')
        names = [event['home_team'], event['away_team']]
        if not event.get('id') or any(not isinstance(n, str) or not n.strip() for n in names) or normalise(names[0]) == normalise(names[1]):
            raise ValueError('Identités de participants absentes ou ambiguës.')
        start = utc(event['commence_time'])
        if not pd.Timedelta(minutes=10) <= start - now <= pd.Timedelta(days=14):
            raise ValueError('Hors fenêtre : début requis entre dix minutes et quatorze jours.')
        outcomes = [names[0], 'Draw', names[1]] if sport == 'football' else names
        books, seen, rejected = {}, set(), []
        for book in event.get('bookmakers') or []:
            key = book.get('key')
            if key not in {*BOOKMAKERS, 'pinnacle'}:
                continue
            if key in seen:
                raise ValueError('Bookmaker dupliqué : relevé ambigu.')
            seen.add(key)
            try:
                books[key] = complete_market(book, outcomes, now)
            except (ValueError, KeyError, TypeError):
                rejected.append(f'{BOOKMAKERS.get(key, key)} : marché incomplet, périmé ou invalide.')
        model = None
        if sport == 'football':
            if 'pinnacle' not in books:
                raise ValueError('Référence Pinnacle 1X2 complète et récente absente.')
            reference = books['pinnacle']
            scores = reference_scores(reference['prices'])
        else:
            if not bundle or not cards:
                raise ValueError('Données du modèle ou programme officiel UFC indisponibles.')
            official = match_official(event, cards, now)
            short, long = [score_fight(bundle, official, rounds, now) for rounds in (3, 5)]
            p3, p5 = np.array(short['win_probabilities']), np.array(long['win_probabilities'])
            model = {'three_rounds': short, 'five_rounds': long, 'official': official}
            base['model_probabilities_3'] = p3.tolist()
            base['model_probabilities_5'] = p5.tolist()
            if np.argmax(p3) != np.argmax(p5) or abs(p3[0] - .5) < 1e-12 or abs(p5[0] - .5) < 1e-12:
                raise ValueError('Le favori du modèle change avec la durée prévue, ou les probabilités sont égales.')
            scores = np.minimum(p3, p5)  # Duration sensitivity, not a confidence bound.
            favourite = int(np.argmax(p3))
        candidates = []
        for key, pair in sorted(books.items()):
            if key == 'pinnacle':
                continue
            odds = np.array(pair['prices'])
            if not 1 - 1e-12 <= (1 / odds).sum() <= rule['execution_margin_max']:
                rejected.append(f'{BOOKMAKERS[key]} : marge hors limites (0–20 %).'); continue
            if sport == 'football' and abs(utc(pair['quote_at']) - utc(reference['quote_at'])) > pd.Timedelta(seconds=180):
                rejected.append(f'{BOOKMAKERS[key]} : référence et cote espacées de plus de trois minutes.'); continue
            base['checked_books'] += 1
            ev = scores * (1 + (odds - 1) * .98) - 1
            eligible = (odds >= 1.3) & (odds <= 5.) & (ev >= rule['minimum_ev'])
            if sport == 'ufc':
                eligible &= np.arange(2) == favourite
            if not eligible.any():
                continue
            side = int(np.argmax(np.where(eligible, ev, -np.inf)))
            fixture = {'start': start.isoformat(), 'bookmaker': key, 'prices': odds.tolist(),
                       'outcomes': outcomes, 'quote_at': pair['quote_at'], 'source_event': deepcopy(event)}
            if sport == 'football':
                fixture['reference'] = reference
            else:
                fixture['model_sha256'] = bundle['sha256']
            candidates.append({'strategy_id': STRATEGIES[sport], 'rule_sha256': RULE_HASHES[sport],
                'fixture': fixture, 'event_key': event_key(sport, event, start), 'computed_at': now.isoformat(),
                'selected_side': side, 'pick': 'Match nul' if outcomes[side] == 'Draw' else outcomes[side],
                'odds': float(odds[side]), 'probability': float(scores[side]), 'scores': scores.tolist(),
                'expected_returns': ev.tolist(), 'model': model, 'eligible': True, 'real_money_authorised': False})
        chosen = max(candidates, key=lambda c: c['expected_returns'][c['selected_side']]) if candidates else None
        return {**base, 'blocked_books': rejected, 'candidate': chosen,
                'status': 'candidate' if chosen else 'no_signal' if base['checked_books'] else 'blocked',
                'reason': 'Critères théoriques satisfaits.' if chosen else 'Aucune cote ne passe le filtre.' if base['checked_books'] else 'Aucun marché français complet et récent analysable.'}
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return {**base, 'status': 'blocked', 'reason': str(error)}


def verify_candidate(root, sport, candidate, now=None):
    now = utc(now)
    if candidate.get('strategy_id') != STRATEGIES[sport] or candidate.get('rule_sha256') != RULE_HASHES[sport]:
        raise ValueError('Stratégie ou règle modifiée : recalcul obligatoire.')
    if not pd.Timedelta(0) <= now - utc(candidate['computed_at']) <= pd.Timedelta(seconds=300):
        raise ValueError('Calcul périmé : relancer le scan.')
    bundle = cards = None
    if sport == 'ufc':
        from src.app.value_methods_data import load_ufc
        bundle, cards = load_ufc(root)
    current = analyse_event(sport, candidate.get('fixture', {}).get('source_event', {}), now, bundle, cards).get('candidate')
    if not current or any(current[k] != candidate.get(k) for k in current if k != 'computed_at'):
        raise ValueError('Sélection, modèle ou cotes périmés ou modifiés : relancer le scan.')
    return current
