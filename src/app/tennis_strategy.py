"""Prospective ATP paper selections using the frozen nonlinear research family.

No fallback to Elo, synthetic prices, WTA or inferred surface is permitted.
The historical ROI is not evidence that this French-price deployment is profitable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtesting.nonlinear_market import inputs
from src.app.tennis_strategy_features import one_fixture_features
from src.backtesting.three_sport_residual import abbreviated_key
from src.data.tennis_pipeline import _full_name_key

STRATEGY_ID = 'atp_trees_recent_paper_v1'
BOOKMAKERS = {'betclic_fr': 'Betclic', 'winamax_fr': 'Winamax', 'pmu_fr': 'PMU',
              'unibet_fr': 'Unibet', 'netbet_fr': 'NetBet'}
ROUNDS = {'1st Round': '1er tour', '2nd Round': '2e tour', '3rd Round': '3e tour',
          '4th Round': '4e tour', 'Quarterfinals': 'Quart de finale',
          'Semifinals': 'Demi-finale', 'The Final': 'Finale', 'Round Robin': 'Poules'}
ROUND_CODES = dict(zip(ROUNDS, ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR']))
MAX_QUOTE_MINUTES = 15
MAX_DATA_DAYS = 7


def utc(value=None):
    stamp = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Un horodatage avec fuseau horaire est obligatoire.')
    return stamp.tz_convert('UTC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(root):
    folder = Path(root) / 'models/tennis_strategy'
    meta = json.loads((folder / 'metadata.json').read_text())
    for name, expected in meta['files'].items():
        if name not in {'booster.ubj', 'history.csv.gz', 'protocol.json'}:
            raise ValueError('Fichier inattendu dans le manifeste ATP.')
        if digest(folder / name) != expected:
            raise ValueError(f'Artefact ATP incohérent : {name}. Reconstruire le paquet.')
    if meta['strategy_id'] != STRATEGY_ID or meta['candidate'] != 'trees_recent':
        raise ValueError('Mauvaise stratégie ATP : aucun remplacement automatique autorisé.')
    history = pd.read_csv(folder / 'history.csv.gz', low_memory=False)
    history['match_date'] = pd.to_datetime(history['match_date'])
    protocol = json.loads((folder / 'protocol.json').read_text())
    model = xgb.Booster(params={'nthread': 1})
    model.load_model(folder / 'booster.ubj')
    model.set_param({'nthread': 1})
    return meta, history, protocol, model


def freshness_reasons(meta, now=None):
    now = utc(now)
    reasons = []
    current_day = now.tz_convert('Europe/Paris').tz_localize(None).normalize()
    if int(meta['model_year']) != current_day.year:
        reasons.append('Modèle annuel à reconstruire pour cette année.')
    latest = pd.Timestamp(meta['history_last_date'])
    age = (current_day - latest).days
    if age < 0:
        reasons.append('Historique daté dans le futur : calcul bloqué.')
    elif age > MAX_DATA_DAYS:
        reasons.append(f'Historique trop ancien ({age} jours ; limite {MAX_DATA_DAYS}). '
                       'Actualiser les données ATP puis reconstruire le paquet Stratégie ATP.')
    return reasons


def validate_fixture(fixture, now=None):
    now = utc(now)
    if not fixture.get('start') or not fixture.get('quote_at'):
        raise ValueError('Heure du match et horodatage des cotes obligatoires.')
    start, quote = utc(fixture['start']), utc(fixture['quote_at'])
    if start <= now:
        raise ValueError('Match déjà commencé : aucun pari pré-match.')
    if start > now + pd.Timedelta(days=7):
        raise ValueError('Match à plus de sept jours : attendre des informations plus proches.')
    if quote > now or now - quote > pd.Timedelta(minutes=MAX_QUOTE_MINUTES):
        raise ValueError('Cote future ou âgée de plus de 15 minutes : relever les deux cotes à nouveau.')
    if fixture.get('bookmaker') not in BOOKMAKERS:
        raise ValueError('Bookmaker français non pris en charge.')
    if fixture.get('tour') != 'ATP' or fixture.get('singles_main_draw') is not True:
        raise ValueError('Cette stratégie concerne uniquement le simple ATP, tableau principal.')
    if fixture.get('surface') not in {'Hard', 'Clay', 'Grass'}:
        raise ValueError('Surface non confirmée.')
    if fixture.get('indoor') not in {'Indoor', 'Outdoor'} or fixture.get('round') not in ROUNDS:
        raise ValueError('Contexte de match incomplet.')
    if fixture.get('best_of') not in {3, 5} or fixture.get('level') not in {'G', 'M', 'F', '500', '250', 'A'}:
        raise ValueError('Format ou niveau de tournoi invalide.')
    if not str(fixture.get('tournament', '')).strip():
        raise ValueError('Tournoi obligatoire.')
    if not fixture.get('player_1') or fixture.get('player_1') == fixture.get('player_2'):
        raise ValueError('Deux joueurs distincts sont nécessaires.')
    odds = np.array([fixture['odds_1'], fixture['odds_2']], float)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Les deux cotes doivent être réelles et supérieures à 1.')
    overround = float((1 / odds).sum())
    if not 1 <= overround <= 1.20:
        raise ValueError('Paire de cotes hors du domaine du test (somme des probabilités : 100–120 %).')
    api = fixture.get('api_pair')
    if api:
        for side in [1, 2]:
            if _full_name_key(api[f'player_{side}']) != abbreviated_key(fixture[f'player_{side}']):
                raise ValueError('Identité du joueur différente du relevé API ; aucun transfert de cote autorisé.')
            if float(api[f'odds_{side}']) != float(fixture[f'odds_{side}']):
                raise ValueError('Paire API modifiée.')
        if (api['bookmaker'] != fixture['bookmaker'] or utc(api['start']) != start
                or utc(api['quote_at']) != quote):
            raise ValueError('Bookmaker ou horodatage API modifié.')
    return odds


def latest_profiles(history, before):
    past = history[history['match_date'] < pd.Timestamp(before).tz_localize(None).normalize()]
    records = []
    for side in [1, 2]:
        names = ['id', 'name', 'rank', 'rank_points', 'age', 'ht', 'hand']
        cols = [f'player_{side}_{name}' for name in names]
        block = past[['match_date', *cols]].rename(columns={c: c.replace(f'player_{side}_', '') for c in cols})
        records.append(block)
    profiles = pd.concat(records).sort_values('match_date', kind='stable')
    # No fuzzy alias or implicit transfer between homonyms.
    ambiguous = profiles.groupby('name')['id'].nunique()
    profiles = profiles[~profiles['name'].isin(ambiguous[ambiguous != 1].index)]
    return profiles.drop_duplicates('name', keep='last').set_index('name')


def fixture_feature_row(history, fixture, progress=lambda _: None):
    """Replay the historical formulas for one resultless, non-updating fixture.

    This is deliberately an explicit, cached calculation, not a per-rerun fit.
    Current-day history is excluded; no invented result enters player state.
    """
    start = utc(fixture['start'])
    day = start.tz_convert('Europe/Paris').tz_localize(None).normalize()
    past = history[history['match_date'] < day].copy()
    profiles = latest_profiles(past, day)
    row = {'match_id': 'prospective:fixture', 'match_date': day, 'match_status': 'upcoming',
           'surface': fixture['surface'], 'tourney_name': fixture['tournament'],
           'round': ROUND_CODES[fixture['round']], 'indoor': 'I' if fixture['indoor'] == 'Indoor' else 'O',
           'best_of': fixture['best_of'], 'tourney_level': fixture['level'],
           'player_1_won': 0, 'minutes': np.nan}
    for side in [1, 2]:
        name = fixture[f'player_{side}']
        if name not in profiles.index:
            raise ValueError(f'Joueur absent ou identité ambiguë : {name}.')
        profile = profiles.loc[name]
        row[f'player_{side}_name'] = name
        for field in ['id', 'rank', 'rank_points', 'age', 'ht', 'hand']:
            row[f'player_{side}_{field}'] = profile[field]
        # Current ranks/points must be confirmed explicitly in the UI; they are
        # not silently taken from the player's last historical match.
        for field in ['rank', 'rank_points']:
            value = float(fixture[f'player_{side}_{field}'])
            if not math.isfinite(value) or value < (1 if field == 'rank' else 0):
                raise ValueError('Classement ou points non valides.')
            row[f'player_{side}_{field}'] = value
        row[f'player_{side}_odds'] = fixture[f'odds_{side}']
        old_age = float(profile['age'])
        row[f'player_{side}_age'] = old_age + (day - profile['match_date']).days / 365.25
    selected = one_fixture_features(past, row)
    selected['price_0'], selected['price_1'] = fixture['odds_1'], fixture['odds_2']
    return selected


def score_fixture(bundle, fixture, now=None, progress=lambda _: None):
    meta, history, protocol, model = bundle
    now = utc(now)
    reasons = freshness_reasons(meta, now)
    if reasons:
        raise ValueError(' '.join(reasons))
    odds = validate_fixture(fixture, now)
    features = fixture_feature_row(history, fixture, progress)
    x, q, _, names, signs = inputs(features, 'atp', protocol)
    if names != meta['features'] or signs.tolist() != meta['swap_signs']:
        raise ValueError('Descripteurs incompatibles avec le modèle figé.')
    margin = np.log(q[:, 0] / q[:, 1])
    direct = model.predict(xgb.DMatrix(x, base_margin=margin, nthread=1)).astype(float)
    reverse = model.predict(xgb.DMatrix(x * signs, base_margin=-margin, nthread=1)).astype(float)
    p0 = float((direct[0] + 1 - reverse[0]) / 2)
    probabilities = np.clip([p0, 1 - p0], 1e-7, 1 - 1e-7)
    ev = probabilities * (1 + (odds - 1) * .98) - 1
    eligible = (odds >= 1.3) & (odds <= 5) & (ev >= .02)
    side = int(np.argmax(np.where(eligible, ev, -np.inf))) if eligible.any() else None
    identity = {'date': utc(fixture['start']).tz_convert('Europe/Paris').date().isoformat(),
                'players': sorted([fixture['player_1'], fixture['player_2']])}
    event_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {'strategy_id': STRATEGY_ID, 'model_sha256': meta['files']['booster.ubj'],
            'history_sha256': meta['files']['history.csv.gz'], 'event_key': event_key,
            'fixture': dict(fixture), 'computed_at': now.isoformat(),
            'probabilities': probabilities.tolist(), 'market_probabilities': q[0].tolist(),
            'expected_returns': ev.tolist(), 'selected_side': side,
            'pick': fixture[f'player_{side + 1}'] if side is not None else '',
            'odds': float(odds[side]) if side is not None else None,
            'probability': float(probabilities[side]) if side is not None else None,
            'eligible': side is not None,
            'reason': 'Critères théoriques satisfaits — simulation uniquement.' if side is not None
                      else 'Aucun côté ne passe EV ≥ 2 % après décote, avec une cote de 1,30 à 5,00.',
            'real_money_authorised': False}


def french_quotes(events, now=None):
    """Coherent same-book pairs only. Never mix maxima across bookmakers."""
    now = utc(now)
    rows = []
    for event in events:
        if not str(event.get('sport_key', '')).startswith('tennis_atp_'):
            continue
        left, right = event.get('home_team'), event.get('away_team')
        if not event.get('id') or not left or not right or left == right:
            continue
        if not event.get('commence_time'):
            continue
        try:
            start = utc(event.get('commence_time'))
        except (ValueError, TypeError):
            continue
        if not now < start <= now + pd.Timedelta(days=7):
            continue
        for book in event.get('bookmakers', []):
            if book.get('key') not in BOOKMAKERS:
                continue
            for market in book.get('markets', []):
                if market.get('key') != 'h2h':
                    continue
                outcomes = market.get('outcomes', [])
                if len(outcomes) != 2 or {o.get('name') for o in outcomes} != {left, right}:
                    continue
                timestamp_value = market.get('last_update') or book.get('last_update')
                if not timestamp_value:
                    continue
                try:
                    timestamp = utc(timestamp_value)
                    prices = {o['name']: float(o['price']) for o in outcomes}
                    if not all(math.isfinite(v) and v > 1 for v in prices.values()):
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                if timestamp > now or now - timestamp > pd.Timedelta(minutes=MAX_QUOTE_MINUTES):
                    continue
                rows.append({'event_id': event['id'], 'start': start.isoformat(),
                             'player_1': left, 'player_2': right, 'bookmaker': book['key'],
                             'odds_1': prices[left], 'odds_2': prices[right],
                             'quote_at': timestamp.isoformat(), 'competition': event.get('sport_title', '')})
    return rows
