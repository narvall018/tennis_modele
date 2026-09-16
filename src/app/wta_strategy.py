"""WTA recent-tree paper inference; preserve Bet365/Pinnacle model inputs."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.app.tennis_strategy import BOOKMAKERS, utc, digest
from src.backtesting.wta_price_residual import features
from src.backtesting.three_sport_residual import abbreviated_key
from src.data.tennis_pipeline import _full_name_key
from src.features.elo_system import TennisEloEngine, ROUND_MAP
from src.features.feature_builder import PlayerState

STRATEGY_ID = 'wta_trees_recent_cross_book_paper_v1'
HISTORY_COLUMNS = ['_source_row_id', '_date', '_p1', '_p2', '_label', '_status',
                   '_surface', '_series', '_round', '_tournament']


def freshness_reasons(meta, now=None):
    day = utc(now).tz_convert('Europe/Paris').date()
    reasons = []
    if meta['model_year'] != day.year:
        reasons.append('Modèle annuel WTA à reconstruire après audit.')
    age = (day - pd.Timestamp(meta['history_last_date']).date()).days
    if age < 0 or age > 7:
        reasons.append(f'Historique WTA hors limite de fraîcheur ({age} jours ; maximum 7). Actualiser les données WTA.')
    return reasons


def load_bundle(root):
    folder = Path(root) / 'models/wta_live_strategy'
    meta = json.loads((folder / 'metadata.json').read_text())
    expected_files = {'booster.ubj', 'history.csv.gz', 'protocol.json'}
    if set(meta['files']) != expected_files or meta['strategy_id'] != STRATEGY_ID or meta['candidate'] != 'trees_recent':
        raise ValueError('Paquet WTA incompatible.')
    for name, expected in meta['files'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'Empreinte WTA incohérente : {name}.')
    history = pd.read_csv(folder / 'history.csv.gz', low_memory=False)
    history['_date'] = pd.to_datetime(history['_date'])
    if len(history) != meta['history_rows'] or str(history['_date'].max().date()) != meta['history_last_date']:
        raise ValueError('Dates ou taille du paquet WTA incohérentes.')
    protocol = json.loads((folder / 'protocol.json').read_text())
    model = xgb.Booster(params={'nthread': 1})
    model.load_model(folder / 'booster.ubj')
    model.set_param({'nthread': 1})
    return meta, history, protocol, model


def valid_pair(left, right):
    pair = np.asarray([left, right], dtype=float)
    if not np.isfinite(pair).all() or (pair <= 1).any() or not 1 <= (1 / pair).sum() <= 1.2:
        raise ValueError('Paire incomplète ou incohérente : deux cotes > 1, somme des probabilités 100–120 %.')
    return pair


def validate_fixture(f, now=None):
    now = utc(now)
    if f.get('tour') != 'WTA' or f.get('singles_main_draw') is not True:
        raise ValueError('Simple WTA tableau principal uniquement, pas ATP, doubles, qualifications ou WTA 125.')
    if not f.get('player_1') or not f.get('player_2') or f['player_1'] == f['player_2']:
        raise ValueError('Deux joueuses distinctes sont obligatoires.')
    if f.get('surface') not in {'Hard', 'Clay', 'Grass'} or not str(f.get('tournament', '')).strip():
        raise ValueError('Surface et tournoi WTA à confirmer.')
    if f.get('bookmaker') not in BOOKMAKERS:
        raise ValueError('Bookmaker français non pris en charge.')
    if not now < utc(f['start']) <= now + pd.Timedelta(days=7):
        raise ValueError('Match commencé ou à plus de sept jours.')
    timestamps = []
    for field in ['quote_at', 'bet365_quote_at', 'pinnacle_quote_at']:
        stamp = utc(f.get(field)) if f.get(field) else None
        if stamp is None or stamp > now or now - stamp > pd.Timedelta(minutes=15):
            raise ValueError('Les trois relevés doivent être horodatés, non futurs et âgés de moins de 15 minutes.')
        timestamps.append(stamp)
    if max(timestamps) - min(timestamps) > pd.Timedelta(minutes=5):
        raise ValueError('Les trois relevés sont espacés de plus de cinq minutes : relever les cotes ensemble.')
    for side in [1, 2]:
        rank = float(f[f'player_{side}_rank'])
        if not math.isfinite(rank) or not 1 <= rank <= 3000:
            raise ValueError('Classements WTA actuels à confirmer.')
    for prefix in ['', 'bet365_', 'pinnacle_']:
        valid_pair(f[prefix+'odds_1'], f[prefix+'odds_2'])
    for field, prefix in [('api_pair', ''), ('pinnacle_api_pair', 'pinnacle_')]:
        api = f.get(field)
        if not api:
            continue
        for side in [1, 2]:
            if _full_name_key(api[f'player_{side}']) != abbreviated_key(f[f'player_{side}']):
                raise ValueError('Identité différente du relevé API ; vérifier les deux joueuses dans le même ordre.')
            if float(api[f'odds_{side}']) != float(f[f'{prefix}odds_{side}']):
                raise ValueError('Cote API modifiée.')
        expected_book = 'pinnacle' if prefix else f['bookmaker']
        if api['bookmaker'] != expected_book or utc(api['start']) != utc(f['start']) or utc(api['quote_at']) != utc(f[prefix+'quote_at']):
            raise ValueError('Horodatage ou bookmaker API modifié.')
    if f.get('api_pair') and f.get('pinnacle_api_pair') and f['api_pair']['event_id'] != f['pinnacle_api_pair']['event_id']:
        raise ValueError('Les références API ne correspondent pas au même match.')
    return valid_pair(f['odds_1'], f['odds_2'])


def replay(history, before):
    """Same updates/order as audited rebuild_daily_features, without every feature row."""
    day = pd.Timestamp(before).tz_localize(None).normalize()
    past = history[pd.to_datetime(history['_date']) < day].copy()
    past['_round_order'] = past['_round'].map(ROUND_MAP).fillna(3)
    past = past.sort_values(['_date', '_tournament', '_round_order', '_p1'], kind='stable')
    engine, states = TennisEloEngine(), defaultdict(PlayerState)
    for stamp, batch in past.groupby('_date', sort=True):
        played = pd.Timestamp(stamp).date()
        completed = batch[batch['_status'].eq('completed')]
        for name in set(completed['_p1']) | set(completed['_p2']):
            snapshot = deepcopy(engine._get_or_create(name))
            engine._apply_decay(snapshot, played)
            engine._players[name] = snapshot
        # itertuples renames underscore-prefixed fields, so unpack positional rows.
        for a, b, label, surface, series, round_ in completed[['_p1', '_p2', '_label', '_surface', '_series', '_round']].itertuples(index=False, name=None):
            winner, loser = (a, b) if label == 1 else (b, a)
            engine._update_pair(engine._players[winner], engine._players[loser], surface, series, round_, played)
            states[a].update(played, label == 1, surface, b, series)
            states[b].update(played, label == 0, surface, a, series)
    return engine, states


def fixture_features(history, f, state=None):
    before = utc(f['start']).tz_convert('Europe/Paris').tz_localize(None).normalize()
    engine, states = state if state is not None else replay(history, before)
    names = [f['player_1'], f['player_2']]
    if any(name not in engine._players for name in names):
        raise ValueError('Joueuse absente de l’historique : pas de profil inventé.')
    a, b = [deepcopy(engine._players[name]) for name in names]
    for rating in [a, b]:
        engine._apply_decay(rating, before.date())
    s1, s2 = [states[name] for name in names]
    surface = f['surface']
    return pd.DataFrame([{'elo_diff': a.global_elo - b.global_elo,
        'surface_elo_diff': a.surface_elo[surface] - b.surface_elo[surface],
        'form_10_diff': s1.form(10) - s2.form(10),
        'rest_diff': s1.days_rest(before.date()) - s2.days_rest(before.date()),
        'fatigue_diff': s1.fatigue(before.date()) - s2.fatigue(before.date()),
        'log_rank_diff': math.log(float(f['player_1_rank']) + 1) - math.log(float(f['player_2_rank']) + 1),
        '_surface': surface, 'B365_1': f['bet365_odds_1'], 'B365_2': f['bet365_odds_2'],
        'Pinnacle_1': f['pinnacle_odds_1'], 'Pinnacle_2': f['pinnacle_odds_2']}])


def model_inputs(frame, protocol):
    x, q, _, names = features(frame, 'cross_book_form', protocol['wta_features'])
    context = np.column_stack([frame['_surface'].eq(s).to_numpy(float) for s in ['Hard', 'Clay', 'Grass']])
    return np.column_stack([x, context]), q, names + ['is_hard', 'is_clay', 'is_grass'], np.r_[-np.ones(x.shape[1]), np.ones(3)]


def score_fixture(bundle, f, now=None, state=None):
    meta, history, protocol, model = bundle
    now = utc(now)
    reasons = freshness_reasons(meta, now)
    if reasons:
        raise ValueError(' '.join(reasons))
    odds = validate_fixture(f, now)
    x, q, names, signs = model_inputs(fixture_features(history, f, state), protocol)
    if names != meta['features'] or signs.tolist() != meta['swap_signs']:
        raise ValueError('Variables différentes du modèle WTA figé.')
    margin = np.log(q[:, 0] / q[:, 1])
    direct = model.predict(xgb.DMatrix(x, base_margin=margin, nthread=1))
    reverse = model.predict(xgb.DMatrix(x * signs, base_margin=-margin, nthread=1))
    p0 = float((float(direct[0]) + 1 - float(reverse[0])) / 2)
    p = np.clip([p0, 1 - p0], 1e-7, 1 - 1e-7)
    ev = p * (1 + (odds - 1) * .98) - 1
    eligible = (odds >= 1.3) & (odds <= 5) & (ev >= .02)
    side = int(np.argmax(np.where(eligible, ev, -np.inf))) if eligible.any() else None
    identity = {'tour': 'WTA', 'date': utc(f['start']).tz_convert('Europe/Paris').date().isoformat(),
                'players': sorted([f['player_1'], f['player_2']])}
    return {'strategy_id': STRATEGY_ID, 'fixture': dict(f), 'computed_at': now.isoformat(),
            'event_key': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            'model_sha256': meta['files']['booster.ubj'], 'history_sha256': meta['files']['history.csv.gz'],
            'probabilities': p.tolist(), 'expected_returns': ev.tolist(), 'market_probabilities': q[0].tolist(),
            'eligible': side is not None, 'selected_side': side,
            'pick': f[f'player_{side+1}'] if side is not None else '',
            'odds': float(odds[side]) if side is not None else None,
            'probability': float(p[side]) if side is not None else None,
            'reason': 'Critères théoriques satisfaits — simulation WTA, prix français non validés.' if side is not None
                      else 'Aucune sélection WTA ne respecte le seuil et la plage de cotes.',
            'real_money_authorised': False}


def quotes(events, now=None):
    now = utc(now)
    rows = []
    for event in events:
        if not str(event.get('sport_key', '')).startswith('tennis_wta_') or not event.get('id'):
            continue
        left, right = event.get('home_team'), event.get('away_team')
        if not left or not right or left == right or not event.get('commence_time'):
            continue
        try:
            start = utc(event['commence_time'])
            if not now < start <= now + pd.Timedelta(days=7):
                continue
        except (ValueError, TypeError):
            continue
        for book in event.get('bookmakers') or []:
            if book.get('key') not in {*BOOKMAKERS, 'pinnacle'}:
                continue
            for market in book.get('markets') or []:
                outcomes = market.get('outcomes') or []
                if market.get('key') != 'h2h' or len(outcomes) != 2 or {o.get('name') for o in outcomes} != {left, right}:
                    continue
                timestamp = market.get('last_update') or book.get('last_update')
                if not timestamp:
                    continue
                try:
                    stamp = utc(timestamp)
                    pair = {o['name']: float(o['price']) for o in outcomes}
                    valid_pair(pair[left], pair[right])
                    if stamp > now or now - stamp > pd.Timedelta(minutes=15):
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                rows.append({'event_id': event['id'], 'competition': event.get('sport_title', ''),
                             'start': start.isoformat(), 'player_1': left, 'player_2': right,
                             'bookmaker': book['key'], 'odds_1': pair[left], 'odds_2': pair[right],
                             'quote_at': stamp.isoformat()})
    return rows
