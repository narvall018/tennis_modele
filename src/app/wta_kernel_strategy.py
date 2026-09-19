"""Portable WTA kernel inference. No research imports, no reference bookmaker."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

STRATEGY_ID = 'wta_kernel_flexible_paper_v1'
BOOKMAKERS = {'betclic_fr': 'Betclic', 'winamax_fr': 'Winamax', 'pmu_fr': 'PMU',
             'unibet_fr': 'Unibet', 'netbet_fr': 'NetBet'}
FOLDER = 'models/wta_kernel_strategy'


def utc(value=None):
    stamp = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None: raise ValueError('Horodatage avec fuseau obligatoire.')
    return stamp.tz_convert('UTC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def name_key(name):
    plain = unicodedata.normalize('NFKD', str(name))
    return ' '.join(re.findall('[a-z]+', ''.join(c for c in plain if not unicodedata.combining(c)).lower()))


def prepare_history(raw, today):
    """Validate service counts; retain invalid-stat records only as identity evidence."""
    rows = raw.copy()
    rows['_start'] = pd.to_datetime(rows.tourney_date.astype(str), format='%Y%m%d', errors='raise')
    rows = rows[rows._start.between(pd.Timestamp('2007-01-01'), pd.Timestamp(today))].copy()
    if rows.duplicated(['tourney_id', 'match_num']).any(): raise ValueError('Statistiques dupliquées.')
    rows = rows.dropna(subset=['winner_id', 'loser_id', 'winner_name', 'loser_name'])
    rows = rows[rows.winner_id.ne(rows.loser_id)].copy()
    valid = rows.surface.isin(['Hard', 'Clay', 'Grass']) & rows.best_of.eq(3)
    valid &= ~rows.score.fillna('').astype(str).str.contains(r'[A-Za-z]|^\s*$', regex=True)
    for side in ['w', 'l']:
        fields = [side+'_'+s for s in ['svpt', '1stIn', '1stWon', '2ndWon']]
        counts = rows[fields].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        points, firstin, firstwon, secondwon = counts.T
        valid &= (np.isfinite(counts).all(axis=1) & (counts >= 0).all(axis=1)
                  & (counts == np.floor(counts)).all(axis=1) & (points > 0)
                  & (firstin <= points) & (firstwon <= firstin) & (secondwon <= points-firstin))
        rows[side+'_points'], rows[side+'_won'] = points, firstwon+secondwon
    rows['_valid'] = valid
    columns = ['tourney_id', 'match_num', '_start', 'surface', 'winner_id', 'loser_id',
               'winner_name', 'loser_name', 'w_points', 'w_won', 'l_points', 'l_won', '_valid']
    return rows[columns].sort_values(['_start', 'tourney_id', 'match_num']).reset_index(drop=True)


def freshness_reasons(meta, now=None):
    day = utc(now).tz_convert('Europe/Paris').tz_localize(None).normalize()
    reasons = []
    if meta['model_year'] != day.year: reasons.append('Modèle annuel WTA à reconstruire pour cette année.')
    age = (day-pd.Timestamp(meta['history_last_date'])).days
    # With the registered28day lag, starts newer than day-28 cannot be consumed.
    if not 0 <= age <= 35:
        reasons.append('Statistiques WTA trop anciennes : actualiser le paquet (délai du modèle 28 jours, tolérance 7 jours).')
    return reasons


def load_bundle(root):
    folder = Path(root)/FOLDER
    meta = json.loads((folder/'metadata.json').read_text())
    if meta.get('strategy_id') != STRATEGY_ID or set(meta['files']) != {'model.npz', 'history.csv.gz'}:
        raise ValueError('Paquet WTA kernel incompatible.')
    for name, expected in meta['files'].items():
        if digest(folder/name) != expected: raise ValueError('Empreinte du paquet WTA incohérente.')
    with np.load(folder/'model.npz', allow_pickle=False) as saved:
        model = {name: saved[name] for name in saved.files}
    shapes = {'mean': (14,), 'scale': (14,), 'signs': (14,), 'projection': (14, 128),
              'phase': (128,), 'coefficients': (143, 1)}
    if set(model) != set(shapes) or any(model[k].shape != s or not np.isfinite(model[k]).all() for k, s in shapes.items()):
        raise ValueError('Dimensions du modèle WTA incompatibles.')
    if (model['scale'] <= 0).any() or not np.array_equal(model['signs'], [-1.]*9+[1.]*5):
        raise ValueError('Transformation WTA invalide.')
    history = pd.read_csv(folder/'history.csv.gz', parse_dates=['_start'])
    if len(history) != meta['history_rows'] or history.duplicated(['tourney_id', 'match_num']).any():
        raise ValueError('Historique WTA incohérent.')
    if str(history.loc[history._valid, '_start'].max().date()) != meta['history_last_date']:
        raise ValueError('Dernière date de statistiques incohérente.')
    for prefix in ['winner', 'loser']:
        history['_'+prefix+'_key'] = history[prefix+'_name'].map(name_key)
    return meta, history, model


def valid_pair(a, b):
    prices = np.asarray([a, b], dtype=float)
    if not np.isfinite(prices).all() or (prices <= 1).any() or not 1 <= (1/prices).sum() <= 1.2:
        raise ValueError('Deux cotes réelles du même bookmaker, marge entre 0 et 20 %, sont requises.')
    return prices


def validate_fixture(f, now=None):
    now = utc(now)
    if f.get('tour') != 'WTA' or f.get('singles_main_draw') is not True:
        raise ValueError('Confirmer le simple WTA tableau principal, hors qualifications et WTA125.')
    if f.get('surface') not in ['Hard', 'Clay', 'Grass'] or not f.get('tournament'):
        raise ValueError('Surface et tournoi à confirmer.')
    if not f.get('player_1') or not f.get('player_2') or name_key(f['player_1']) == name_key(f['player_2']):
        raise ValueError('Deux joueuses distinctes sont requises.')
    if f.get('bookmaker') not in BOOKMAKERS: raise ValueError('Bookmaker français non pris en charge.')
    if not now+pd.Timedelta(minutes=10) < utc(f['start']) <= now+pd.Timedelta(days=7):
        raise ValueError('Match commencé, à moins de dix minutes ou à plus de sept jours.')
    if not f.get('quote_at') or not pd.Timedelta(0) <= now-utc(f['quote_at']) <= pd.Timedelta(minutes=5):
        raise ValueError('Cotes expirées ou futures : relever une paire de moins de cinq minutes.')
    prices = valid_pair(f['odds_1'], f['odds_2'])
    api = f.get('api_pair')
    if api:
        for key in ['player_1', 'player_2', 'bookmaker', 'start', 'quote_at', 'odds_1', 'odds_2']:
            if api.get(key) != f.get(key): raise ValueError('Relevé API modifié : consulter à nouveau les cotes.')
    return prices


def fixture_inputs(history, f, now=None):
    # Never anticipate statistics becoming available between now and match day.
    day = min(utc(now), utc(f['start'])).tz_convert('Europe/Paris').tz_localize(None).normalize()
    past = history[history._start+pd.Timedelta(days=28) < day]
    identities = []
    for player in [f['player_1'], f['player_2']]:
        ids = set()
        for prefix in ['winner', 'loser']:
            keys = past['_'+prefix+'_key'] if '_'+prefix+'_key' in past else past[prefix+'_name'].map(name_key)
            ids.update(past.loc[keys.eq(name_key(player)), prefix+'_id'].tolist())
        if len(ids) != 1: raise ValueError(f'Identité absente ou ambiguë dans les données antérieures : {player}.')
        identities.append(next(iter(ids)))
    if identities[0] == identities[1]: raise ValueError('Les deux noms désignent la même joueuse.')
    recent = past[past._valid & past._start.ge(day-pd.Timedelta(days=365))]
    profiles, counts = [], []
    for identity in identities:
        own = []
        for side, other, prefix in [('w', 'l', 'winner'), ('l', 'w', 'loser')]:
            part = recent[recent[prefix+'_id'].eq(identity)]
            values = pd.DataFrame({'date': part._start, 'surface': part.surface,
                'sw': part[side+'_won'], 'sp': part[side+'_points'],
                'rw': part[other+'_points']-part[other+'_won'], 'rp': part[other+'_points']})
            own.append(values)
        own = pd.concat(own); counts.append(len(own))
        if len(own) < 5: raise ValueError('Moins de cinq matchs de statistiques antérieures pour une joueuse.')
        vector = []
        for surface in [None, f['surface']]:
            subset = own if surface is None else own[own.surface.eq(surface)]
            weights = np.exp2(-(day-subset.date).dt.days.to_numpy()/180.)
            sw, sp, rw, rp = weights@subset[['sw', 'sp', 'rw', 'rp']].to_numpy(float)
            s, r = (sw+300)/(sp+500), (rw+200)/(rp+500)
            vector.extend([np.log(s/(1-s)), np.log(r/(1-r)), np.log1p(sp+rp)])
        profiles.append(np.asarray(vector))
    prices = valid_pair(f['odds_1'], f['odds_2']); inverse = 1/prices; q = inverse/inverse.sum()
    z = np.log(q[0]/q[1])
    x = np.r_[z, z*abs(z), z*inverse.sum(), profiles[0]-profiles[1],
              [float(f['surface'] == s) for s in ['Hard', 'Clay', 'Grass']],
              np.log1p(min(counts)), np.log1p(sum(counts)/2)]
    return x[None, :], q[None, :], identities


def probabilities(model, x, q):
    def raw(values, market):
        z = np.clip((values-model['mean'])/model['scale'], -5, 5)
        design = np.column_stack([np.ones(len(z)), z/np.sqrt(z.shape[1]),
            np.sqrt(2/128)*np.cos(z@model['projection']+model['phase'])])
        logits = np.log(market).copy(); logits[:, :1] += design@model['coefficients']
        weights = np.exp(logits-logits.max(axis=1, keepdims=True))
        return weights/weights.sum(axis=1, keepdims=True)
    left = (raw(x, q)[:, 0]+raw(x*model['signs'], q[:, ::-1])[:, 1])/2
    p = np.clip(np.column_stack([left, 1-left]), 1e-12, 1-1e-12)
    return p/p.sum(axis=1, keepdims=True)


def score_fixture(bundle, f, now=None):
    now = utc(now); meta, history, model = bundle
    reasons = freshness_reasons(meta, now)
    if reasons: raise ValueError(' '.join(reasons))
    odds = validate_fixture(f, now)
    if utc(f['start']).tz_convert('Europe/Paris').year != meta['model_year']: raise ValueError('Match hors année du modèle.')
    x, q, identities = fixture_inputs(history, f, now)
    p = probabilities(model, x, q)[0]; ev = p*(1+(odds-1)*.98)-1
    eligible = (odds >= 1.3) & (odds <= 5) & (ev >= .03)
    side = int(np.argmax(np.where(eligible, ev, -np.inf))) if eligible.any() else None
    event = {'tour': 'WTA', 'day': str(utc(f['start']).tz_convert('Europe/Paris').date()),
             'players': sorted(map(str, identities))}
    return {'strategy_id': STRATEGY_ID, 'fixture': dict(f), 'computed_at': now.isoformat(),
        'event_key': hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),
        'model_sha256': meta['files']['model.npz'], 'history_sha256': meta['files']['history.csv.gz'],
        'probabilities': p.tolist(), 'expected_returns': ev.tolist(), 'market_probabilities': q[0].tolist(),
        'eligible': side is not None, 'selected_side': side,
        'pick': f[f'player_{side+1}'] if side is not None else '',
        'odds': float(odds[side]) if side is not None else None,
        'probability': float(p[side]) if side is not None else None,
        'reason': 'Signal expérimental — simulation uniquement.' if side is not None else 'Aucune opportunité selon la règle figée.',
        'real_money_authorised': False}


def quotes(events, now=None):
    result = []
    for event in events:
        if not str(event.get('sport_key', '')).startswith('tennis_wta_') or not event.get('id'): continue
        a, b = event.get('home_team'), event.get('away_team')
        if not a or not b or a == b: continue
        for book in event.get('bookmakers') or []:
            if book.get('key') not in BOOKMAKERS: continue
            for market in book.get('markets') or []:
                outcomes = market.get('outcomes') or []
                if market.get('key') != 'h2h' or len(outcomes) != 2 or {o.get('name') for o in outcomes} != {a, b}: continue
                try:
                    prices = {o['name']: float(o['price']) for o in outcomes}
                    stamp = market.get('last_update') or book.get('last_update')
                    row = {'event_id': event['id'], 'competition': event.get('sport_title', event['sport_key']),
                        'sport_key': event['sport_key'], 'player_1': a, 'player_2': b,
                        'start': utc(event['commence_time']).isoformat(), 'quote_at': utc(stamp).isoformat() if stamp else '',
                        'bookmaker': book['key'], 'odds_1': prices[a], 'odds_2': prices[b]}
                    validate_fixture({**row, 'tour': 'WTA', 'surface': 'Hard', 'singles_main_draw': True,
                                      'tournament': row['competition']}, now)  # context validated separately in UI
                    result.append(row)
                except (ValueError, TypeError, KeyError): continue
    return result
