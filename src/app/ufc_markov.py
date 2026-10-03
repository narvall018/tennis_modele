"""Experimental UFC continuous-time Markov model; no published ROI is inherited.

The seven states are standing, either fighter controlling the ground, and four
absorbing KO/submission results. Rates use shrunk, pre-card UFCStats counts.
Each round starts standing. A symmetric decision model is fitted only to
chronological pre-card features. This is an adaptation, not the 13-GLM Bayesian
implementation in Holmes, McHale and Zychaluk (2023).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import expit

VERSION = 'ufc-markov-counts-v1'
CONFIG = {'history_years': 8, 'prior_minutes': 30., 'minimum_fights': 2,
          'decision_ridge': 1., 'minimum_decisions': 100,
          'states': ['standing', 'ground_1', 'ground_2', 'ko_1', 'sub_1', 'ko_2', 'sub_2']}
SOURCE = 'https://doi.org/10.1016/j.ijforecast.2022.01.007'
FIELDS = ['minutes', 'sig', 'sig_att', 'absorbed', 'absorbed_att', 'td', 'td_att',
          'td_allowed', 'td_allowed_att', 'sub_att', 'reversals', 'control',
          'ko_for', 'ko_against', 'sub_for', 'sub_against', 'fights']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def weight_class(value):
    return str(value).replace(' Bout', '').replace(' Title', '').strip()


def _empty():
    return dict.fromkeys(FIELDS, 0.)


def _prior(total):
    minutes = total['minutes']
    defaults = {'sig': 3., 'sig_att': 7., 'absorbed': 3., 'absorbed_att': 7.,
                'td': .12, 'td_att': .3, 'td_allowed': .12, 'td_allowed_att': .3,
                'sub_att': .06, 'reversals': .02, 'control': .2,
                'ko_for': .015, 'ko_against': .015, 'sub_for': .008, 'sub_against': .008}
    return {k: max(total[k] / minutes, 1e-6) if minutes > 0 else v for k, v in defaults.items()}


def _rates(profile, prior):
    exposure = profile['minutes'] + CONFIG['prior_minutes']
    return {k: (profile[k] + CONFIG['prior_minutes'] * v) / exposure for k, v in prior.items()}


def matchup(first, second, prior):
    a, b = _rates(first, prior), _rates(second, prior)
    strikes, takedowns, kos, subs, controls, reversals = [], [], [], [], [], []
    for own, opponent in [(a, b), (b, a)]:
        accuracy = .5 * (own['sig'] / own['sig_att'] + opponent['absorbed'] / opponent['absorbed_att'])
        strikes.append(max(own['sig_att'] * np.clip(accuracy, .01, .99), .001))
        td_accuracy = .5 * (own['td'] / own['td_att'] + opponent['td_allowed'] / opponent['td_allowed_att'])
        takedowns.append(max(own['td_att'] * np.clip(td_accuracy, .001, .999), .0001))
        kos.append(float(np.clip(np.sqrt(own['ko_for'] * opponent['ko_against']), .00001, .5)))
        # Attempts matter in addition to the observed finish/defence counts.
        sub_conversion = np.sqrt(own['sub_for'] * opponent['sub_against']) / max(prior['sub_att'], .0001)
        subs.append(float(np.clip(own['sub_att'] * sub_conversion, .00001, .5)))
        controls.append(float(np.clip(own['control'], .03, .75)))
        reversals.append(float(np.clip(own['reversals'], .0001, 1.)))
    scale = max(1., sum(controls) / .9)
    controls = [c / scale for c in controls]
    features = [np.log(strikes[0] / strikes[1]), np.log(takedowns[0] / takedowns[1]), controls[0] - controls[1]]
    return {'strikes': strikes, 'takedowns': takedowns, 'kos': kos, 'subs': subs,
            'controls': controls, 'reversals': reversals, 'decision_features': features}


def generator(rates):
    q = np.zeros((7, 7), dtype=float)
    td, control, rev = rates['takedowns'], rates['controls'], rates['reversals']
    q[0, 1], q[0, 2] = td
    q[1, 0], q[2, 0] = [min(td[i] / control[i], 5.) for i in range(2)]
    q[1, 2], q[2, 1] = min(rev[1] / control[0], 5.), min(rev[0] / control[1], 5.)
    for state in range(3):
        q[state, 3], q[state, 5] = rates['kos']
    q[1, 4] = min(rates['subs'][0] / control[0], 5.)
    q[2, 6] = min(rates['subs'][1] / control[1], 5.)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def probabilities(rates, coefficients, rounds):
    if rounds not in (3, 5):
        raise ValueError('Durée prévue requise : trois ou cinq rounds.')
    q = generator(rates)
    transition = expm(q * 5.)  # Rates are per minute; five minutes per round.
    state = np.array([1., 0., 0., 0., 0., 0., 0.])
    for index in range(rounds):
        state = state @ transition
        if index + 1 < rounds:
            state[0], state[1], state[2] = state[:3].sum(), 0., 0.
    survival = state[:3].sum()
    decision = float(expit(np.dot(coefficients, rates['decision_features'])))
    methods = np.array([state[3], state[4], survival * decision,
                        state[5], state[6], survival * (1 - decision)])
    if not np.isfinite(methods).all() or methods.min() < -1e-10:
        raise ValueError('Distribution Markov invalide.')
    methods = np.maximum(methods, 0.)
    methods /= methods.sum()
    return {'win_probabilities': [float(methods[:3].sum()), float(methods[3:].sum())],
            'method_probabilities': methods.tolist(), 'decision_probability': decision}


def _performance(row, side):
    other = 3 - side
    required = ['sig_lnd', 'sig_att', 'td_lnd', 'td_att', 'sub_att', 'rev', 'ctrl_secs']
    numbers = [row.get(f'p{s}_{k}') for s in (side, other) for k in required]
    if not np.isfinite(np.asarray(numbers, float)).all() or min(numbers) < 0:
        return None
    minutes = float(row['duration_secs']) / 60.
    if not np.isfinite(minutes) or not 0 < minutes <= 25:
        return None
    for s in (side, other):
        if row[f'p{s}_sig_lnd'] > row[f'p{s}_sig_att'] or row[f'p{s}_td_lnd'] > row[f'p{s}_td_att']:
            return None
    winner = 1 if row['y'] == 1 else 2 if row['y'] == 0 else None
    method = str(row['method']).upper()
    ko, sub = 'KO' in method or 'TKO' in method, method in {'SUB', 'SUBMISSION'}
    return {'minutes': minutes, 'sig': float(row[f'p{side}_sig_lnd']),
            'sig_att': float(row[f'p{side}_sig_att']), 'absorbed': float(row[f'p{other}_sig_lnd']),
            'absorbed_att': float(row[f'p{other}_sig_att']), 'td': float(row[f'p{side}_td_lnd']),
            'td_att': float(row[f'p{side}_td_att']), 'td_allowed': float(row[f'p{other}_td_lnd']),
            'td_allowed_att': float(row[f'p{other}_td_att']), 'sub_att': float(row[f'p{side}_sub_att']),
            'reversals': float(row[f'p{side}_rev']), 'control': min(float(row[f'p{side}_ctrl_secs']) / 60., minutes),
            'ko_for': float(ko and winner == side), 'ko_against': float(ko and winner == other),
            'sub_for': float(sub and winner == side), 'sub_against': float(sub and winner == other), 'fights': 1.}


def build_bundle(history, as_of, checked_at=None):
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert('UTC').tz_localize(None)
    cutoff = cutoff.normalize()
    data = history.copy()
    data['event_date'] = pd.to_datetime(data.event_date).dt.tz_localize(None)
    data = data[data.event_date.lt(cutoff) & data.event_date.ge(cutoff - pd.DateOffset(years=CONFIG['history_years']))]
    if data.fight_id.duplicated().any():
        raise ValueError('Combats historiques dupliqués.')
    states, populations, names = defaultdict(_empty), defaultdict(_empty), {}
    x, y, used = [], [], 0
    for _, card in data.sort_values(['event_date', 'fight_id']).groupby('event_date', sort=True):
        pending = []
        for row in card.to_dict('records'):
            identities = [str(row['fighter_1_id']), str(row['fighter_2_id'])]
            if any(not i or i == 'nan' for i in identities) or identities[0] == identities[1]:
                continue
            performance = [_performance(row, i) for i in (1, 2)]
            if any(p is None for p in performance):
                continue
            division = weight_class(row['weight_class'])
            # Canonical orientation, never the winner-first ordering of UFCStats.
            first, second = sorted(identities)
            a, b = states[first], states[second]
            method = str(row['method']).upper()
            if ('DEC' in method or method.startswith('DEC')) and row['y'] in (0., 1.) and min(a['fights'], b['fights']) >= CONFIG['minimum_fights']:
                rates = matchup(a, b, _prior(populations[division]))
                x.append(rates['decision_features'])
                winner = identities[0] if row['y'] == 1 else identities[1]
                y.append(float(winner == first))
            for i, stats in zip((1, 2), performance):
                identity = identities[i-1]
                names[identity] = str(row[f'fighter_{i}'])
                pending.append((identity, division, stats))
            used += 1
        # No fighter or population may see another result from the same card.
        for identity, division, stats in pending:
            for k in FIELDS:
                states[identity][k] += stats[k]
                populations[division][k] += stats[k]
    if len(y) < CONFIG['minimum_decisions']:
        raise ValueError('Historique insuffisant pour ajuster le modèle de décision.')
    features, labels = np.asarray(x), np.asarray(y)
    def objective(beta):
        logits = features @ beta
        return (np.logaddexp(0, logits).sum() - labels @ logits + .5 * CONFIG['decision_ridge'] * (beta @ beta),
                features.T @ (expit(logits) - labels) + CONFIG['decision_ridge'] * beta)
    fit = minimize(objective, np.zeros(3), method='L-BFGS-B', jac=True)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Ajustement du jugement non convergent.')
    bundle = {'version': VERSION, 'config': CONFIG, 'cutoff': cutoff.date().isoformat(),
              'checked_at': str(checked_at or pd.Timestamp.now(tz='UTC').isoformat()),
              'history_last_date': data.event_date.max().date().isoformat(), 'history_fights_used': used,
              'decision_training_count': len(y), 'decision_coefficients': fit.x.tolist(),
              'fighters': {k: {'name': names.get(k, ''), **v} for k, v in states.items() if k in names},
              'division_priors': {k: _prior(v) for k, v in populations.items()},
              'published_source': SOURCE, 'published_model_reproduced': False,
              'profitability_validated': False}
    bundle['sha256'] = fingerprint(bundle)
    return bundle


def verify_bundle(bundle):
    content = {k: v for k, v in bundle.items() if k != 'sha256'}
    if bundle.get('version') != VERSION or bundle.get('config') != CONFIG or fingerprint(content) != bundle.get('sha256'):
        raise ValueError('Paquet Markov absent, modifié ou incompatible : actualiser les données UFC.')


def score_fight(bundle, official, rounds, now):
    verify_bundle(bundle)
    stamp = pd.Timestamp(now)
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    checked = pd.Timestamp(bundle['checked_at'])
    if checked.tzinfo is None or not pd.Timedelta(0) <= stamp - checked <= pd.Timedelta(hours=48):
        raise ValueError('Vérification des résultats UFC périmée : actualiser les données.')
    fight_day = pd.Timestamp(official['event_date']).tz_localize(None).normalize()
    if fight_day < pd.Timestamp(bundle['cutoff']):
        raise ValueError('Le modèle contient des résultats postérieurs au début de cette carte.')
    fighters = [bundle['fighters'].get(official[f'fighter_{i}_id']) for i in (1, 2)]
    if any(f is None or f['fights'] < CONFIG['minimum_fights'] for f in fighters):
        raise ValueError('Au moins deux combats UFC avec statistiques requis par combattant.')
    division = weight_class(official['weight_class'])
    if division not in bundle['division_priors']:
        raise ValueError('Catégorie de poids absente de l’historique.')
    rates = matchup(*fighters, bundle['division_priors'][division])
    return {**probabilities(rates, bundle['decision_coefficients'], rounds),
            'fighter_names': [f['name'] for f in fighters], 'previous_fights': [int(f['fights']) for f in fighters],
            'rates': rates, 'model_sha256': bundle['sha256'], 'rounds': rounds}
