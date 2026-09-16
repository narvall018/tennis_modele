"""One-fixture replay of the frozen phase-4 statistics without 70k feature rows.

All-player Elo and surface totals are replayed, but only the requested players'
rolling histories and the requested court's observations need to be retained.
Feature formula parity is tested against the original historical builder.
"""
from collections import defaultdict
import math

import numpy as np
import pandas as pd

from src.backtesting.tennis_phase4 import (
    PlayerHistory, CourtHistory, SURFACES, SURFACE_INTERACTIONS_BASE,
    _player_key, _normalise_text, _extract_match_stats, _surface_prior,
    _player_rates, _safe_logit, _round_progress, _level_strength,
)


def one_fixture_features(history, row):
    day = pd.Timestamp(row['match_date']).date()
    surface = row['surface']
    tracked = {_player_key(row[f'player_{s}_id'], row[f'player_{s}_name']) for s in [1, 2]}
    if len(tracked) != 2:
        raise ValueError('Les deux noms désignent la même identité de joueur.')
    states = defaultdict(PlayerHistory)
    totals = defaultdict(lambda: [0., 0.])
    court = CourtHistory()
    court_key = (_normalise_text(row['tourney_name']), surface)
    past = history[pd.to_datetime(history['match_date']).dt.date < day].sort_values(['match_date', 'match_id'], kind='mergesort')
    for values in past.itertuples(index=False, name=None):
        match = dict(zip(past.columns, values))
        played = pd.Timestamp(match['match_date']).date()
        keys = [_player_key(match.get(f'player_{s}_id'), match.get(f'player_{s}_name')) for s in [1, 2]]
        first, second = [states[key] for key in keys]
        played_surface = match['surface'] if match['surface'] in SURFACES else 'Unknown'
        status, label = match['match_status'], int(match['player_1_won'])
        if status == 'completed':
            expected = 1 / (1 + 10 ** ((second.global_elo - first.global_elo) / 400))
            change = 24 * (label - expected)
            first.global_elo += change; second.global_elo -= change
            expected_surface = 1 / (1 + 10 ** ((second.surface_elo.get(played_surface, 1500.) - first.surface_elo.get(played_surface, 1500.)) / 400))
            change_surface = 32 * (label - expected_surface)
            first.surface_elo[played_surface] = first.surface_elo.get(played_surface, 1500.) + change_surface
            second.surface_elo[played_surface] = second.surface_elo.get(played_surface, 1500.) - change_surface
        stats = _extract_match_stats(match) if status == 'completed' else None
        minutes = float(pd.to_numeric(match.get('minutes'), errors='coerce'))
        for index, (key, player) in enumerate(zip(keys, [first, second])):
            if key not in tracked:
                continue
            if status == 'completed':
                player.outcomes.append((played, played_surface, bool(label if index == 0 else not label)))
            if np.isfinite(minutes) and minutes > 0 and status not in {'walkover', 'defaulted'}:
                player.workload.append((played, minutes))
            if status in {'completed', 'retired'}:
                player.retirement_exposures += 1
                if status == 'retired' and (index == 1 if label == 1 else index == 0):
                    player.retirements += 1
            if stats is not None:
                player.stats.append((played, played_surface, stats[index]))
        if stats is not None:
            won = stats[0]['serve_won'] + stats[1]['serve_won']
            total = stats[0]['serve_total'] + stats[1]['serve_total']
            totals[played_surface][0] += won; totals[played_surface][1] += total
            if (_normalise_text(match['tourney_name']), played_surface) == court_key:
                court.observations.append((played, won, total))

    p1, p2 = [states[_player_key(row[f'player_{s}_id'], row[f'player_{s}_name'])] for s in [1, 2]]
    prior = _surface_prior(surface, totals)
    court_rate, court_samples = court.serve_rate(day, prior)
    pace = _safe_logit(court_rate) - _safe_logit(prior)
    r1, r2 = [_player_rates(player, day, surface, prior) for player in [p1, p2]]
    serve1 = float(np.clip(prior + (r1['serve'] - prior) - (r2['return'] - (1 - prior)), .45, .80))
    serve2 = float(np.clip(prior + (r2['serve'] - prior) - (r1['return'] - (1 - prior)), .45, .80))
    odds1, odds2 = [float(row[f'player_{s}_odds']) for s in [1, 2]]
    overround = 1 / odds1 + 1 / odds2
    market = _safe_logit((1 / odds1) / overround)
    def numeric(side, field, default, positive=False):
        value = float(pd.to_numeric(row.get(f'player_{side}_{field}'), errors='coerce'))
        return value if np.isfinite(value) and (not positive or value > 0) else default
    ranks = [numeric(s, 'rank', 500., True) for s in [1, 2]]
    points = [max(0., numeric(s, 'rank_points', 0.)) for s in [1, 2]]
    ages = [numeric(s, 'age', 27.) for s in [1, 2]]
    heights = [numeric(s, 'ht', 185.) for s in [1, 2]]
    workloads = {n: [p.workload_totals(day, n) for p in [p1, p2]] for n in [3, 7, 14]}
    features = {
        'market_logit': market,
        'global_elo_diff': (p1.global_elo - p2.global_elo) / 400,
        'surface_elo_diff': (p1.surface_elo.get(surface, 1500.) - p2.surface_elo.get(surface, 1500.)) / 400,
        'form_10_logit_diff': _safe_logit(p1.win_rate(day)) - _safe_logit(p2.win_rate(day)),
        'surface_form_logit_diff': _safe_logit(p1.win_rate(day, surface, 20)) - _safe_logit(p2.win_rate(day, surface, 20)),
        'rank_log_advantage': math.log1p(ranks[1]) - math.log1p(ranks[0]),
        'points_log_advantage': math.log1p(points[0]) - math.log1p(points[1]),
        'age_diff': (ages[0] - ages[1]) / 10,
        'peak_age_advantage': (abs(ages[1] - 27.) - abs(ages[0] - 27.)) / 10,
        'height_diff': (heights[0] - heights[1]) / 20,
        'left_handed_diff': float(str(row.get('player_1_hand')) == 'L') - float(str(row.get('player_2_hand')) == 'L'),
        'rest_diff': (p1.days_rest(day) - p2.days_rest(day)) / 30,
        'matches_14d_diff': float(workloads[14][0][0] - workloads[14][1][0]) / 5,
        'serve_advantage': _safe_logit(serve1) - _safe_logit(serve2),
        'serve_sample_log_diff': math.log1p(r1['serve_total']) - math.log1p(r2['serve_total']),
        'retirement_risk_diff': p1.retirement_rate() - p2.retirement_rate(),
    }
    for column, rate in [('serve_skill', 'serve'), ('return_skill', 'return'), ('ace_rate', 'ace'),
                         ('double_fault_rate', 'df'), ('first_in', 'first_in'), ('first_won', 'first_won'),
                         ('second_won', 'second_won'), ('break_save', 'bp_save')]:
        features[column + '_logit_diff'] = _safe_logit(r1[rate]) - _safe_logit(r2[rate])
    for n, scale in [(3, 300.), (7, 600.), (14, 1000.)]:
        features[f'minutes_{n}d_diff'] = (workloads[n][0][1] - workloads[n][1][1]) / scale
    for name in ['hard', 'clay', 'grass']:
        flag = float(surface.lower() == name)
        features['is_' + name] = flag
        features['market_logit_x_' + name] = market * flag
        for base in SURFACE_INTERACTIONS_BASE:
            features[base + '_x_' + name] = features[base] * flag
    for name in ['serve_advantage', 'height_diff', 'market_logit']:
        features[name + '_x_court_pace'] = features[name] * pace
    features.update(is_indoor=float(str(row['indoor']).upper() in {'I', '1', 'TRUE'}),
                    best_of_5=float(row['best_of'] == 5), round_progress=_round_progress(row['round']),
                    level_strength=_level_strength(row['tourney_level']), court_pace=pace,
                    log_surface_samples=math.log1p(court_samples),
                    _p1_serve_sample=r1['serve_total'], _p2_serve_sample=r2['serve_total'],
                    _date=pd.Timestamp(row['match_date']), _match_id=row['match_id'])
    return pd.DataFrame([features])
