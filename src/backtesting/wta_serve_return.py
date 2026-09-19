"""Frozen WTA single-price study of conservatively delayed service statistics.

No current-match result is used to link identities or build predictors. Stored
tournament start dates are NOT match dates or verified publication timestamps.
"""
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import logit

from src.backtesting.football_cross_market import file_hash
from src.backtesting.wta_price_residual import fit_symmetric, predict_symmetric
from src.backtesting.wta_single_book import summarise, gate

STUDY = 'wta_serve_return_2026_09_18'
SOURCES = ['data/wta_tennis.csv', 'data/raw/tennis_mylife/wta_matches_2000_current.csv.gz']
MATCH_COLUMNS = ['Date', 'Tournament', 'Series', 'Surface', 'Best of', 'Round',
                 'Player_1', 'Player_2', 'Winner', 'Status', 'B365_1', 'B365_2']
STATS_COLUMNS = ['tourney_date', 'tourney_id', 'match_num', 'surface', 'best_of', 'score',
                 'winner_id', 'loser_id', 'winner_name', 'loser_name',
                 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
CANDIDATES = ['price_control', 'serve_global', 'serve_surface']
SERIES = {'International', 'Premier', 'Grand Slam', 'WTA250', 'WTA500', 'WTA1000',
          'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tour Championships'}
SURFACES = {'Hard', 'Clay', 'Grass'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def register(root):
    root = Path(root)
    protocol = {'study': STUDY, 'registered_at': datetime.now(timezone.utc).isoformat(),
        'sources': {p: file_hash(root/p) for p in SOURCES},
        'hypothesis': 'Prior service and return point success, globally and by surface, may add '
                      'information to single-book calibration beyond earlier win/loss form models.',
        'candidates': CANDIDATES, 'match_columns': MATCH_COLUMNS, 'statistics_columns': STATS_COLUMNS,
        'stats_first_year': 2007, 'match_first_year': 2010,
        'tuning_years': [2021, 2022], 'evaluation_years': [2023, 2024, 2025],
        'fit': 'Annual expanding history, dates strictly before January1 minus7days. Symmetric '
               'ridge logistic correction to same-book market; scaling learned on training only.',
        'embargo_days': 7, 'penalty': .01, 'min_train': 3000,
        'statistics_delay_days': 28, 'statistics_lookback_days': 365, 'half_life_days': 180.,
        'prior_points': 500., 'serve_prior': .60, 'return_prior': .40, 'minimum_history_matches': 5,
        'stats_timing': 'A statistic enters only when tournament_start+28days is STRICTLY before '
                        'prediction date; retain starts within last365days. No intra-tournament updates. '
                        'Conservative availability assumption, not verified publication timestamps.',
        'stats_validation': 'Only best_of3, known surface, no retirement/walkover/unfinished score, '
                            'finite integer service counts: 0<=firstWon<=firstIn<=servicePoints, '
                            '0<=secondWon<=servicePoints-firstIn. Both players required.',
        'identity': 'No outcome-based match join. Registry of IDs from delayed prior records only. '
                    'Full-name surname suffixes plus first initial compared to complete abbreviated '
                    'surname and first initial. Reject keys with zero or multiple historical IDs, '
                    'or both participants mapped to same ID. No fuzzy similarity or future registry.',
        'predictors': 'Control: same-book logit, signed-quadratic logit, logit*overround. '
                      'Global adds differences of smoothed service logit, return logit, log1p point count. '
                      'Surface adds same three on current surface; missing surface history uses fixed priors. '
                      'All three evaluated on identical past-history eligible population. No current ranking.',
        'market': 'Historical B365_1/B365_2 only. This is the execution-price input, not a required '
                  'external reference bookmaker. French substitution remains unvalidated.',
        'selection': {'minimum_ev_after_haircut': .03, 'profit_haircut': .02,
                      'odds_min': 1.3, 'odds_max': 5., 'flat_fraction': .0025,
                      'maximum_daily_exposure': .02, 'sensitivity_haircuts': [0., .02, .05]},
        'staking': 'Separate1000 initial bankroll per candidate/period; .25% day-start, 2% daily '
                   'cap, floor cents, no intraday reinvestment, deterministic date/tournament/name order.',
        'settlement': 'Completed pays winner; retired/walkover/cancelled/defaulted/unfinished/unknown '
                      'assumed void, included in quota. Actual bookmaker retirement rules not verified.',
        'choice': 'Highest tuning95% lower among >=50settled, declaration order breaks ties. '
                  'Admit only positive tuningROI and lower; persist lock before diagnostic evaluation.',
        'uncertainty': {'samples': 100000, 'seed': 20260918, 'family_size': 3,
                        'lower_quantile_per_candidate': .05/3,
                        'scope': 'Only three variants, not all previous adaptive research; circular3month blocks.'},
        'gate': 'Chosen and admitted at tuning, >=200settled, positiveROI2/5%, familylower>0, '
                '>=2positiveyears, positiveROIwithoutbestyear. No real-money deployment even if passed.',
        'evidence': 'EXPLORATORY_REUSED_HISTORY_NOT_INDEPENDENT_VALIDATION',
        'no_2026_fitting_or_evaluation': True, 'ufc_reserve': 'NOT OPENED',
        'real_money_authorised': False, 'app_modified': False,
        'limitations': ['Historical years repeatedly explored; not an independent holdout.',
                       'No verified historical publication time or French accepted odds.',
                       'Delayed name registry can miss aliases or unresolved real-world homonyms.',
                       'Service/return rates not opponent-adjusted; surface samples may be sparse.',
                       'Fixed smoothing and 28day delay are modelling assumptions, not optimised findings.',
                       'No data update, live integration, automatic schedule or actual bet.']}
    dependencies = ['src/backtesting/wta_serve_return.py', 'scripts/run_wta_serve_return.py',
                    'src/backtesting/football_cross_market.py', 'src/backtesting/wta_price_residual.py',
                    'src/backtesting/wta_single_book.py']
    folder = root/'models'/STUDY
    folder.mkdir(exist_ok=False)
    write_json(folder/'protocol.json', protocol)
    write_json(folder/'registration.json', {'protocol_sha256': file_hash(folder/'protocol.json'),
        'implementations': {p: file_hash(root/p) for p in dependencies}, 'before_new_returns': True})
    return folder


def tokens(value):
    plain = unicodedata.normalize('NFKD', str(value))
    return re.findall('[a-z]+', ''.join(c for c in plain if not unicodedata.combining(c)).lower())


def full_keys(name):
    words = tokens(name)
    return [' '.join(words[i:])+'|'+words[0][0] for i in range(1, len(words))]


def short_key(name):
    words = tokens(name)
    return ' '.join(words[:-1])+'|'+words[-1][0] if len(words) >= 2 else ''


def prepare_stats(raw, protocol):
    rows = raw[STATS_COLUMNS].copy()
    rows['_start'] = pd.to_datetime(rows.tourney_date.astype(str), format='%Y%m%d', errors='raise')
    rows = rows[rows._start.dt.year.between(protocol['stats_first_year'], 2025)].copy()
    if rows.duplicated(['tourney_id', 'match_num']).any():
        raise ValueError('Duplicate source statistics identity')
    # Records with missing/broken points still contribute to the identity registry.
    rows = rows.dropna(subset=['winner_id', 'loser_id', 'winner_name', 'loser_name'])
    rows = rows[rows.winner_id.ne(rows.loser_id)].copy()
    valid = rows.surface.isin(SURFACES) & rows.best_of.eq(3)
    valid &= ~rows.score.fillna('').astype(str).str.contains(r'[A-Za-z]|^\s*$', regex=True)
    for side in ['w', 'l']:
        fields = [side+'_'+s for s in ['svpt', '1stIn', '1stWon', '2ndWon']]
        counts = rows[fields].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        points, firstin, firstwon, secondwon = counts.T
        valid &= (np.isfinite(counts).all(axis=1) & (counts >= 0).all(axis=1)
                  & (counts == np.floor(counts)).all(axis=1) & (points > 0)
                  & (firstin <= points) & (firstwon <= firstin) & (secondwon <= points-firstin))
        rows[side+'_points'] = points
        rows[side+'_won'] = firstwon+secondwon
    rows['_valid'] = valid
    rows['_available'] = rows._start+pd.Timedelta(days=protocol['statistics_delay_days'])
    return rows.sort_values(['_available', 'tourney_id', 'match_num']).reset_index(drop=True)


def prepare_matches(raw, protocol):
    rows = raw[MATCH_COLUMNS].copy()
    rows['_source_row_id'] = np.arange(len(rows))
    rows['_date'] = pd.to_datetime(rows.Date, errors='raise')
    rows = rows[rows._date.dt.year.between(protocol['match_first_year'], 2025)].copy()
    if rows[['Player_1', 'Player_2', 'Tournament', '_date']].isna().any().any():
        raise ValueError('Missing match identity')
    pair = ['|'.join(sorted([a, b])) for a, b in zip(rows.Player_1, rows.Player_2)]
    if rows.assign(pair=pair).duplicated(['_date', 'pair']).any() or rows.Player_1.eq(rows.Player_2).any():
        raise ValueError('Duplicate or ambiguous match')
    if not rows.Status.isin(['completed', 'retired', 'walkover', 'cancelled', 'defaulted', 'unfinished', 'unknown']).all():
        raise ValueError('Unknown settlement status')
    if not (rows.Winner.eq(rows.Player_1) | rows.Winner.eq(rows.Player_2) | rows.Status.ne('completed')).all():
        raise ValueError('Winner identity mismatch')
    rows = rows[rows.Series.isin(SERIES) & rows.Surface.isin(SURFACES) & rows['Best of'].eq(3)].copy()
    rows = rows.rename(columns={'Player_1': '_p1', 'Player_2': '_p2', 'Tournament': '_tournament', 'Status': '_status'})
    rows['_label'] = np.where(rows.Winner.eq(rows._p1), 0, 1)
    rows['price_0'], rows['price_1'] = rows.B365_1, rows.B365_2
    return rows.sort_values(['_date', '_source_row_id']).reset_index(drop=True)


def profile(history, date, surface, protocol):
    selected = [r for r in history if surface is None or r[1] == surface]
    if selected:
        ages = np.array([(date-r[0]).days for r in selected])
        weights = np.exp2(-ages/protocol['half_life_days'])
        # serve won, service points, return won, return points
        sums = weights@np.asarray([r[2:] for r in selected], dtype=float)
    else:
        sums = np.zeros(4)
    prior = protocol['prior_points']
    serve = (sums[0]+prior*protocol['serve_prior'])/(sums[1]+prior)
    ret = (sums[2]+prior*protocol['return_prior'])/(sums[3]+prior)
    return np.array([logit(serve), logit(ret), np.log1p(sums[1]+sums[3])])


def build_features(matches, statistics, protocol):
    registry, histories = defaultdict(set), defaultdict(deque)
    records = statistics.to_dict('records'); cursor = 0; outputs = []
    quality = defaultdict(int)
    for date, day in matches.groupby('_date', sort=True):
        while cursor < len(records) and records[cursor]['_available'] < date:
            row = records[cursor]; cursor += 1
            for side, opponent, prefix in [('w', 'l', 'winner'), ('l', 'w', 'loser')]:
                identity = row[prefix+'_id']
                for key in full_keys(row[prefix+'_name']): registry[key].add(identity)
                if row['_valid']:
                    histories[identity].append((row['_start'], row['surface'], row[side+'_won'],
                        row[side+'_points'], row[opponent+'_points']-row[opponent+'_won'], row[opponent+'_points']))
        for row in day.to_dict('records'):
            identities = [registry.get(short_key(row[p]), set()) for p in ['_p1', '_p2']]
            if any(len(ids) != 1 for ids in identities):
                quality['blocked_identity'] += 1; continue
            a, b = [next(iter(ids)) for ids in identities]
            if a == b:
                quality['blocked_identity'] += 1; continue
            for identity in [a, b]:
                queue = histories[identity]
                while queue and queue[0][0] < date-pd.Timedelta(days=protocol['statistics_lookback_days']):
                    queue.popleft()
            if min(len(histories[a]), len(histories[b])) < protocol['minimum_history_matches']:
                quality['blocked_history'] += 1; continue
            output = {**row, 'id_1': a, 'id_2': b,
                      'history_count_1': len(histories[a]), 'history_count_2': len(histories[b]),
                      'stats_max_start': max(histories[a][-1][0], histories[b][-1][0])}
            for name, surface in [('global', None), ('surface', row['Surface'])]:
                difference = profile(histories[a], date, surface, protocol)-profile(histories[b], date, surface, protocol)
                for suffix, value in zip(['serve', 'return', 'volume'], difference):
                    output[name+'_'+suffix] = value
            outputs.append(output)
    if not outputs: raise ValueError('No matches with unambiguous prior statistics')
    rows = pd.DataFrame(outputs)
    prices = rows[['price_0', 'price_1']].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        margin = (1/prices).sum(axis=1)
    valid = np.isfinite(prices).all(axis=1) & (prices > 1).all(axis=1) & (margin >= 1) & (margin <= 1.2)
    quality['blocked_price'] = int((~valid).sum()); quality['eligible'] = int(valid.sum())
    rows = rows.loc[valid].reset_index(drop=True)
    quality['by_year'] = {str(y): int(n) for y, n in rows.groupby(rows._date.dt.year).size().items()}
    quality['statistics_valid_rows'] = int(statistics._valid.sum())
    return rows, dict(quality)


def arrays(frame, candidate):
    if candidate not in CANDIDATES: raise ValueError('Unregistered candidate')
    prices = frame[['price_0', 'price_1']].to_numpy(float)
    inv = 1/prices; margin = inv.sum(axis=1); q = inv/margin[:, None]
    market_logit = np.log(q[:, 0]/q[:, 1])
    columns = [market_logit, market_logit*np.abs(market_logit), market_logit*margin]
    if candidate != 'price_control':
        columns.extend(frame['global_'+s].to_numpy(float) for s in ['serve', 'return', 'volume'])
    if candidate == 'serve_surface':
        columns.extend(frame['surface_'+s].to_numpy(float) for s in ['serve', 'return', 'volume'])
    return np.column_stack(columns), q


def predict_year(frame, candidate, year, protocol):
    cutoff = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=protocol['embargo_days'])
    train = frame._date.lt(cutoff) & frame._status.eq('completed')
    test = frame._date.dt.year.eq(year)
    if train.sum() < protocol['min_train'] or not test.any(): raise ValueError('Insufficient annual fold')
    x, q = arrays(frame, candidate); labels = frame._label.to_numpy(int)
    model = fit_symmetric(x[train], q[train], labels[train], protocol['penalty'])
    p = predict_symmetric(model, x[test], q[test])
    rows = frame.loc[test].copy(); rows['year'] = year
    rows['p0'], rows['p1'] = p[:, 0], p[:, 1]
    ix = np.arange(len(rows)); y = labels[test]
    rows['model_loss'], rows['market_loss'] = -np.log(p[ix, y]), -np.log(q[test][ix, y])
    rows.loc[rows._status.ne('completed'), ['model_loss', 'market_loss']] = np.nan
    return rows, {'candidate': candidate, 'year': year, 'train_rows': int(train.sum()),
                  'train_max_date': str(frame.loc[train, '_date'].max().date()),
                  'cutoff_exclusive': str(cutoff.date()), 'features': x.shape[1],
                  'test_rows': int(test.sum()), 'test_completed': int(rows._status.eq('completed').sum())}


def run(root, progress=print):
    root = Path(root); folder = root/'models'/STUDY
    protocol = json.loads((folder/'protocol.json').read_text())
    registration = json.loads((folder/'registration.json').read_text())
    if (folder/'run_started.json').exists(): raise FileExistsError('Existing study preserved; no rerun')
    for name, digest in {**registration['implementations'], **protocol['sources'],
            str((folder/'protocol.json').relative_to(root)): registration['protocol_sha256']}.items():
        if file_hash(root/name) != digest: raise ValueError(f'Frozen input changed: {name}')
    write_json(folder/'run_started.json', {'at': datetime.now(timezone.utc).isoformat()})
    stats = prepare_stats(pd.read_csv(root/SOURCES[1], usecols=STATS_COLUMNS, low_memory=False), protocol)
    matches = prepare_matches(pd.read_csv(root/SOURCES[0], usecols=MATCH_COLUMNS), protocol)
    progress('WTA : reconstruction des statistiques de service/retour disponibles avant chaque match…')
    frame, quality = build_features(matches, stats, protocol)
    frame.to_parquet(folder/'features.parquet', index=False)
    write_json(folder/'quality.json', quality)
    folds = []
    def evaluate(candidate, years, period):
        batches = []
        for year in years:
            progress(f'WTA {candidate} : entraînement antérieur, année {year}…')
            rows, fold = predict_year(frame, candidate, year, protocol)
            batches.append(rows); folds.append(fold)
        predictions = pd.concat(batches, ignore_index=True)
        summary, bets = summarise(predictions, protocol)
        summary['cash_roi_2pct'] = float(bets.profit_cash.sum()/bets.stake_cash.sum()) if len(bets) else None
        predictions.to_parquet(folder/f'{candidate}_{period}_predictions.parquet', index=False)
        bets.attrs = {}; bets.to_parquet(folder/f'{candidate}_{period}_bets.parquet', index=False)
        return summary
    tuning = {c: evaluate(c, protocol['tuning_years'], 'tuning') for c in CANDIDATES}
    eligible = [c for c in CANDIDATES if tuning[c]['settled'] >= 50 and tuning[c]['uncertainty']['ci95'][0] is not None]
    chosen = max(eligible, key=lambda c: tuning[c]['uncertainty']['ci95'][0]) if eligible else None
    admitted = bool(chosen and (tuning[chosen]['roi']['0.02'] or 0) > 0 and tuning[chosen]['uncertainty']['ci95'][0] > 0)
    lock = {'chosen': chosen, 'admitted': admitted, 'tuning': tuning, 'at': datetime.now(timezone.utc).isoformat()}
    write_json(folder/'selection_lock.json', lock)
    evaluations = {}
    for candidate in CANDIDATES:
        result = evaluate(candidate, protocol['evaluation_years'], 'evaluation')
        result['checks'] = gate(result, admitted and candidate == chosen)
        result['research_gate_passed'] = all(result['checks'].values()); evaluations[candidate] = result
        progress(f"{candidate}: {result['settled']} réglés, ROI={result['roi']['0.02']}, filtre={result['research_gate_passed']}")
    report = {'study': STUDY, 'selection': lock, 'evaluation': evaluations, 'quality': quality, 'folds': folds,
              'evidence': protocol['evidence'], 'real_money_authorised': False, 'app_modified': False,
              'limitations': protocol['limitations']}
    write_json(folder/'report.json', report)
    return report
