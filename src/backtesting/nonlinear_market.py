"""Frozen shallow tree corrections of market odds; exploratory, never deployment."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtesting.economic_selection import admitted, choose, gate
from src.backtesting.football_cross_market import file_hash
from src.backtesting.three_sport_residual import design, summarise


def repair_atp_context(frame, metadata):
    if metadata['match_id'].duplicated().any():
        raise ValueError('Ambiguous ATP context IDs')
    indexed = metadata.set_index('match_id')
    ids = frame['_source_row_id']
    if not ids.isin(indexed.index).all():
        raise ValueError('Missing ATP context IDs')
    context = indexed.loc[ids]
    indoor = context['indoor'].map({'Indoor': 1., 'Outdoor': 0.})
    rounds = context['round'].map({'1st Round': 1, '2nd Round': 2, '3rd Round': 3,
                                  '4th Round': 4, 'Quarterfinals': 5, 'Semifinals': 6,
                                  'The Final': 7, 'Round Robin': 4}) / 7
    if indoor.isna().any() or rounds.isna().any():
        raise ValueError('Unknown ATP indoor or round label')
    frame = frame.copy()
    audit = {}
    for column, values in [('is_indoor', indoor), ('round_progress', rounds)]:
        audit[column + '_rows_changed'] = int((frame[column].to_numpy() != values.to_numpy()).sum())
        frame[column] = values.to_numpy()
    return frame, audit


def inputs(frame, sport, protocol):
    x, q, odds, names = design(frame, sport, 'enriched')
    signs = -np.ones(x.shape[1])
    extra = {}
    if sport == 'atp':
        extra = {c: frame[c].to_numpy(float) for c in protocol['features']['atp_context']}
    elif sport == 'ufc':
        extra = {
            'minimum_experience': np.minimum(frame['experience_1'], frame['experience_2']),
            'mean_age': (frame['age_1'] + frame['age_2']) / 2,
            'maximum_layoff': np.maximum(frame['layoff_days_1'], frame['layoff_days_2']),
        }
    if extra:
        context = np.nan_to_num(np.column_stack(list(extra.values())), nan=0, posinf=0, neginf=0)
        x = np.column_stack([x, context])
        signs = np.r_[signs, np.ones(len(extra))]
        names = names + list(extra)
    return x, q, odds, names, signs


def recency_weights(dates, cutoff, half_life_years):
    age = (pd.Timestamp(cutoff) - pd.DatetimeIndex(dates)).total_seconds().to_numpy() / 86400
    if not len(age) or not np.isfinite(age).all() or (age <= 0).any():
        raise ValueError('Training dates must be strictly before the fold cutoff')
    if half_life_years is None:
        return np.ones(len(age))
    if half_life_years <= 0:
        raise ValueError('Half-life must be positive')
    weights = np.exp2(-age / (365.25 * half_life_years))
    return weights / weights.mean()


def market_margin(q):
    if not np.isfinite(q).all() or (q <= 0).any() or not np.allclose(q.sum(axis=1), 1):
        raise ValueError('Invalid market probability vector')
    return np.log(q[:, 0] / q[:, 1]) if q.shape[1] == 2 else np.log(q)


class OffsetTrees:
    def __init__(self, parameters, rounds, signs):
        self.parameters = dict(parameters)
        self.rounds = rounds
        self.signs = np.asarray(signs)

    def fit(self, x, q, labels, weights):
        self.classes = q.shape[1]
        self.features = x.shape[1]
        if self.classes not in (2, 3) or len(self.signs) != self.features:
            raise ValueError('Unexpected outcome or feature count')
        if not np.isin(labels, np.arange(self.classes)).all():
            raise ValueError('Invalid outcome labels')
        parameters = dict(self.parameters)
        if self.classes == 2:
            # XGBoost probability is P(target=1), deliberately mapped to outcome zero.
            y = (labels == 0).astype(int)
            train_x = np.vstack([x, x * self.signs])
            train_q = np.vstack([q, q[:, ::-1]])
            y = np.r_[y, 1 - y]
            weights = np.r_[weights, weights] / 2
            parameters['objective'] = 'binary:logistic'
        else:
            train_x, train_q, y = x, q, labels
            parameters.update(objective='multi:softprob', num_class=self.classes)
        matrix = xgb.DMatrix(train_x, label=y, weight=weights,
                             base_margin=market_margin(train_q), nthread=1)
        self.model = xgb.train(parameters, matrix, num_boost_round=self.rounds)
        return self

    def _predict(self, x, q):
        matrix = xgb.DMatrix(x, base_margin=market_margin(q), nthread=1)
        return self.model.predict(matrix).astype(float)

    def predict(self, x, q):
        if q.shape[1] != self.classes or x.shape[1] != self.features:
            raise ValueError('Prediction dimensions differ from training')
        p = self._predict(x, q)
        if self.classes == 2:
            p0 = (p + 1 - self._predict(x * self.signs, q[:, ::-1])) / 2
            p = np.column_stack([p0, 1 - p0])
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return p / p.sum(axis=1, keepdims=True)


def blend_market(p, q, correction_fraction):
    if not 0 <= correction_fraction <= 1:
        raise ValueError('Correction fraction must be in [0, 1]')
    return q + correction_fraction * (p - q)


def predict_fold(frame, sport, year, half_life, protocol):
    cutoff = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=protocol['embargo_days'])
    train = frame['_date'].lt(cutoff) & frame['_status'].eq('completed')
    test = frame['_date'].dt.year.eq(year)
    if train.sum() < protocol['sports'][sport]['min_train'] or not test.any():
        raise ValueError(f'Insufficient fold: {sport}/{year}')
    x, q, odds, names, signs = inputs(frame, sport, protocol)
    labels = frame['_label'].to_numpy(int)
    weights = recency_weights(frame.loc[train, '_date'], cutoff, half_life)
    model = OffsetTrees(protocol['booster'], protocol['boost_rounds'], signs)
    model.fit(x[train], q[train], labels[train], weights)
    p = model.predict(x[test], q[test])
    rows = frame.loc[test, ['_source_row_id', '_date', '_status', '_label']].reset_index(drop=True).copy()
    rows['year'] = year
    for j in range(q.shape[1]):
        rows[f'p{j}'], rows[f'q{j}'], rows[f'price_{j}'] = p[:, j], q[test, j], odds[test, j]
    fold = {'sport': sport, 'year': year, 'half_life_years': half_life,
            'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
            'train_max': str(frame.loc[train, '_date'].max().date()),
            'cutoff_exclusive': str(cutoff.date()), 'features': names,
            'swap_signs': signs.tolist(),
            'effective_sample_size': float(weights.sum() ** 2 / (weights ** 2).sum())}
    return rows, fold


def candidate_predictions(rows, candidate, fraction):
    rows = rows.copy()
    classes = len([c for c in rows if c.startswith('price_')])
    q = rows[[f'q{j}' for j in range(classes)]].to_numpy()
    p = blend_market(rows[[f'p{j}' for j in range(classes)]].to_numpy(), q, fraction)
    for j in range(classes):
        rows[f'p{j}'] = p[:, j]
    rows['candidate'] = candidate
    complete = rows['_status'].eq('completed')
    label = rows['_label'].to_numpy(int)
    rows['model_loss'] = np.where(complete, -np.log(p[np.arange(len(p)), label]), np.nan)
    rows['market_loss'] = np.where(complete, -np.log(q[np.arange(len(q)), label]), np.nan)
    return rows


def run(root, progress=print):
    folder = root / 'models/nonlinear_market'
    protocol_path = folder / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if (folder / 'report.json').exists():
        raise FileExistsError('Preserve the completed experiment; use a separately declared replication')
    for path, expected in protocol['frozen_files'].items():
        if file_hash(root / path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    manifest = {'protocol_sha256': file_hash(protocol_path),
                'implementation_sha256': file_hash(Path(__file__)),
                'xgboost_version': xgb.__version__, 'numpy_version': np.__version__,
                'pandas_version': pd.__version__, 'real_money_authorised': False}
    # Seal implementation and runtime before any tuning returns are computed.
    (folder / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    reports = {}
    for sport, spec in protocol['sports'].items():
        target = folder / sport
        target.mkdir(exist_ok=True)
        source = root / f'models/three_sport_residual/{sport}_features.parquet'
        frame = pd.read_parquet(source)
        if frame['_source_row_id'].duplicated().any():
            raise ValueError('Duplicate source match IDs')
        if sport == 'ufc' and frame['_date'].dt.year.max() >= 2025:
            raise ValueError('Reserved UFC holdout must remain closed')
        quality = {}
        if sport == 'atp':
            metadata = pd.read_csv(root / 'models/three_sport_residual/atp_matched_master.csv.gz',
                                   usecols=['match_id', 'indoor', 'round'])
            frame, quality = repair_atp_context(frame, metadata)
        quality['context_repair_before_tuning'] = True
        cache, folds = {}, []

        def predictions(candidate, years):
            setting = protocol['candidates'][candidate]
            batches = []
            for year in years:
                key = (setting['half_life_years'], year)
                if key not in cache:
                    progress(f'{sport}: ajustement {year}, demi-vie {key[0]}')
                    cache[key], fold = predict_fold(frame, sport, year, key[0], protocol)
                    folds.append(fold)
                batches.append(candidate_predictions(cache[key], candidate, setting['correction_fraction']))
            return pd.concat(batches, ignore_index=True)

        scores = []
        for candidate in protocol['candidates']:
            tuning = predictions(candidate, spec['tuning'])
            tuning.to_parquet(target / (candidate + '_tuning_predictions.parquet'), index=False)
            for threshold in protocol['thresholds']:
                summary, _ = summarise(tuning, threshold, 'flat', protocol)
                scores.append({'model': candidate, 'threshold': threshold, 'summary': summary})
            progress(f'{sport}: reglage examine pour {candidate}')
        chosen = choose(scores)
        admission = chosen is not None and admitted(chosen['summary'])
        lock = {'chosen': chosen, 'scores': scores, 'tuning_admitted': admission, **manifest}
        (target / 'selection_lock.json').write_text(json.dumps(lock, indent=2, allow_nan=False) + '\n')
        evaluations = {}
        if chosen is not None:
            progress(f"{sport}: choix fige {chosen['model']}, seuil {chosen['threshold']}; admission={admission}")
            evaluated = predictions(chosen['model'], spec['evaluation'])
            evaluated.to_parquet(target / 'evaluation_predictions.parquet', index=False)
            settings = deepcopy(protocol)
            settings['uncertainty']['samples'] = protocol['uncertainty']['evaluation_samples']
            for stake in ['flat', 'quarter_kelly']:
                summary, bets = summarise(evaluated, chosen['threshold'], stake, settings)
                summary['research_gate_passed'] = gate(summary, admission)
                bets.attrs = {}
                bets.to_parquet(target / (stake + '_bets.parquet'), index=False)
                evaluations[stake] = summary
                progress(f"{sport}/{stake}: {summary['settled']} regles; ROI2={summary['roi']['0.02']}")
        status = ('NO_ELIGIBLE_TUNING_PAIR' if chosen is None else
                  'PAPER_REPLICATION_ONLY' if evaluations['flat']['research_gate_passed'] else
                  'NO_ROBUST_CANDIDATE')
        report = {'sport': sport, 'status': status, 'selection': lock, 'evaluation': evaluations,
                  'folds': folds, 'quality': quality, 'source_features_sha256': file_hash(source), **manifest,
                  'evidence': protocol['evidence'], 'ufc_2025_onward_evaluated': False}
        (target / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        reports[sport] = report
    (folder / 'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')
    return reports
