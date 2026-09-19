"""Bounded three-sport random-Fourier interaction experiment, never live picks."""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from src.backtesting.football_cross_market import file_hash
from src.backtesting.three_sport_residual import design
from src.backtesting.wta_serve_return import arrays as serve_arrays
from src.backtesting.wta_price_residual import select
from src.backtesting.wta_single_book import summarise, gate

STUDY = 'kernel_matchups_2026_09_19'
SOURCES = {'football': 'models/three_sport_residual/football_features.parquet',
           'ufc': 'models/three_sport_residual/ufc_features.parquet',
           'wta': 'models/wta_serve_return_2026_09_18/features.parquet'}
CANDIDATES = {'kernel_flexible': .001, 'kernel_strong': .01}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def register(root):
    root = Path(root)
    prior = json.loads((root/'models/nonlinear_market/protocol.json').read_text())
    for name in [SOURCES['football'], SOURCES['ufc']]:
        if file_hash(root/name) != prior['frozen_files'][name]:
            raise ValueError('Earlier frozen feature source changed')
    protocol = {'study': STUDY, 'registered_at': datetime.now(timezone.utc).isoformat(),
        'sources': {p: file_hash(root/p) for p in SOURCES.values()},
        'hypothesis': 'Smooth nonlinear interactions among prior performance, matchup context and '
                      'single-book probabilities may add information beyond linear or shallow-tree models.',
        'candidates': CANDIDATES, 'random_features': 128, 'random_seed': 20260919,
        'kernel': 'Gaussian random Fourier features, gamma=1/input_dimension; fixed random projection. '
                  'Concatenate intercept, standardised linear features/sqrt(d), and sqrt(2/128)*cosine '
                  'features. No subsequent output-feature standardisation.',
        'scaling': 'Mean/std learned from training only, including mirrored binary rows; std<1e-8 becomes1. '
                   'Standardised coordinates clipped to[-5,5]. No full-data vocabulary or scaler.',
        'objective': 'Recency-weighted multinomial log loss with log market offset and L2 on all '
                     'coefficients including intercept; last class reference. Analytic gradient L-BFGS-B '
                     'maxiter500 ftol1e-10 gtol1e-6; failure aborts.',
        'sports': {'football': {'tuning': [2021, 2022], 'evaluation': [2023, 2024, 2025], 'min_train': 5000},
                   'ufc': {'tuning': [2019, 2020, 2021], 'evaluation': [2022, 2023, 2024], 'min_train': 1000},
                   'wta': {'tuning': [2021, 2022], 'evaluation': [2023, 2024, 2025], 'min_train': 3000}},
        'fit': 'Annual, six-year rolling training, completed results only, strictly before Jan1 minus7days.',
        'lookback_years': 6, 'embargo_days': 7, 'half_life_years': 3.,
        'features': {'football': 'Previously frozen enriched prior-day design plus total rest, minimum '
                                 'team experience and division rank. No current scores or closing prices.',
                     'ufc': 'Previously frozen enriched design plus minimum experience, mean age and '
                            'maximum layoff. No new physical metadata, postfight stats or 2025+ records.',
                     'wta': 'Previous single-book serve_surface design, surface one-hot, log1p minimum '
                            'and mean historical match counts. Same28day delay, five prior matches and '
                            'past-only identity registry as registered service study. No Pinnacle dependence.'},
        'binary_symmetry': 'Signed differences negate and shared context remains invariant under player '
                           'swap. Train both orientations with equal weight; symmetrise predicted probabilities.',
        'selection': {'minimum_ev_after_haircut': .03, 'profit_haircut': .02,
                      'odds_min': 1.3, 'odds_max': 5., 'flat_fraction': .0025,
                      'maximum_daily_exposure': .02, 'sensitivity_haircuts': [0., .02, .05]},
        'staking': 'Separate1000 per sport/model/period. .25% of day-start bankroll, floor cents, '
                   '2% daily cap, no intraday reinvestment; deterministic date/competition/participant order.',
        'settlement': 'Football completed1X2 and UFC labelled bouts; WTA noncompleted assumed void '
                      'and consume quota. Historical retirement rules and accepted wagers unverified.',
        'choice': 'Per sport, highest tuning95% lower among >=50settled; candidate order breaks ties. '
                  'Admit only positive ROI and lower. Save ALL sport locks before ANY evaluation returns.',
        'uncertainty': {'samples': 100000, 'seed': 20260919, 'family_size': 6,
                        'lower_quantile_per_candidate': .05/6,
                        'scope': 'Circular three-month blocks. Six new configurations only; does not '
                                 'correct the many prior adaptive experiments.'},
        'nominal_gate': 'Chosen/admitted at tuning; >=200settled evaluation; ROI2/5%>0; familylower>0; '
                        '>=2positiveyears; ROIwithoutbestyear>0. Not validation of real execution.',
        'evidence': 'EXPLORATORY_REUSED_HISTORY_NOT_INDEPENDENT_VALIDATION',
        'ufc_2025_onward': 'REMAINS_CLOSED', 'no_2026_fitting_or_evaluation': True,
        'real_money_authorised': False, 'app_modified': False,
        'limitations': ['Repeatedly explored histories are not pristine holdouts.',
                       'UFC prices legacy_unverified: unknown quote timing and execution provenance.',
                       'Football/WTA Bet365 historical prices do not establish French execution.',
                       'No new point-in-time data; cached feature revisions and missingness remain limitations.',
                       'Finite random features and frozen bandwidth are modelling assumptions.',
                       'No post-result threshold, seed, feature, league or model search.',
                       'No actual wagers, deployment, application change or automatic schedule.']}
    deps = ['src/backtesting/kernel_matchups.py', 'scripts/run_kernel_matchups.py',
            'src/backtesting/football_cross_market.py', 'src/backtesting/three_sport_residual.py',
            'src/backtesting/wta_serve_return.py', 'src/backtesting/wta_price_residual.py',
            'src/backtesting/wta_single_book.py', 'src/features/football_features.py',
            'models/wta_serve_return_2026_09_18/protocol.json',
            'models/wta_serve_return_2026_09_18/registration.json']
    folder = root/'models'/STUDY; folder.mkdir(exist_ok=False)
    write_json(folder/'protocol.json', protocol)
    write_json(folder/'registration.json', {'protocol_sha256': file_hash(folder/'protocol.json'),
        'implementations': {p: file_hash(root/p) for p in deps}, 'before_new_returns': True})
    return folder


class KernelResidual:
    def __init__(self, penalty, components=128, seed=20260919, signs=None):
        self.penalty, self.components, self.seed = penalty, components, seed
        self.signs = None if signs is None else np.asarray(signs, dtype=float)

    def transform(self, x):
        z = np.clip((x-self.mean)/self.scale, -5, 5)
        radial = np.sqrt(2/self.components)*np.cos(z@self.projection+self.phase)
        return np.column_stack([np.ones(len(x)), z/np.sqrt(x.shape[1]), radial])

    @staticmethod
    def objective(flat, features, q, labels, weights, penalty):
        coefficients = flat.reshape(features.shape[1], q.shape[1]-1)
        logits = np.log(q).copy(); logits[:, :-1] += features@coefficients
        logp = logits-logsumexp(logits, axis=1, keepdims=True)
        normal = weights.sum()
        loss = -np.dot(weights, logp[np.arange(len(labels)), labels])/normal
        residual = np.exp(logp); residual[np.arange(len(labels)), labels] -= 1
        gradient = features.T@(weights[:, None]*residual[:, :-1])/normal+penalty*coefficients
        return float(loss+.5*penalty*np.sum(coefficients**2)), gradient.ravel()

    def fit(self, x, q, labels, weights):
        x, q = np.asarray(x, float), np.asarray(q, float)
        labels, weights = np.asarray(labels, int), np.asarray(weights, float)
        if (not np.isfinite(x).all() or not np.isfinite(q).all() or (q <= 0).any()
                or not np.allclose(q.sum(axis=1), 1) or not np.isfinite(weights).all()
                or (weights <= 0).any() or not np.isin(labels, np.arange(q.shape[1])).all()):
            raise ValueError('Invalid training arrays')
        if self.signs is not None:
            if q.shape[1] != 2 or len(self.signs) != x.shape[1]: raise ValueError('Invalid symmetry')
            x = np.vstack([x, x*self.signs]); q = np.vstack([q, q[:, ::-1]])
            labels = np.r_[labels, 1-labels]; weights = np.r_[weights, weights]/2
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(0, np.sqrt(2/x.shape[1]), (x.shape[1], self.components))
        self.phase = rng.uniform(0, 2*np.pi, self.components)
        features = self.transform(x)
        initial = np.zeros(features.shape[1]*(q.shape[1]-1))
        result = minimize(self.objective, initial, args=(features, q, labels, weights, self.penalty),
            jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f'Kernel optimisation failed: {result.message}')
        self.coefficients = result.x.reshape(features.shape[1], q.shape[1]-1)
        self.iterations = int(result.nit)
        return self

    def raw_predict(self, x, q):
        logits = np.log(q).copy(); logits[:, :-1] += self.transform(x)@self.coefficients
        return softmax(logits, axis=1)

    def predict(self, x, q):
        p = self.raw_predict(x, q)
        if self.signs is not None:
            other = self.raw_predict(x*self.signs, q[:, ::-1])
            left = (p[:, 0]+other[:, 1])/2
            p = np.column_stack([left, 1-left])
        p = np.clip(p, 1e-12, 1-1e-12)
        return p/p.sum(axis=1, keepdims=True)


def load_frame(root, sport, protocol):
    frame = pd.read_parquet(Path(root)/SOURCES[sport])
    frame['_date'] = pd.to_datetime(frame['_date'])
    if frame['_date'].isna().any() or frame._source_row_id.duplicated().any():
        raise ValueError('Missing dates or duplicate IDs')
    if frame._date.dt.year.max() > max(protocol['sports'][sport]['evaluation']):
        raise ValueError('Reserved future data in cache')
    if sport == 'football':
        frame['_p1'], frame['_p2'], frame['_tournament'] = frame.home_team, frame.away_team, frame.league
    elif sport == 'ufc':
        frame['_p1'], frame['_p2'], frame['_tournament'] = frame.fighter_1, frame.fighter_2, frame.event_name
    prices = frame[[f'price_{j}' for j in range(3 if sport == 'football' else 2)]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 1).any(): raise ValueError('Invalid historical odds')
    margin = (1/prices).sum(axis=1)
    if not ((margin >= 1) & (margin <= 1.2)).all(): raise ValueError('Invalid overround')
    if not frame._label.isin(range(prices.shape[1])).all(): raise ValueError('Invalid outcome')
    if sport == 'wta':
        if not (frame.stats_max_start+pd.Timedelta(days=28) < frame._date).all():
            raise ValueError('WTA statistics timing mismatch')
    return frame.sort_values(['_date', '_source_row_id']).reset_index(drop=True)


def arrays(frame, sport):
    if sport == 'wta':
        x, q = serve_arrays(frame, 'serve_surface')
        names = ['market_logit', 'market_curve', 'market_margin']+[
            prefix+'_'+stat for prefix in ['global', 'surface'] for stat in ['serve', 'return', 'volume']]
        context = {f'is_{s.lower()}': frame.Surface.eq(s).to_numpy(float) for s in ['Hard', 'Clay', 'Grass']}
        context['log_min_history'] = np.log1p(np.minimum(frame.history_count_1, frame.history_count_2))
        context['log_mean_history'] = np.log1p((frame.history_count_1+frame.history_count_2)/2)
        signs = -np.ones(x.shape[1])
    else:
        x, q, _, names = design(frame, sport, 'enriched')
        if sport == 'ufc':
            context = {'minimum_experience': np.minimum(frame.experience_1, frame.experience_2),
                       'mean_age': (frame.age_1+frame.age_2)/2,
                       'maximum_layoff': np.maximum(frame.layoff_days_1, frame.layoff_days_2)}
            signs = -np.ones(x.shape[1])
        else:
            context = {'total_rest': frame.home_rest_days+frame.away_rest_days,
                       'minimum_experience': np.minimum(frame.home_matches_played, frame.away_matches_played),
                       'division_rank': frame.division_rank}
            signs = None
    extra = np.nan_to_num(np.column_stack(list(context.values())).astype(float), nan=0, posinf=0, neginf=0)
    if signs is not None: signs = np.r_[signs, np.ones(extra.shape[1])]
    return np.column_stack([x, extra]), q, names+list(context), signs


def predict_year(frame, matrices, sport, candidate, year, protocol):
    x, q, names, signs = matrices
    cutoff = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=protocol['embargo_days'])
    train = frame._date.lt(cutoff) & frame._date.ge(cutoff-pd.DateOffset(years=protocol['lookback_years'])) & frame._status.eq('completed')
    test = frame._date.dt.year.eq(year)
    if train.sum() < protocol['sports'][sport]['min_train'] or not test.any(): raise ValueError('Insufficient fold')
    age = (cutoff-frame.loc[train, '_date']).dt.total_seconds().to_numpy()/86400
    weights = np.exp2(-age/(365.25*protocol['half_life_years']))
    model = KernelResidual(CANDIDATES[candidate], protocol['random_features'], protocol['random_seed'], signs)
    model.fit(x[train], q[train], frame.loc[train, '_label'].to_numpy(int), weights)
    p = model.predict(x[test], q[test])
    columns = ['_source_row_id', '_date', '_p1', '_p2', '_tournament', '_status', '_label']+[
        f'price_{j}' for j in range(q.shape[1])]
    rows = frame.loc[test, columns].copy(); rows['year'] = year
    for j in range(q.shape[1]): rows[f'p{j}'], rows[f'q{j}'] = p[:, j], q[test, j]
    ix, labels = np.arange(len(rows)), rows._label.to_numpy(int)
    rows['model_loss'], rows['market_loss'] = -np.log(p[ix, labels]), -np.log(q[test][ix, labels])
    rows.loc[rows._status.ne('completed'), ['model_loss', 'market_loss']] = np.nan
    fold = {'sport': sport, 'candidate': candidate, 'year': year, 'train_rows': int(train.sum()),
            'test_rows': int(test.sum()), 'train_max': str(frame.loc[train, '_date'].max().date()),
            'train_min': str(frame.loc[train, '_date'].min().date()), 'cutoff_exclusive': str(cutoff.date()),
            'features': names, 'iterations': model.iterations}
    return rows.reset_index(drop=True), fold


def evaluate(predictions, protocol):
    classes = sum(c.startswith('price_') for c in predictions.columns)
    p = predictions[[f'p{j}' for j in range(classes)]].to_numpy(float)
    prices = predictions[[f'price_{j}' for j in range(classes)]].to_numpy(float)
    choices = select(p, prices, protocol['selection'])
    # Adapt the chosen outcome to the tested sequential binary ledger. Preserve
    # all original probabilities/outcomes in separate immutable predictions.
    rows = predictions.copy(); rows['original_label'] = rows._label
    rows['selected_original_outcome'] = choices
    index, safe = np.arange(len(rows)), np.maximum(choices, 0)
    rows['_label'] = np.where(choices == rows.original_label, 0, 1)
    rows['p0'], rows['p1'] = np.where(choices >= 0, p[index, safe], 0), 0.
    rows['price_0'], rows['price_1'] = prices[index, safe], 1.01  # internal nonselectable dummy
    summary, bets = summarise(rows, protocol)
    summary['cash_roi_2pct'] = float(bets.profit_cash.sum()/bets.stake_cash.sum()) if len(bets) else None
    return summary, bets


def run(root, progress=print):
    root = Path(root); folder = root/'models'/STUDY
    protocol = json.loads((folder/'protocol.json').read_text())
    registration = json.loads((folder/'registration.json').read_text())
    if (folder/'run_started.json').exists(): raise FileExistsError('Existing experiment preserved; no rerun')
    for path, digest in {**protocol['sources'], **registration['implementations'],
            str((folder/'protocol.json').relative_to(root)): registration['protocol_sha256']}.items():
        if file_hash(root/path) != digest: raise ValueError(f'Frozen input changed: {path}')
    write_json(folder/'run_started.json', {'at': datetime.now(timezone.utc).isoformat()})
    locks, frames, matrices, folds = {}, {}, {}, []
    def period(sport, candidate, years, name):
        batches = []
        for year in years:
            progress(f'{sport}/{candidate} : ajustement chronologique {year}…')
            rows, fold = predict_year(frames[sport], matrices[sport], sport, candidate, year, protocol)
            batches.append(rows); folds.append(fold)
        predictions = pd.concat(batches, ignore_index=True)
        summary, bets = evaluate(predictions, protocol)
        predictions.to_parquet(folder/f'{sport}_{candidate}_{name}_predictions.parquet', index=False)
        bets.attrs = {}; bets.to_parquet(folder/f'{sport}_{candidate}_{name}_bets.parquet', index=False)
        return summary
    for sport, spec in protocol['sports'].items():
        frames[sport] = load_frame(root, sport, protocol); matrices[sport] = arrays(frames[sport], sport)
        tuning = {c: period(sport, c, spec['tuning'], 'tuning') for c in CANDIDATES}
        eligible = [c for c in CANDIDATES if tuning[c]['settled'] >= 50 and tuning[c]['uncertainty']['ci95'][0] is not None]
        chosen = max(eligible, key=lambda c: tuning[c]['uncertainty']['ci95'][0]) if eligible else None
        admitted = bool(chosen and (tuning[chosen]['roi']['0.02'] or 0) > 0 and tuning[chosen]['uncertainty']['ci95'][0] > 0)
        locks[sport] = {'chosen': chosen, 'admitted': admitted, 'tuning': tuning,
                        'at': datetime.now(timezone.utc).isoformat()}
        write_json(folder/f'{sport}_selection_lock.json', locks[sport])
    write_json(folder/'all_selection_locks.json', locks)
    evaluations = {}
    for sport, spec in protocol['sports'].items():
        evaluations[sport] = {}
        for candidate in CANDIDATES:
            result = period(sport, candidate, spec['evaluation'], 'evaluation')
            result['checks'] = gate(result, locks[sport]['admitted'] and locks[sport]['chosen'] == candidate)
            result['nominal_historical_gate_passed'] = all(result['checks'].values())
            result['real_money_authorised'] = False
            evaluations[sport][candidate] = result
            progress(f"{sport}/{candidate}: {result['settled']} réglés; ROI={result['roi']['0.02']}; filtre={result['nominal_historical_gate_passed']}")
    report = {'study': STUDY, 'selection': locks, 'evaluation': evaluations, 'folds': folds,
              'ufc_quote_quality': frames['ufc'].temporal_quality.fillna('missing').value_counts().to_dict(),
              'evidence': protocol['evidence'], 'limitations': protocol['limitations'],
              'app_modified': False, 'real_money_authorised': False}
    write_json(folder/'report.json', report)
    return report
