"""Registered WTA ablation: one actual price pair, never Bet365 or a consensus.

Historical Pinnacle prices are an execution proxy, NOT French-book evidence.
Reused historical years are exploratory even with strictly chronological fits.
No live application artifact is changed by this experiment.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtesting.football_cross_market import file_hash
from src.backtesting.nonlinear_market import OffsetTrees, recency_weights
from src.backtesting.wta_price_residual import rebuild_daily_features, select, uncertainty, roi

STUDY = 'wta_single_book_2026_09_17'
CANDIDATES = ['price_only', 'price_and_history']
LAGGED = ['elo_diff', 'surface_elo_diff', 'form_10_diff', 'rest_diff', 'fatigue_diff']
SOURCE = 'models/wta_strategy/pre_match_features.csv.gz'
SOURCE_COLUMNS = ['_source_row_id', '_date', '_p1', '_p2', '_label', '_status',
                  '_surface', '_series', '_round', '_tournament', 'Pinnacle_1', 'Pinnacle_2']
SERIES = {'International', 'Premier', 'Grand Slam', 'WTA250', 'WTA500', 'WTA1000',
          'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tour Championships'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def register(root):
    root = Path(root)
    folder = root / 'models' / STUDY
    tree = json.loads((root / 'models/nonlinear_market/protocol.json').read_text())
    protocol = {
        'study': STUDY, 'registered_at': datetime.now(timezone.utc).isoformat(),
        'evidence': 'EXPLORATORY_REUSED_HISTORY_NOT_INDEPENDENT_VALIDATION',
        'source': SOURCE, 'source_sha256': file_hash(root / SOURCE),
        'source_columns': SOURCE_COLUMNS, 'candidates': CANDIDATES,
        'hypothesis': 'Remove all Bet365 and cross-book dependence. Compare price-only shallow trees '
                      'with the same price plus strictly prior-day form and surface context. No current ranks required.',
        'execution': 'Pinnacle historical single-book pair only; no historical French quotes available. '
                     'French price substitution is a separate unvalidated prospective experiment.',
        'training_year_start': 2007, 'tuning_years': [2021, 2022], 'evaluation_years': [2023, 2024, 2025],
        'reserved_2026': 'NOT USED IN FIT, FEATURES, TUNING OR EVALUATION',
        'min_train': 5000, 'embargo_days': 7, 'half_life_years': 3.,
        'booster': tree['booster'], 'boost_rounds': tree['boost_rounds'],
        'features': {'price_only': 'single-book logit, signed quadratic logit, logit times overround',
                     'price_and_history': LAGGED + ['is_hard', 'is_clay', 'is_grass'],
                     'imputation': 'Fixed zero for nonfinite signed lagged differences; no learned full-sample transformation'},
        'fit': 'Annual model fitted strictly before January 1 minus seven days; completed training only. '
               'Symmetric orientations, fixed tree parameters. Prior-day state updated as results become historical. '
               'No random split, early stopping, hyperparameter/threshold/surface search.',
        'selection': tree['selection'],
        'staking': 'Flat 0.25% day-start bankroll; 2% day cap; floor cents; no same-day reinvestment '
                   'or reuse of void stakes. Process date then tournament and sorted player names; '
                   'do not redistribute based on future number of daily selections. This deterministic '
                   'date-only ordering is NOT a verified live quote/settlement sequence.',
        'choice': 'Highest tuning 95% lower ROI among candidates with >=50 settled. '
                  'Admit only if tuning ROI and lower bound positive; ties follow candidate declaration order. '
                  'Persist choice before diagnostic evaluation of both variants. No selection using evaluation.',
        'gate': 'Chosen candidate admitted at tuning; >=200 evaluation settlements; positive ROI at '
                '2% and 5% profit haircut; family-adjusted lower>0; >=2 positive years; ROI without best year>0.',
        'uncertainty': {'samples': 20000, 'evaluation_samples': 100000, 'seed': 20260917,
                        'family_size': 2, 'lower_quantile_per_candidate': .05 / 2,
                        'scope': 'Only these two new variants, NOT repeated project searches. '
                                 'Circular three-month blocks including zero-bet months; fraction-weighted ROI.'},
        'limitations': ['Previously examined years are not a pristine holdout.',
                       'No timestamped historical odds or accepted stakes; execution cannot be reproduced exactly.',
                       'Pinnacle prices are not assumed available with French operators.',
                       'Noncompleted matches assumed void; actual bookmaker rules may differ.',
                       'Surface and identity data still required for the history variant in live use.',
                       'Historical revisions and missing prices may bias the available sample.'],
        'real_money_authorised': False,
    }
    dependencies = ['src/backtesting/wta_single_book.py', 'scripts/run_wta_single_book.py',
                    'src/backtesting/nonlinear_market.py', 'src/backtesting/wta_price_residual.py',
                    'src/backtesting/football_cross_market.py', 'src/features/elo_system.py',
                    'src/features/feature_builder.py', 'models/nonlinear_market/protocol.json']
    hashes = {p: file_hash(root / p) for p in dependencies}
    folder.mkdir(exist_ok=False)
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'registration.json', {
        'protocol_sha256': file_hash(folder / 'protocol.json'), 'implementations': hashes,
        'registered_before_new_returns': True, 'numpy': np.__version__,
        'pandas': pd.__version__, 'xgboost': xgb.__version__})
    return folder


def prepare(raw, protocol):
    # Discard inherited predictors and every other bookmaker before rebuilding.
    frame = raw[SOURCE_COLUMNS].copy()
    frame['_date'] = pd.to_datetime(frame['_date'], errors='raise')
    frame = frame[frame['_date'].dt.year.between(protocol['training_year_start'],
                                                max(protocol['evaluation_years']))].copy()
    if frame.empty or frame['_date'].isna().any() or frame['_source_row_id'].duplicated().any():
        raise ValueError('Missing dates, empty history or duplicate source IDs')
    if not frame['_label'].isin([0, 1]).all():
        raise ValueError('Invalid outcome labels')
    if not frame['_status'].isin(['completed', 'retired', 'walkover', 'cancelled', 'defaulted', 'unfinished', 'unknown']).all():
        raise ValueError('Unknown match settlement status')
    valid_context = frame['_surface'].isin(['Hard', 'Clay', 'Grass']) & frame['_series'].isin(SERIES)
    quality = {'rows_before_context_filter': len(frame), 'excluded_context_rows': int((~valid_context).sum())}
    frame = frame[valid_context].copy()
    pair = pd.Series(['|'.join(sorted([a, b])) for a, b in zip(frame['_p1'], frame['_p2'])], index=frame.index)
    if frame.assign(_pair=pair).duplicated(['_date', '_pair']).any() or frame['_p1'].eq(frame['_p2']).any():
        raise ValueError('Duplicate/ambiguous match identity')
    # ALL valid sporting results update state, even those without Pinnacle prices.
    frame = rebuild_daily_features(frame)
    odds = frame[['Pinnacle_1', 'Pinnacle_2']].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        overround = (1 / odds).sum(axis=1)
    valid = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1) & (overround >= 1) & (overround <= 1.2)
    quality.update(rebuilt_rows=len(frame), excluded_price_rows=int((~valid).sum()))
    frame = frame.loc[valid].copy()
    frame['_label'] = 1 - frame['_label'].astype(int)  # class zero = player 1
    frame['price_0'], frame['price_1'] = frame['Pinnacle_1'], frame['Pinnacle_2']
    frame = frame.drop(columns=['Pinnacle_1', 'Pinnacle_2']).sort_values(['_date', '_source_row_id']).reset_index(drop=True)
    quality.update(usable_rows=len(frame), last_date=str(frame['_date'].max().date()),
                   rows_by_year={str(y): int(n) for y, n in frame.groupby(frame['_date'].dt.year).size().items()},
                   no_bet365_columns=True, no_current_rank_input=True, strict_prior_day_rebuild=True)
    return frame, quality


def arrays(frame, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Unregistered model')
    odds = frame[['price_0', 'price_1']].to_numpy(float)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Invalid single-book price pair')
    inv = 1 / odds
    overround = inv.sum(axis=1)
    q = inv / overround[:, None]
    logit = np.log(q[:, 0] / q[:, 1])
    columns = {'book_logit': logit, 'book_signed_quadratic': logit * np.abs(logit),
               'book_logit_overround': logit * overround}
    if candidate == 'price_and_history':
        columns.update({c: np.nan_to_num(frame[c].to_numpy(float), nan=0, posinf=0, neginf=0) for c in LAGGED})
    signs = [-1.] * len(columns)
    if candidate == 'price_and_history':
        columns.update({f'is_{s.lower()}': frame['_surface'].eq(s).to_numpy(float) for s in ['Hard', 'Clay', 'Grass']})
        signs += [1.] * 3
    return np.column_stack(list(columns.values())), q, odds, list(columns), np.asarray(signs)


def predict_fold(frame, candidate, year, protocol):
    cutoff = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=protocol['embargo_days'])
    train = (frame['_date'].lt(cutoff) & frame['_status'].eq('completed')).to_numpy()
    test = frame['_date'].dt.year.eq(year).to_numpy()
    if train.sum() < protocol['min_train'] or not test.any():
        raise ValueError(f'Insufficient chronological data: {candidate}/{year}')
    x, q, odds, names, signs = arrays(frame, candidate)
    y = frame['_label'].to_numpy(int)
    weights = recency_weights(frame.loc[train, '_date'], cutoff, protocol['half_life_years'])
    model = OffsetTrees(protocol['booster'], protocol['boost_rounds'], signs)
    model.fit(x[train], q[train], y[train], weights)
    p = model.predict(x[test], q[test])
    rows = frame.loc[test, ['_source_row_id', '_date', '_label', '_status', '_p1', '_p2', '_tournament']].reset_index(drop=True)
    rows['year'], rows['candidate'] = year, candidate
    for j in [0, 1]:
        rows[f'p{j}'], rows[f'q{j}'], rows[f'price_{j}'] = p[:, j], q[test, j], odds[test, j]
    completed = rows['_status'].eq('completed')
    rows['model_loss'] = np.where(completed, -np.log(p[np.arange(len(p)), y[test]]), np.nan)
    rows['market_loss'] = np.where(completed, -np.log(q[test][np.arange(len(p)), y[test]]), np.nan)
    return rows, {'candidate': candidate, 'year': year, 'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
                  'train_max_date': str(frame.loc[train, '_date'].max().date()), 'cutoff_exclusive': str(cutoff.date()),
                  'features': names, 'swap_signs': signs.tolist()}


def ledger(predictions, rule):
    rows = predictions.copy()
    odds = rows[['price_0', 'price_1']].to_numpy(float)
    p = rows[['p0', 'p1']].to_numpy(float)
    chosen = select(p, odds, rule)
    safe = np.maximum(chosen, 0)
    rows['selected'] = chosen
    rows['odds'] = odds[np.arange(len(rows)), safe]
    rows['probability'] = p[np.arange(len(rows)), safe]
    rows['won'] = (chosen == rows['_label']) & rows['_status'].eq('completed')
    rows['_order'] = ['|'.join(sorted([a, b])) for a, b in zip(rows['_p1'], rows['_p2'])]
    rows = rows[rows['selected'] >= 0].sort_values(['_date', '_tournament', '_order', '_source_row_id'])
    balance, peak, drawdown = 100000, 100000, 0.
    records, skipped = [], 0
    for _, day in rows.groupby('_date', sort=True):
        base = balance
        budget = math.floor(base * rule['maximum_daily_exposure'])
        fixed = math.floor(base * rule['flat_fraction'])
        daily_profit = 0
        for row in day.to_dict('records'):
            stake = min(fixed, budget, base)
            if stake <= 0:
                skipped += 1
                continue
            budget -= stake
            settled = row['_status'] == 'completed'
            returns = {f'return_{h}': (0 if not settled else (row['odds']-1)*(1-h) if row['won'] else -1)
                       for h in rule['sensitivity_haircuts']}
            profit = round(stake * returns[f"return_{rule['profit_haircut']}"])
            records.append({**row, **returns, 'settled': settled, 'fraction': stake / base,
                            'stake_cash': stake / 100, 'profit_cash': profit / 100, 'bankroll_before': base / 100})
            daily_profit += profit
        balance += daily_profit
        peak = max(peak, balance)
        drawdown = max(drawdown, 1 - balance / peak)
    columns = [*rows.columns, *[f'return_{h}' for h in rule['sensitivity_haircuts']],
               'settled', 'fraction', 'stake_cash', 'profit_cash', 'bankroll_before']
    bets = pd.DataFrame(records, columns=columns)
    bets['settled'] = bets['settled'].astype(bool)
    bets.attrs.update(final_bankroll=balance / 100, maximum_drawdown=drawdown,
                      skipped_daily_budget=skipped, theoretical_signals=len(rows))
    return bets


def summarise(predictions, protocol):
    bets = ledger(predictions, protocol['selection'])
    yearly = {str(year): {'settled': int(group['settled'].sum()), 'roi': roi(group, .02),
                           'profit_fraction': float((group['fraction'] * group['return_0.02']).sum())}
              for year, group in bets.groupby('year')}
    best = max(yearly, key=lambda year: yearly[year]['profit_fraction']) if yearly else None
    summary = {'matches': len(predictions), 'selections': len(bets), 'settled': int(bets['settled'].sum()),
               'void': int((~bets['settled']).sum()), 'roi': {str(h): roi(bets, h) for h in [0., .02, .05]},
               'uncertainty': uncertainty(bets, predictions['_date'], protocol['uncertainty']),
               'yearly': yearly, 'bankroll': dict(bets.attrs), 'best_year': best,
               'roi_without_best_year': roi(bets[bets['year'] != int(best)], .02) if best else None,
               'model_log_loss': float(predictions['model_loss'].mean()),
               'market_log_loss': float(predictions['market_loss'].mean())}
    return summary, bets


def gate(summary, tuning_admitted):
    return {'tuning_admitted': bool(tuning_admitted), '200_settled': summary['settled'] >= 200,
            'positive_roi_2pct': (summary['roi']['0.02'] or 0) > 0,
            'positive_roi_5pct': (summary['roi']['0.05'] or 0) > 0,
            'positive_family_lower': (summary['uncertainty']['family_lower'] or 0) > 0,
            'two_positive_years': sum((y['roi'] or 0) > 0 for y in summary['yearly'].values()) >= 2,
            'positive_without_best_year': (summary['roi_without_best_year'] or 0) > 0}


def run(root, progress=print):
    root = Path(root)
    folder = root / 'models' / STUDY
    protocol = json.loads((folder / 'protocol.json').read_text())
    registration = json.loads((folder / 'registration.json').read_text())
    if (folder / 'run_started.json').exists():
        raise FileExistsError('Existing run preserved. Any repair requires an explicit new study/amendment.')
    checks = {**registration['implementations'], protocol['source']: protocol['source_sha256'],
              str((folder / 'protocol.json').relative_to(root)): registration['protocol_sha256']}
    for name, expected in checks.items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Registered input changed: {name}')
    write_json(folder / 'run_started.json', {'started_at': datetime.now(timezone.utc).isoformat(),
                                           'registration_sha256': file_hash(folder / 'registration.json')})
    progress('WTA : reconstruction des descripteurs strictement antérieurs au jour du match…')
    raw = pd.read_csv(root / protocol['source'], usecols=SOURCE_COLUMNS, low_memory=False)
    frame, quality = prepare(raw, protocol)
    frame.to_parquet(folder / 'features.parquet', index=False)
    write_json(folder / 'quality.json', quality)
    folds, tuning = [], {}
    def predictions(candidate, years):
        batches = []
        for year in years:
            progress(f'{candidate} : modèle annuel {year}')
            rows, fold = predict_fold(frame, candidate, year, protocol)
            batches.append(rows)
            folds.append(fold)
        return pd.concat(batches, ignore_index=True)
    for candidate in CANDIDATES:
        rows = predictions(candidate, protocol['tuning_years'])
        rows.to_parquet(folder / f'{candidate}_tuning_predictions.parquet', index=False)
        tuning[candidate], _ = summarise(rows, protocol)
    eligible = [c for c in CANDIDATES if tuning[c]['settled'] >= 50 and tuning[c]['uncertainty']['ci95'][0] is not None]
    chosen = max(eligible, key=lambda c: tuning[c]['uncertainty']['ci95'][0]) if eligible else None
    admitted = bool(chosen and (tuning[chosen]['roi']['0.02'] or 0) > 0 and tuning[chosen]['uncertainty']['ci95'][0] > 0)
    lock = {'chosen': chosen, 'tuning_admitted': admitted, 'scores': tuning,
            'written_at': datetime.now(timezone.utc).isoformat(), 'threshold': .02,
            'no_evaluation_selection': True, 'real_money_authorised': False}
    write_json(folder / 'selection_lock.json', lock)
    evaluations = {}
    settings = deepcopy(protocol)
    settings['uncertainty']['samples'] = protocol['uncertainty']['evaluation_samples']
    for candidate in CANDIDATES:
        rows = predictions(candidate, protocol['evaluation_years'])
        rows.to_parquet(folder / f'{candidate}_evaluation_predictions.parquet', index=False)
        summary, bets = summarise(rows, settings)
        summary['checks'] = gate(summary, admitted and candidate == chosen)
        summary['research_gate_passed'] = all(summary['checks'].values())
        summary['diagnostic_only'] = candidate != chosen or not admitted
        evaluations[candidate] = summary
        bets.attrs = {}
        bets.to_parquet(folder / f'{candidate}_bets.parquet', index=False)
        progress(f"{candidate} : {summary['settled']} paris réglés ; ROI net {summary['roi']['0.02']}")
    report = {'study': STUDY, 'quality': quality, 'selection': lock, 'evaluation': evaluations, 'folds': folds,
              'evidence': protocol['evidence'], 'real_money_authorised': False,
              'status': 'PAPER_REPLICATION_ONLY' if any(s['research_gate_passed'] for s in evaluations.values()) else 'NO_ROBUST_CANDIDATE',
              'app_modified': False, 'limitations': protocol['limitations']}
    write_json(folder / 'report.json', report)
    def percent(value):
        return 'non calculable' if value is None else f'{value:+.2%}'
    lines = ['# WTA sans Bet365 — essai exploratoire', '',
             'Cotes historiques Pinnacle uniquement : ce résultat ne prouve rien sur les cotes françaises actuelles.',
             'Années déjà explorées ; aucun nouveau holdout indépendant. Aucune modification du modèle de l’application.', '',
             '| Variante | Réglage 2021–2022 | Réglés 2023–2025 | ROI net 2 % | IC 95 % | ROI net 5 % | Validation |',
             '|---|---:|---:|---:|---|---:|---|']
    for candidate, s in evaluations.items():
        lo, hi = s['uncertainty']['ci95']
        lines.append(f"| {candidate} | {percent(tuning[candidate]['roi']['0.02'])} | {s['settled']} | "
                     f"{percent(s['roi']['0.02'])} | [{percent(lo)} ; {percent(hi)}] | {percent(s['roi']['0.05'])} | "
                     f"{'suivi papier seulement' if s['research_gate_passed'] else 'rejetée'} |")
    lines += ['', f'Choix figé sur 2021–2022 : {chosen}. Admission : {admitted}.', '',
              'Seuil EV net 2 %, cotes 1,30–5,00 ; mise fixe 0,25 %, plafond quotidien 2 %, sans réinvestissement intrajournalier.',
              'Classements actuels non nécessaires. La variante price_only ne requiert que la paire de cotes ; '
              'price_and_history exige aussi des identités et une surface vérifiées.',
              'Le ROI est pondéré par la fraction de bankroll risquée ; le carnet séparé décrit la trajectoire en euros.', '',
              '## Limites', '', *['- ' + limitation for limitation in protocol['limitations']], '',
              'Aucun déploiement automatique et aucune autorisation de mise réelle sur ce seul essai.']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report
