"""Frozen three-sport search. Historical diagnostics never authorise real bets."""
from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.football_cross_market import MarketResidual, file_hash
from src.backtesting.tennis_phase4 import (
    STRUCTURAL_EXTRA, SURFACE_FEATURES, build_phase4_features,
)
from src.backtesting.wta_price_residual import (
    fit_symmetric, predict_symmetric, select, ledger, uncertainty, roi,
)
from src.data.tennis_pipeline import _full_name_key, _pair_key, _tokens
from src.features.football_features import FEATURE_COLUMNS, build_football_features


def football_extra(frame):
    """Read an entire calendar day's histories before any updates."""
    states = defaultdict(lambda: deque(maxlen=30))
    rows = []
    for date, day in frame.sort_values(['match_date', 'match_id']).groupby('match_date', sort=True):
        pending = []
        for row in day.to_dict('records'):
            sides = []
            for side, other in [('home', 'away'), ('away', 'home')]:
                history = list(states[(row['country'], row[side+'_team'])])
                def mean(key, n=10):
                    values = [r[key] for r in history[-n:] if np.isfinite(r[key])]
                    return float(np.mean(values)) if values else np.nan
                sides.append({
                    'form_trend': mean('points', 3)-mean('points', 10),
                    'goal_balance': mean('gf')-mean('ga'),
                    'target_balance': mean('tf')-mean('ta'),
                    'finishing_proxy': mean('finishing'),
                    'defence_proxy': mean('defence'),
                    'market_surprise': mean('surprise'),
                    'games14': sum(pd.Timedelta(0) < date-r['date'] <= pd.Timedelta(days=14) for r in history),
                })
                prices = np.array([row['B365H'], row['B365D'], row['B365A']], float)
                good = np.isfinite(prices).all() and (prices > 1).all()
                q = (1/prices)/(1/prices).sum() if good else np.full(3, np.nan)
                points = 1 if row['result']=='D' else 3 if row['result']==('H' if side=='home' else 'A') else 0
                gf, ga = row[side+'_goals'], row[other+'_goals']
                tf, ta = row['postmatch_'+side+'_shots_on_target'], row['postmatch_'+other+'_shots_on_target']
                pending.append(((row['country'], row[side+'_team']), {
                    'date': date, 'points': points, 'gf': gf, 'ga': ga, 'tf': tf, 'ta': ta,
                    'finishing': gf-.3*tf, 'defence': ga-.3*ta,
                    'surprise': points-(3*q[0 if side=='home' else 2]+q[1]),
                }))
            rows.append({'match_id': row['match_id'], **{'new_'+key: sides[0][key]-sides[1][key] for key in sides[0]}})
        for key, observation in pending:
            states[key].append(observation)
    return pd.DataFrame(rows)


def abbreviated_key(value):
    tokens = _tokens(value)
    initials = []
    while tokens and len(tokens[-1])==1:
        initials.insert(0,tokens.pop())
    return f'{tokens[-1]}|{initials[0]}' if tokens and initials else '|'.join(tokens)


def round_identity(value, draw_size=None):
    name = ''.join(c for c in str(value).lower() if c.isalnum())
    named = {'1stround':'round1','2ndround':'round2','3rdround':'round3','4thround':'round4',
             'quarterfinals':'qf','semifinals':'sf','thefinal':'f','final':'f','roundrobin':'rr'}
    if name in named:
        return named[name]
    size = pd.to_numeric(draw_size,errors='coerce')
    if name.startswith('r') and name[1:].isdigit() and pd.notna(size) and size>0:
        draw = 2**math.ceil(math.log2(size))
        remaining = int(name[1:])
        if remaining>=16 and remaining<=draw:
            return 'round'+str(int(round(math.log2(draw/remaining)))+1)
    return name


def atp_master(legacy, rich, quarantine_before=None):
    """One-to-one identity matching, never winner-based or best-price matching.

    Rich dates may be tournament dates: use Tennis-Data match dates as the master.
    Reject all ambiguous candidates instead of choosing with the known winner.
    """
    left = legacy.copy().reset_index(drop=True)
    right = rich.copy().reset_index(drop=True)
    for frame, p1, p2, date, round_, key in [
        (left,'Player_1','Player_2','Date','Round',abbreviated_key),
        (right,'player_1_name','player_2_name','match_date','round',_full_name_key),
    ]:
        frame['_k1'], frame['_k2'] = frame[p1].map(key), frame[p2].map(key)
        frame['_pair'] = [_pair_key(a,b) for a,b in zip(frame['_k1'],frame['_k2'])]
        frame['_join_date'] = pd.to_datetime(frame[date])
        frame['_join_year'] = frame['_join_date'].dt.year
        if 'draw_size' in frame:
            frame['_join_round'] = [round_identity(r,d) for r,d in zip(frame[round_],frame['draw_size'])]
        else:
            frame['_join_round'] = frame[round_].map(round_identity)
    left['_li'], right['_ri'] = left.index, right.index
    keys = ['_pair','_join_year','_join_round']
    candidates = left[['_li','_join_date',*keys]].merge(right[['_ri','_join_date',*keys]], on=keys, suffixes=('_l','_r'))
    candidates = candidates[(candidates['_join_date_l']-candidates['_join_date_r']).abs().le(pd.Timedelta(days=14))]
    unique = ~candidates['_li'].duplicated(False) & ~candidates['_ri'].duplicated(False)
    pairs = candidates[unique][['_li','_ri']]
    out = pd.DataFrame({
        'match_id': 'atp_td:'+left.index.astype(str), 'match_date': left['_join_date'],
        'match_status': left['Status'], 'surface': left['Surface'], 'tourney_name': left['Tournament'],
        'round': left['Round'], 'best_of': left['Best of'], 'indoor': left['Court'],
        'player_1_id': left['_k1'], 'player_2_id': left['_k2'],
        'player_1_name': left['Player_1'], 'player_2_name': left['Player_2'],
        'player_1_won': left['Winner'].eq(left['Player_1']).astype(int),
        'player_1_odds': left['B365_1'], 'player_2_odds': left['B365_2'],
        'player_1_rank': left['Rank_1'], 'player_2_rank': left['Rank_2'],
        'player_1_rank_points': left['Pts_1'], 'player_2_rank_points': left['Pts_2'],
    })
    attached = left[['_li','_k1']].merge(pairs,on='_li').merge(right,on='_ri',suffixes=('_l','_r'))
    swap = attached['_k1_l'].ne(attached['_k1_r']).to_numpy()
    # Mapping must identify both distinct participants, not just the pair string.
    assert left['_k1'].ne(left['_k2']).all()
    targets = attached['_li'].to_numpy()
    for suffix in ['age','ht','hand']:
        a, b = attached['player_1_'+suffix], attached['player_2_'+suffix]
        for side, values in [(1,np.where(swap,b,a)),(2,np.where(swap,a,b))]:
            out['player_'+str(side)+'_'+suffix] = np.nan if suffix!='hand' else ''
            out.loc[targets,'player_'+str(side)+'_'+suffix] = values
    for suffix in ['ace','df','svpt','1stIn','1stWon','2ndWon','bpSaved','bpFaced']:
        a, b = attached['postmatch_player_1_'+suffix], attached['postmatch_player_2_'+suffix]
        for side, values in [(1,np.where(swap,b,a)),(2,np.where(swap,a,b))]:
            column = 'postmatch_player_'+str(side)+'_'+suffix
            out[column] = np.nan
            out.loc[targets,column] = values
    for column in ['minutes','tourney_level']:
        out[column] = np.nan if column=='minutes' else ''
        out.loc[targets,column] = attached[column].to_numpy()
    # Agreement is reported after matching, never used to filter the population.
    rich_y = np.where(swap,1-attached['player_1_won'],attached['player_1_won'])
    disagreement = out.loc[targets,'player_1_won'].to_numpy()!=rich_y
    quality = {'round_matching_version':2,'master_rows':len(out),'unique_rich_matches':len(pairs),
               'ambiguous_candidate_links':int((~unique).sum()),
               'matched_label_disagreements':int(disagreement.sum()),
               'actual_date_master':'Tennis-Data; no tournament-start-date features',
               'join_uses_results':False}
    # A source mismatch is an integrity error, not a reason to remove losing bets.
    if disagreement.any():
        details = attached.loc[disagreement,['_li','match_date','player_1_name','player_2_name','round','player_1_won']].to_dict('records')
        disputed = targets[disagreement]
        completed_disputes = disputed[out.loc[disputed,'match_status'].eq('completed')]
        if quarantine_before is None or out.loc[completed_disputes,'match_date'].ge(pd.Timestamp(quarantine_before)).any():
            raise ValueError(f'ATP identity audit: {disagreement.sum()} source result disagreements; review before modelling: {details}')
        quality['conflicts'] = []
        for i, record, oriented_label in zip(disputed,details,rich_y[disagreement]):
            record.update(master_date=str(out.loc[i,'match_date'].date()),
                          master_player_1=out.loc[i,'player_1_name'],master_player_2=out.loc[i,'player_2_name'],
                          master_label=int(out.loc[i,'player_1_won']),rich_label_in_master_orientation=int(oriented_label),
                          original_status=out.loc[i,'match_status'])
            quality['conflicts'].append(record)
        out.loc[disputed,'match_status'] = 'source_conflict'
        stats_columns = [c for c in out if c.startswith('postmatch_')]
        out.loc[disputed,stats_columns] = np.nan
    return out, quality


def load_sport(root, folder, sport, protocol, progress=print):
    quality = {}
    if sport=='football':
        raw = pd.read_csv(root/'data/football/football_matches.csv.gz',low_memory=False)
        raw['match_date'] = pd.to_datetime(raw['match_date'])
        raw = raw[raw['match_date'].dt.year.le(2025)].copy()
        frame = build_football_features(raw,progress=progress).merge(football_extra(raw),on='match_id',validate='one_to_one')
        frame['_date'], frame['_id'] = frame['match_date'], frame['match_id']
        frame['_label'] = frame['result'].map({'H':0,'D':1,'A':2})
        frame['_status'] = np.where(frame['_label'].notna(),'completed','void')
        price_columns = ['B365H','B365D','B365A']
    elif sport=='atp':
        legacy = pd.read_csv(root/'data/atp_tennis.csv',low_memory=False)
        rich = pd.read_csv(root/'data/processed/atp_matches_enriched.csv.gz',low_memory=False)
        # Exclude new years before building the research table.
        legacy = legacy[pd.to_datetime(legacy['Date']).dt.year.le(2025)]
        rich = rich[pd.to_datetime(rich['match_date']).dt.year.le(2025)]
        amendment = folder/'atp_data_amendment.json'
        if not amendment.exists():
            raise ValueError('ATP source conflict handling must be recorded before returns')
        master, quality = atp_master(legacy,rich,quarantine_before='2023-01-01')
        quality['amendment_sha256'] = file_hash(amendment)
        quality['round_amendment_sha256'] = file_hash(folder/'atp_round_amendment.json')
        quality['noncompleted_amendment_sha256'] = file_hash(folder/'atp_noncompleted_conflicts_amendment.json')
        master.to_csv(folder/'atp_matched_master.csv.gz',index=False,compression='gzip')
        frame, audit = build_phase4_features(folder/'atp_matched_master.csv.gz',progress=progress)
        quality['feature_audit'] = audit
        frame['_id'], frame['_label'] = frame['_match_id'], 1-frame['_label']
        price_columns = ['_odds1','_odds2']
    else:
        frame = pd.read_parquet(root/'predictor_ufc/data/rigorous/processed/features.parquet',filters=[('event_date','<',pd.Timestamp('2025-01-01'))])
        frame['_date'], frame['_id'] = pd.to_datetime(frame['event_date']), frame['fight_id']
        frame['_label'] = 1-frame['y']
        frame['_status'] = np.where(frame['y'].notna(),'completed','void')
        price_columns = ['odds_1','odds_2']
        quality['quote_temporal_quality'] = frame['temporal_quality'].fillna('missing').value_counts().to_dict()
    prices = frame[price_columns].to_numpy(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        margins = (1/prices).sum(axis=1)
    low,high = protocol['overround_bounds']
    valid = np.isfinite(prices).all(axis=1)&(prices>1).all(axis=1)&(margins>=low)&(margins<=high)
    valid &= frame['_date'].dt.year.ge(protocol['sports'][sport]['first_train_year'])
    quality.update(rows_before_quote_filter=len(frame),eligible_rows=int(valid.sum()),
                   last_source_date=str(frame['_date'].max().date()))
    frame = frame.loc[valid].sort_values(['_date','_id']).reset_index(drop=True)
    if frame['_id'].duplicated().any():
        raise ValueError('Duplicate event outcome IDs')
    frame['_label'] = frame['_label'].fillna(0).astype(int)
    frame['_source_row_id'] = frame['_id']
    for j,column in enumerate(price_columns):
        frame[f'price_{j}'] = frame[column].astype(float)
    frame.to_parquet(folder/(sport+'_features.parquet'),index=False)
    return frame,quality


def design(frame, sport, candidate):
    classes = 3 if sport=='football' else 2
    odds = frame[[f'price_{j}' for j in range(classes)]].to_numpy(float)
    inv = 1/odds
    q = inv/inv.sum(axis=1,keepdims=True)
    z = np.log(q[:,:-1]/q[:,-1:])
    columns = {f'market_logit_{j}':z[:,j] for j in range(classes-1)}
    for j in range(classes-1):
        columns[f'market_curve_{j}'] = z[:,j]*np.abs(z[:,j])
        columns[f'market_margin_{j}'] = z[:,j]*inv.sum(axis=1)
    if candidate!='market':
        if sport=='football':
            names = FEATURE_COLUMNS
        elif sport=='atp':
            names = STRUCTURAL_EXTRA
        else:
            names = ['elo_diff','experience_diff','recent_win_rate_diff','age_diff','layoff_days_diff']
        columns.update({c:frame[c].to_numpy(float) for c in names})
    if candidate.startswith('enriched'):
        if sport=='football':
            extra = ['new_'+c for c in ['form_trend','goal_balance','target_balance','finishing_proxy',
                                      'defence_proxy','market_surprise','games14']]
            columns.update({c:frame[c].to_numpy(float) for c in extra})
            for c in ['elo_diff','target_for_diff','new_form_trend','new_finishing_proxy']:
                columns[c+'_curve'] = columns[c]*np.abs(columns[c])
                columns[c+'_market'] = columns[c]*(z[:,0]-z[:,1])
        elif sport=='atp':
            extra = [c for c in SURFACE_FEATURES if c not in STRUCTURAL_EXTRA and not c.startswith('market_')]
            columns.update({c:frame[c].to_numpy(float) for c in extra})
            n = np.minimum(frame['_p1_serve_sample'],frame['_p2_serve_sample']).to_numpy(float)
            reliability = n/(n+500)
            columns['reliable_serve'] = frame['serve_advantage'].to_numpy()*reliability
            columns['surface_disagreement'] = frame['surface_elo_diff']-frame['global_elo_diff']
            columns['fatigue_curve'] = frame['minutes_7d_diff']*np.abs(frame['minutes_7d_diff'])
            for surface in ['hard','clay','grass']:
                columns['new_market_'+surface] = z[:,0]*frame['is_'+surface].to_numpy()
        else:
            stats = ['career_win_rate_diff','sig_landed_pm_diff','sig_absorbed_pm_diff','sig_accuracy_diff',
                     'td_landed_p15_diff','td_accuracy_diff','sub_attempts_p15_diff','ctrl_share_diff','kd_p15_diff']
            n = np.minimum(frame['experience_1'],frame['experience_2']).to_numpy(float)
            reliability = n/(n+5)
            for c in stats:
                columns[c] = frame[c].to_numpy(float)
                columns[c+'_reliable'] = columns[c]*reliability
            columns['age_peak_diff'] = np.abs(frame['age_2']-30)-np.abs(frame['age_1']-30)
            columns['layoff_curve'] = frame['layoff_days_diff']*np.abs(frame['layoff_days_diff'])
            columns['market_reliability'] = z[:,0]*reliability
            columns['market_elo_gap'] = frame['elo_diff']*np.log(10)/400-z[:,0]
    x = np.nan_to_num(np.column_stack(list(columns.values())).astype(float),nan=0,posinf=0,neginf=0)
    return x,q,odds,list(columns)


def predict_year(frame,sport,candidate,year,protocol):
    spec = protocol['sports'][sport]
    cutoff = pd.Timestamp(year,1,1)-pd.Timedelta(days=protocol['embargo_days'])
    train = frame['_date'].lt(cutoff)&frame['_status'].eq('completed')
    test = frame['_date'].dt.year.eq(year)
    if train.sum()<spec['min_train'] or not test.any():
        raise ValueError(f'Insufficient frozen fold {sport}/{year}: {train.sum()} train, {test.sum()} test')
    x,q,odds,names = design(frame,sport,candidate)
    labels = frame['_label'].to_numpy(int)
    penalty = protocol['candidates'][candidate]
    if sport=='football':
        model = MarketResidual(penalty).fit(x[train],q[train],labels[train])
        p = model.predict(x[test],q[test])
    else:
        model = fit_symmetric(x[train],q[train],labels[train],penalty)
        p = predict_symmetric(model,x[test],q[test])
    rows = frame.loc[test,['_source_row_id','_date','_status','_label']].copy().reset_index(drop=True)
    rows['year'], rows['candidate'] = year,candidate
    for j in range(q.shape[1]):
        rows[f'p{j}'], rows[f'q{j}'], rows[f'price_{j}'] = p[:,j],q[test,j],odds[test,j]
    complete = rows['_status'].eq('completed')
    rows['model_loss'] = np.where(complete,-np.log(p[np.arange(len(p)),labels[test]]),np.nan)
    rows['market_loss'] = np.where(complete,-np.log(q[test][np.arange(len(p)),labels[test]]),np.nan)
    fold = {'candidate':candidate,'year':year,'train_rows':int(train.sum()),'test_rows':int(test.sum()),
            'train_max':str(frame.loc[train,'_date'].max().date()),'cutoff_exclusive':str(cutoff.date()),'features':names}
    return rows,fold


def selections(predictions, threshold, protocol):
    rows = predictions.copy()
    columns = [c for c in rows if c.startswith('price_')]
    odds = rows[columns].to_numpy(float)
    p = rows[[f'p{j}' for j in range(len(columns))]].to_numpy(float)
    rule = dict(protocol['selection'],minimum_ev_after_haircut=threshold)
    choice = select(p,odds,rule)
    safe, indices = np.maximum(choice,0),np.arange(len(rows))
    rows['selected'] = choice
    rows['odds'] = np.where(choice>=0,odds[indices,safe],np.nan)
    rows['probability'] = np.where(choice>=0,p[indices,safe],np.nan)
    rows['won'] = (choice==rows['_label']) & rows['_status'].eq('completed')
    return rows


def summarise(predictions,threshold,staking,protocol):
    bets = ledger(selections(predictions,threshold,protocol),staking,protocol['selection'])
    interval = uncertainty(bets,predictions['_date'],protocol['uncertainty'])
    yearly = {str(year):{'settled':int(group['settled'].sum()),'roi':roi(group,.02),
                         'profit_fraction':float((group['fraction']*group['return_0.02']).sum())}
              for year,group in bets.groupby('year')}
    best = max(yearly,key=lambda y:yearly[y]['profit_fraction']) if yearly else None
    result = {'matches':len(predictions),'selections':len(bets),'settled':int(bets['settled'].sum()),
              'void':int((~bets['settled']).sum()),'roi':{str(h):roi(bets,h) for h in [0.,.02,.05]},
              'uncertainty':interval,'yearly':yearly,'bankroll':dict(bets.attrs),
              'best_year':best,'roi_without_best_year':roi(bets[bets['year']!=int(best)],.02) if best else None,
              'model_log_loss':float(predictions['model_loss'].mean()),
              'market_log_loss':float(predictions['market_loss'].mean())}
    return result,bets


def run(root, sports=None, progress=print):
    folder = root/'models/three_sport_residual'
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name,expected in {**protocol['inputs'],**protocol['upstream_implementations']}.items():
        if file_hash(root/name)!=expected:
            raise ValueError(f'Frozen input changed: {name}')
    outputs = {}
    for sport in sports or protocol['sports']:
        target = folder/sport
        target.mkdir(exist_ok=True)
        previous = target/'report.json'
        if sport=='atp' and previous.exists() and json.loads(previous.read_text())['quality'].get('round_matching_version',1)==1:
            archive = target/'before_round_fix'
            if not archive.exists():
                archive.mkdir()
                for artifact in list(target.iterdir()):
                    if artifact.is_file():
                        shutil.copy2(artifact,archive/artifact.name)
        progress(f'{sport}: preparation des donnees')
        frame,quality = load_sport(root,folder,sport,protocol,progress)
        spec = protocol['sports'][sport]
        development,folds,metrics = [],[],{}
        for candidate in protocol['candidates']:
            batches = []
            for year in spec['development']:
                batch,fold = predict_year(frame,sport,candidate,year,protocol)
                batches.append(batch); folds.append(fold)
            predictions = pd.concat(batches,ignore_index=True)
            metrics[candidate] = float(predictions['model_loss'].mean())
            development.append(predictions)
            progress(f'{sport}: {candidate}, log-loss developpement {metrics[candidate]:.6f}')
        chosen = min(metrics,key=metrics.get)
        pd.concat(development,ignore_index=True).to_parquet(target/'development_predictions.parquet',index=False)
        tune_batches = []
        for year in spec['tuning']:
            batch,fold = predict_year(frame,sport,chosen,year,protocol)
            tune_batches.append(batch); folds.append(fold)
        tuning = pd.concat(tune_batches,ignore_index=True)
        tuning.to_parquet(target/'tuning_predictions.parquet',index=False)
        threshold_reports = {str(t):summarise(tuning,t,'flat',protocol)[0] for t in protocol['thresholds']}
        eligible = [t for t in protocol['thresholds'] if threshold_reports[str(t)]['settled']>=50]
        threshold = max(eligible,key=lambda t:threshold_reports[str(t)]['uncertainty']['ci95'][0]) if eligible else .02
        lock = {'chosen_model':chosen,'threshold':threshold,'tuning_passed':bool(eligible),
                'development_log_loss':metrics,'threshold_diagnostics':threshold_reports,
                'protocol_sha256':file_hash(protocol_path),'real_money_authorised':False}
        # Lock physically saved before evaluation returns or predictions.
        (target/'selection_lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
        progress(f'{sport}: choix fige {chosen}, seuil {threshold:.0%}; evaluation chronologique')
        batches = []
        for year in spec['evaluation']:
            batch,fold = predict_year(frame,sport,chosen,year,protocol)
            batches.append(batch); folds.append(fold)
        evaluation = pd.concat(batches,ignore_index=True)
        evaluation.to_parquet(target/'evaluation_predictions.parquet',index=False)
        results = {}
        for stake in ['flat','quarter_kelly']:
            summary,bets = summarise(evaluation,threshold,stake,protocol)
            primary = summary['roi']['0.02']; stressed=summary['roi']['0.05']
            lower = summary['uncertainty']['family_lower']; without=summary['roi_without_best_year']
            summary['checks'] = {'tuning_enough_bets':bool(eligible),'200_settled':summary['settled']>=200,
                'positive_2pct':primary is not None and primary>0,'positive_5pct':stressed is not None and stressed>0,
                'positive_adjusted_lower':lower is not None and lower>0,
                'two_positive_years':sum(v['roi'] is not None and v['roi']>0 for v in summary['yearly'].values())>=2,
                'positive_without_best_year':without is not None and without>0}
            summary['research_gate_passed'] = all(summary['checks'].values())
            bets.attrs = {}
            bets.to_parquet(target/(stake+'_bets.parquet'),index=False)
            results[stake] = summary
            progress(f"{sport}/{stake}: {summary['settled']} paris regles, ROI={primary}")
        report = {'sport':sport,'quality':quality,'selection':lock,'evaluation':results,'folds':folds,
                  'implementation_sha256':file_hash(Path(__file__)), 'evidence':protocol['evidence'],
                  'real_money_authorised':False,'ufc_2025_onward_evaluated':False,
                  'status':'PAPER_REPLICATION_ONLY' if results['flat']['research_gate_passed'] else 'NO_ROBUST_CANDIDATE'}
        (target/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        outputs[sport] = report
    return outputs
