"""Registered, exploratory football total-goals intensity experiment."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincinv

from src.backtesting.football_cross_market import MarketResidual, file_hash
from src.backtesting.three_sport_residual import summarise


INTENSITY_KEYS = ['gf','ga','tf','ta','finishing','goal_trend','games14','experience']
INTENSITY_COLUMNS = [f'{key}_{operation}' for key in INTENSITY_KEYS for operation in ['sum','absdiff']]


def intensity_features(raw, progress=None):
    frame=raw.copy()
    frame['match_date']=pd.to_datetime(frame['match_date'])
    histories=defaultdict(lambda:deque(maxlen=30))
    records=[]
    for day_no,(date,day) in enumerate(frame.sort_values(['match_date','match_id']).groupby('match_date',sort=True)):
        if progress and day_no%1500==0:
            progress(f'Totaux: descripteurs {date.date()}, {len(records)} matchs')
        pending=[]
        for row in day.to_dict('records'):
            descriptors=[]
            for side,opponent in [('home','away'),('away','home')]:
                key=(row['country'],row[side+'_team'])
                past=list(histories[key])
                def mean(column,n=10,prior=0.):
                    values=[r[column] for r in past[-n:] if np.isfinite(r[column])]
                    return float(np.mean(values)) if values else prior
                descriptors.append({
                    'gf':mean('gf',prior=1.3),'ga':mean('ga',prior=1.3),
                    'tf':mean('tf',prior=4.),'ta':mean('ta',prior=4.),
                    'finishing':mean('finishing'),
                    'goal_trend':mean('gf',3,1.3)-mean('gf',10,1.3),
                    'games14':sum(pd.Timedelta(0)<date-r['date']<=pd.Timedelta(days=14) for r in past),
                    'experience':len(past),
                })
                gf,ga=row[side+'_goals'],row[opponent+'_goals']
                tf,ta=row['postmatch_'+side+'_shots_on_target'],row['postmatch_'+opponent+'_shots_on_target']
                pending.append((key,dict(date=date,gf=gf,ga=ga,tf=tf,ta=ta,finishing=gf-.3*tf)))
            record={'match_id':row['match_id']}
            for key in INTENSITY_KEYS:
                a,b=descriptors[0][key],descriptors[1][key]
                record[key+'_sum'],record[key+'_absdiff']=a+b,abs(a-b)
            records.append(record)
        for key,values in pending:
            histories[key].append(values)
    return pd.DataFrame(records)


class PoissonOffset:
    def __init__(self,penalty=.01):
        self.penalty=penalty

    def _design(self,x):
        return np.column_stack([np.ones(len(x)),(x-self.mean_)/self.scale_])

    def fit(self,x,offset_mean,counts):
        self.mean_=x.mean(axis=0)
        self.scale_=x.std(axis=0)
        self.scale_[self.scale_<1e-10]=1
        design=self._design(x)
        offset=np.log(offset_mean)
        def objective(weights):
            raw=offset+design@weights
            eta=np.clip(raw,-15,15)
            rate=np.exp(eta)
            value=np.mean(rate-counts*eta)+self.penalty*(weights@weights)/2
            gradient=design.T@((rate-counts)*((raw>-15)&(raw<15)))/len(x)+self.penalty*weights
            return value,gradient
        solution=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',
                          options={'maxiter':1000,'ftol':1e-12})
        if not solution.success:
            raise RuntimeError(solution.message)
        self.weights_=solution.x
        return self

    def predict_mean(self,x,offset_mean):
        return np.exp(np.clip(np.log(offset_mean)+self._design(x)@self.weights_,-15,15))


def inputs(frame,descriptors=True):
    odds=frame[['B365>2.5','B365<2.5']].to_numpy(float)
    inv=1/odds
    q=inv/inv.sum(axis=1,keepdims=True)
    mean=gammaincinv(3,q[:,0])
    z=np.log(q[:,0]/q[:,1])
    names=['market_logit','market_curve','market_margin']
    columns=[z,z*np.abs(z),inv.sum(axis=1)]
    if descriptors:
        names+=INTENSITY_COLUMNS
        columns.extend(frame[c].to_numpy(float) for c in INTENSITY_COLUMNS)
    return np.column_stack(columns),q,odds,mean,names


def predict_fold(frame,candidate,year,protocol):
    spec=protocol['candidates'][candidate]
    cutoff=pd.Timestamp(year,1,1)-pd.Timedelta(days=protocol['embargo_days'])
    train=frame['match_date'].lt(cutoff)
    test=frame['match_date'].dt.year.eq(year)
    if train.sum()<protocol['min_train'] or not test.any():
        raise ValueError(f'Insufficient fold {year}')
    x,q,odds,mean,names=inputs(frame,spec['descriptors'])
    labels=np.where(frame['total_goals'].ge(3),0,1)
    if spec['family']=='binary_market_residual':
        model=MarketResidual(spec['penalty']).fit(x[train],q[train],labels[train])
        probability=model.predict(x[test],q[test])
    else:
        model=PoissonOffset(spec['penalty']).fit(x[train],mean[train],frame.loc[train,'total_goals'].to_numpy(float))
        over=np.clip(gammainc(3,model.predict_mean(x[test],mean[test])),1e-9,1-1e-9)
        probability=np.column_stack([over,1-over])
    output=pd.DataFrame({'_source_row_id':frame.loc[test,'match_id'].to_numpy(),
                         '_date':frame.loc[test,'match_date'].to_numpy(),'_status':'completed',
                         '_label':labels[test],'year':year,'candidate':candidate})
    for j in range(2):
        output[f'p{j}'],output[f'q{j}'],output[f'price_{j}']=probability[:,j],q[test,j],odds[test,j]
    output['model_loss']=-np.log(probability[np.arange(test.sum()),labels[test]])
    output['market_loss']=-np.log(q[test][np.arange(test.sum()),labels[test]])
    audit={'year':year,'candidate':candidate,'train_rows':int(train.sum()),'test_rows':int(test.sum()),
           'cutoff_exclusive':str(cutoff.date()),'train_max':str(frame.loc[train,'match_date'].max().date()),'features':names}
    return output,audit


def run(root,progress=print):
    folder=root/'models/football_totals_count'
    protocol_path=folder/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    amendment_path=folder/'availability_amendment.json'
    amendment=json.loads(amendment_path.read_text())
    if amendment['original_protocol_sha256']!=file_hash(protocol_path):
        raise ValueError('Availability amendment does not match the original protocol')
    for field in ['development','tuning','evaluation']:
        protocol[field]=amendment[field]
    path=root/protocol['input']
    if file_hash(path)!=protocol['input_sha256']:
        raise ValueError('Frozen source changed')
    raw=pd.read_csv(path,low_memory=False)
    raw['match_date']=pd.to_datetime(raw['match_date'])
    raw=raw[raw['match_date'].dt.year.le(max(protocol['evaluation']))].copy()
    if raw['match_id'].duplicated().any():
        raise ValueError('Duplicate source IDs')
    cache=folder/'features.parquet'
    if cache.exists() and file_hash(cache)==amendment['features_sha256']:
        progress('Totaux: reutilisation des descripteurs verifies par SHA256')
        frame=pd.read_parquet(cache)
    else:
        descriptors=intensity_features(raw,progress)
        frame=raw.merge(descriptors,on='match_id',validate='one_to_one')
        odds=frame[['B365>2.5','B365<2.5']].to_numpy(float)
        with np.errstate(divide='ignore',invalid='ignore'):
            margins=(1/odds).sum(axis=1)
        low,high=protocol['overround_bounds']
        good=np.isfinite(odds).all(axis=1)&(odds>1).all(axis=1)&(margins>=low)&(margins<=high)
        frame=frame.loc[good].sort_values(['match_date','match_id']).reset_index(drop=True)
        frame.to_parquet(cache,index=False)
    if frame['total_goals'].isna().any() or not np.isfinite(frame[INTENSITY_COLUMNS]).all().all():
        raise ValueError('Missing outcomes or nonfinite descriptors')
    metrics,folds,development={},[],[]
    def predict_years(candidate,years):
        batches=[]
        for year in years:
            batch,audit=predict_fold(frame,candidate,year,protocol)
            batches.append(batch);folds.append(audit)
        return pd.concat(batches,ignore_index=True)
    for candidate in protocol['candidates']:
        predictions=predict_years(candidate,protocol['development'])
        metrics[candidate]={'log_loss':float(predictions['model_loss'].mean()),
                            'market_log_loss':float(predictions['market_loss'].mean())}
        development.append(predictions)
        progress(f'Totaux: {candidate}, log-loss {metrics[candidate]["log_loss"]:.6f}')
    pd.concat(development,ignore_index=True).to_parquet(folder/'development_predictions.parquet',index=False)
    candidate=min(metrics,key=lambda name:metrics[name]['log_loss'])
    tuning=predict_years(candidate,protocol['tuning'])
    tuning.to_parquet(folder/'tuning_predictions.parquet',index=False)
    choices={str(t):summarise(tuning,t,'flat',protocol)[0] for t in protocol['thresholds']}
    eligible=[t for t in protocol['thresholds'] if choices[str(t)]['settled']>=50]
    threshold=max(eligible,key=lambda t:choices[str(t)]['uncertainty']['ci95'][0]) if eligible else .02
    lock={'candidate':candidate,'threshold':threshold,'development':metrics,'threshold_diagnostics':choices,
          'tuning_passed':bool(eligible),'protocol_sha256':file_hash(protocol_path),
          'availability_amendment_sha256':file_hash(amendment_path),'real_money_authorised':False}
    (folder/'selection_lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
    progress(f'Totaux: choix fige {candidate}, seuil {threshold:.0%}')
    evaluation=predict_years(candidate,protocol['evaluation'])
    evaluation.to_parquet(folder/'evaluation_predictions.parquet',index=False)
    results={}
    for staking in ['flat','quarter_kelly']:
        summary,bets=summarise(evaluation,threshold,staking,protocol)
        lower=summary['uncertainty']['family_lower'];without=summary['roi_without_best_year']
        summary['research_gate_passed']=bool(eligible) and summary['settled']>=200 and all(
            value is not None and value>0 for value in [summary['roi']['0.02'],summary['roi']['0.05'],lower,without]
        ) and sum(v['roi'] is not None and v['roi']>0 for v in summary['yearly'].values())>=2
        bets.attrs={}
        bets.to_parquet(folder/(staking+'_bets.parquet'),index=False)
        results[staking]=summary
        progress(f'Totaux/{staking}: {summary["settled"]} paris, ROI={summary["roi"]["0.02"]}')
    report={'selection':lock,'evaluation':results,'folds':folds,'source_rows':len(raw),'eligible_rows':len(frame),
            'implementation_sha256':file_hash(Path(__file__)),
            'dependency_sha256':file_hash(root/'src/backtesting/three_sport_residual.py'),
            'features_sha256':file_hash(cache),'periods':{k:protocol[k] for k in ['development','tuning','evaluation']},
            'evidence':protocol['evidence'],'real_money_authorised':False,
            'status':'PAPER_REPLICATION_ONLY' if results['flat']['research_gate_passed'] else 'NO_ROBUST_CANDIDATE'}
    (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return report
