"""Pre-registered economic rather than predictive selection; research only."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.backtesting.football_cross_market import file_hash
from src.backtesting.football_totals_count import predict_fold
from src.backtesting.three_sport_residual import predict_year, summarise


def choose(scores, minimum_settled=50):
    eligible=[s for s in scores if s['summary']['settled']>=minimum_settled
              and s['summary']['uncertainty']['ci95'][0] is not None]
    return max(eligible,key=lambda s:s['summary']['uncertainty']['ci95'][0]) if eligible else None


def admitted(summary):
    roi=summary['roi']['0.02']
    lower=summary['uncertainty']['ci95'][0]
    return roi is not None and roi>0 and lower is not None and lower>0


def gate(summary,tuning_admitted):
    values=[summary['roi']['0.02'],summary['roi']['0.05'],
            summary['uncertainty']['family_lower'],summary['roi_without_best_year']]
    return (tuning_admitted and summary['settled']>=200
            and all(v is not None and v>0 for v in values)
            and sum(y['roi'] is not None and y['roi']>0 for y in summary['yearly'].values())>=2)


def run(root,progress=print):
    folder=root/'models/economic_selection'
    protocol_path=folder/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    for name,expected in protocol['frozen_files'].items():
        if file_hash(root/name)!=expected:
            raise ValueError(f'Frozen file changed: {name}')
    three=json.loads((root/'models/three_sport_residual/protocol.json').read_text())
    totals=json.loads((root/'models/football_totals_count/protocol.json').read_text())
    reports={}
    for market,spec in protocol['markets'].items():
        target=folder/market
        target.mkdir(exist_ok=True)
        source=(root/'models/football_totals_count/features.parquet' if market=='football_totals'
                else root/f'models/three_sport_residual/{market}_features.parquet')
        frame=pd.read_parquet(source)
        if market=='ufc' and frame['_date'].dt.year.max()>=2025:
            raise ValueError('UFC reserved holdout must not enter this experiment')
        folds=[]
        def predictions(model,years):
            batches=[]
            for year in years:
                batch,fold=(predict_fold(frame,model,year,totals) if market=='football_totals'
                            else predict_year(frame,market,model,year,three))
                batches.append(batch);folds.append(fold)
            return pd.concat(batches,ignore_index=True)
        scores=[]
        for model in spec['models']:
            tuning=predictions(model,spec['tuning'])
            tuning.to_parquet(target/(model+'_tuning_predictions.parquet'),index=False)
            for threshold in protocol['thresholds']:
                summary,_=summarise(tuning,threshold,'flat',protocol)
                scores.append({'model':model,'threshold':threshold,'summary':summary})
            progress(f'{market}: economie de reglage examinee pour {model}')
        chosen=choose(scores)
        lock={'market':market,'scores':scores,'chosen':chosen,
              'tuning_admitted':chosen is not None and admitted(chosen['summary']),
              'protocol_sha256':file_hash(protocol_path),'real_money_authorised':False}
        # Persist the complete economic choice before evaluating the selected pair.
        (target/'selection_lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
        evaluations={}
        if chosen is not None:
            progress(f"{market}: choix fige {chosen['model']}, seuil {chosen['threshold']:.0%}; admission={lock['tuning_admitted']}")
            evaluated=predictions(chosen['model'],spec['evaluation'])
            evaluated.to_parquet(target/'evaluation_predictions.parquet',index=False)
            evaluation_protocol=deepcopy(protocol)
            evaluation_protocol['uncertainty']['samples']=protocol['uncertainty']['evaluation_samples']
            for stake in ['flat','quarter_kelly']:
                summary,bets=summarise(evaluated,chosen['threshold'],stake,evaluation_protocol)
                summary['research_gate_passed']=gate(summary,lock['tuning_admitted'])
                bets.attrs={}
                bets.to_parquet(target/(stake+'_bets.parquet'),index=False)
                evaluations[stake]=summary
                progress(f"{market}/{stake}: n={summary['settled']}, ROI={summary['roi']['0.02']}")
        status=('NO_ELIGIBLE_TUNING_PAIR' if chosen is None else
                'PAPER_REPLICATION_ONLY' if evaluations['flat']['research_gate_passed'] else
                'NO_ROBUST_CANDIDATE')
        report={'market':market,'status':status,'selection':lock,'evaluation':evaluations,'folds':folds,
                'implementation_sha256':file_hash(Path(__file__)),
                'source_features_sha256':file_hash(source),
                'fit_protocol_sha256':file_hash(root/('models/football_totals_count/protocol.json' if market=='football_totals'
                                                      else 'models/three_sport_residual/protocol.json')),
                'evidence':protocol['evidence'],'real_money_authorised':False}
        (target/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        reports[market]=report
    (folder/'report.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return reports
