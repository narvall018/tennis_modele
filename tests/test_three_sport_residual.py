import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.three_sport_residual import (
    abbreviated_key, atp_master, design, football_extra, predict_year, round_identity, selections, summarise,
)
from src.backtesting.wta_price_residual import fit_symmetric, predict_symmetric

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT/'models/three_sport_residual/protocol.json').read_text())


def ufc_frame():
    rng=np.random.default_rng(9)
    n=180
    q=rng.uniform(.3,.7,n)
    frame=pd.DataFrame({'_date':pd.to_datetime(['2013-02-01']*60+['2014-02-01']*60+['2015-02-01']*60),
                        '_source_row_id':np.arange(n).astype(str),'_status':'completed','_label':rng.integers(0,2,n),
                        'price_0':1/(q*1.05),'price_1':1/((1-q)*1.05)})
    for column in ['elo_diff','experience_diff','recent_win_rate_diff','age_diff','layoff_days_diff',
                   'career_win_rate_diff','sig_landed_pm_diff','sig_absorbed_pm_diff','sig_accuracy_diff',
                   'td_landed_p15_diff','td_accuracy_diff','sub_attempts_p15_diff','ctrl_share_diff','kd_p15_diff']:
        frame[column]=rng.normal(size=n)
    for column in ['experience_1','experience_2','age_1','age_2']:
        frame[column]=rng.uniform(5,35,n)
    return frame


def test_binary_design_and_fit_are_swap_symmetric():
    frame=ufc_frame()
    swapped=frame.copy()
    swapped['price_0'],swapped['price_1']=frame['price_1'],frame['price_0']
    for c in frame:
        if c.endswith('_diff'):
            swapped[c]=-frame[c]
    for c in ['experience','age']:
        swapped[c+'_1'],swapped[c+'_2']=frame[c+'_2'],frame[c+'_1']
    for candidate in ['market','base','enriched','enriched_strong']:
        x,q,_,_=design(frame,'ufc',candidate)
        sx,sq,_,_=design(swapped,'ufc',candidate)
        np.testing.assert_allclose(x,-sx,atol=1e-12)
        model=fit_symmetric(x,q,frame['_label'],.01)
        p=predict_symmetric(model,x,q)
        sp=predict_symmetric(model,sx,sq)
        np.testing.assert_allclose(p,sp[:,::-1],atol=1e-12)


def test_evaluation_results_cannot_affect_fitted_probabilities(protocol):
    protocol['sports']['ufc']['min_train']=60
    frame=ufc_frame()
    a,fold=predict_year(frame,'ufc','enriched',2015,protocol)
    changed=frame.copy()
    changed.loc[changed['_date'].dt.year.eq(2015),'_label']=1-changed.loc[changed['_date'].dt.year.eq(2015),'_label']
    changed.loc[changed['_date'].dt.year.eq(2015),'_status']='void'
    b,_=predict_year(changed,'ufc','enriched',2015,protocol)
    np.testing.assert_array_equal(a['p0'],b['p0'])
    assert pd.Timestamp(fold['train_max'])<pd.Timestamp(fold['cutoff_exclusive'])


def test_allowlist_excludes_result_current_stats_and_closing_prices():
    frame=ufc_frame()
    changed=frame.assign(y=1,postmatch_sig_landed=999,PSCH=1.01,Max_1=100,stance_matchup=3)
    np.testing.assert_array_equal(design(frame,'ufc','enriched')[0],design(changed,'ufc','enriched')[0])


def test_football_new_history_is_prior_day_only():
    rows=[]
    for i,date in enumerate(['2020-01-01','2020-01-02','2020-01-02','2020-01-03']):
        rows.append(dict(match_id=str(i),match_date=pd.Timestamp(date),country='X',home_team='A',away_team='B',
                         result='H',home_goals=2.,away_goals=0.,postmatch_home_shots_on_target=4.,
                         postmatch_away_shots_on_target=1.,B365H=2.,B365D=3.,B365A=4.))
    frame=pd.DataFrame(rows)
    a=football_extra(frame)
    changed=frame.copy()
    changed.loc[[1,2],'result']='A'
    changed.loc[[1,2],'home_goals']=0.
    changed.loc[[1,2],'away_goals']=8.
    changed.loc[[1,2],'postmatch_home_shots_on_target']=19.
    b=football_extra(changed)
    pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])
    np.testing.assert_array_equal(a.iloc[1,1:],a.iloc[2,1:])
    assert a.loc[3,'new_form_trend']!=b.loc[3,'new_form_trend'] or a.loc[3,'new_goal_balance']!=b.loc[3,'new_goal_balance']


def atp_sources():
    legacy=pd.DataFrame([dict(Date='2020-01-05',Round='Final',Player_1='Alpha A.',Player_2='Beta B.',
                             Winner='Alpha A.',Status='completed',Surface='Hard',Tournament='Example',
                             Court='Outdoor',Rank_1=10,Rank_2=20,Pts_1=2000,Pts_2=1000,B365_1=1.8,B365_2=2.1,
                             **{'Best of':3})])
    rich=pd.DataFrame([dict(match_date='2020-01-01',round='F',player_1_name='Alice Alpha',player_2_name='Bob Beta',
                           player_1_won=1,minutes=85,tourney_level='250',player_1_age=25,player_2_age=28,
                           player_1_ht=185,player_2_ht=180,player_1_hand='R',player_2_hand='L')])
    for side in [1,2]:
        for stat in ['ace','df','svpt','1stIn','1stWon','2ndWon','bpSaved','bpFaced']:
            rich[f'postmatch_player_{side}_{stat}']=10*side
    return legacy,rich


def test_atp_uses_match_date_and_fixed_book_not_average():
    assert abbreviated_key('Viloca J.A.')=='viloca|j'
    assert abbreviated_key('Marin J.A.')=='marin|j'
    legacy,rich=atp_sources()
    legacy['Odd_1']=99
    master,audit=atp_master(legacy,rich)
    assert master['match_date'].iloc[0]==pd.Timestamp('2020-01-05')
    assert master['player_1_odds'].iloc[0]==1.8
    assert master['postmatch_player_1_svpt'].iloc[0]==10
    assert audit['join_uses_results'] is False


def test_atp_first_round_matches_using_draw_size_without_losing_digits():
    assert round_identity('1st Round')==round_identity('R32',32)
    assert round_identity('1st Round')==round_identity('R64',48)
    assert round_identity('3rd Round')==round_identity('R32',128)
    assert round_identity('2nd Round')!=round_identity('R32',128)
    legacy,rich=atp_sources()
    legacy['Round']='1st Round'
    rich['round']='R32'
    rich['draw_size']=32
    _,audit=atp_master(legacy,rich)
    assert audit['unique_rich_matches']==1


def test_atp_ambiguous_matches_remain_in_population_without_rich_stats():
    legacy,rich=atp_sources()
    master,audit=atp_master(legacy,pd.concat([rich,rich],ignore_index=True))
    assert len(master)==1
    assert master['postmatch_player_1_svpt'].isna().all()
    assert audit['ambiguous_candidate_links']==2


def test_atp_disagreement_fails_audit_instead_of_filtering_outcome():
    legacy,rich=atp_sources()
    rich['player_1_won']=0
    with pytest.raises(ValueError,match='source result disagreements'):
        atp_master(legacy,rich)
    master,audit=atp_master(legacy,rich,quarantine_before='2023-01-01')
    assert len(master)==1 and audit['matched_label_disagreements']==1
    assert master['player_1_odds'].iloc[0]==legacy['B365_1'].iloc[0]
    assert master['match_status'].iloc[0]=='source_conflict'
    with pytest.raises(ValueError,match='source result disagreements'):
        atp_master(legacy,rich,quarantine_before='2019-01-01')
    legacy['Status']='retired'
    master,_=atp_master(legacy,rich,quarantine_before='2019-01-01')
    assert len(master)==1 and master['match_status'].iloc[0]!='completed'
    assert master['player_1_odds'].iloc[0]==1.8


def test_selection_and_same_day_exposure_do_not_use_status(protocol):
    rows=pd.DataFrame({'_date':pd.to_datetime(['2020-01-01']*12),'year':2020,'_status':'completed',
                       '_label':0,'price_0':2.,'price_1':1.9,'p0':.65,'p1':.35,
                       'model_loss':.6,'market_loss':.7},index=range(12))
    a=selections(rows,.02,protocol)
    b=selections(rows.assign(_status='void',_label=1),.02,protocol)
    np.testing.assert_array_equal(a['selected'],b['selected'])
    report,bets=summarise(rows,.02,'flat',protocol)
    assert bets['fraction'].sum()==pytest.approx(.02)
    assert bets['bankroll_before'].nunique()==1
    assert report['settled']==12
