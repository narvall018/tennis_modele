from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import tennis_strategy as engine
from src.app import tennis_strategy_ledger as ledger
from src.backtesting.tennis_phase4 import build_phase4_features, SURFACE_FEATURES, MLP_INVARIANT_FEATURES

NOW = pd.Timestamp('2026-09-16T10:00:00Z')


def fixture():
    return {'tour':'ATP', 'singles_main_draw':True, 'start':'2026-09-16T18:00:00Z',
            'quote_at':NOW.isoformat(), 'player_1':'Alpha A.', 'player_2':'Beta B.',
            'player_1_rank':10, 'player_2_rank':20, 'player_1_rank_points':1000,
            'player_2_rank_points':500, 'odds_1':2.0, 'odds_2':1.85,
            'bookmaker':'winamax_fr', 'surface':'Hard', 'indoor':'Indoor',
            'round':'The Final', 'best_of':3, 'level':'250', 'tournament':'Test'}


def candidate(key='match'):
    return {'strategy_id':engine.STRATEGY_ID, 'eligible':True, 'event_key':key,
            'computed_at':NOW.isoformat(), 'fixture':fixture(), 'selected_side':0,
            'probability':.6, 'odds':2.0, 'pick':'Alpha A.', 'real_money_authorised':False}


@pytest.mark.parametrize('field,value', [
    ('start','2026-09-16T09:00:00Z'), ('start','2026-09-16T18:00:00'),
    ('quote_at', None), ('quote_at','2026-09-16T10:01:00Z'),
    ('quote_at','2026-09-16T09:44:00Z'), ('bookmaker','bet365'),
    ('tour','WTA'), ('singles_main_draw',False), ('surface','Unknown'),
    ('indoor','Unknown'), ('round','Unknown'), ('odds_1',float('nan')),
    ('odds_1',1), ('odds_1',10), ('player_2','Alpha A.'),
])
def test_bad_fixtures_never_become_selections(field,value):
    f=fixture();f[field]=value
    with pytest.raises(ValueError): engine.validate_fixture(f,NOW)


def test_valid_pair_and_data_freshness():
    np.testing.assert_allclose(engine.validate_fixture(fixture(),NOW),[2,1.85])
    assert not engine.freshness_reasons({'model_year':2026,'history_last_date':'2026-09-15'},NOW)
    assert engine.freshness_reasons({'model_year':2026,'history_last_date':'2026-08-29'},NOW)
    assert engine.freshness_reasons({'model_year':2026,'history_last_date':'2026-09-17'},NOW)
    assert engine.freshness_reasons({'model_year':2025,'history_last_date':'2026-09-15'},NOW)
    assert engine.freshness_reasons({'model_year':2026,'history_last_date':'2026-09-08'},
                                    pd.Timestamp('2026-09-15T22:30:00Z'))  # Already September 16 in Paris.


def api_event():
    return {'id':'one', 'sport_key':'tennis_atp_test', 'home_team':'Alex Alpha',
            'away_team':'Bob Beta','commence_time':fixture()['start'],
            'bookmakers':[{'key':'winamax_fr','last_update':NOW.isoformat(),
                           'markets':[{'key':'h2h','outcomes':[{'name':'Alex Alpha','price':2.0},
                                                             {'name':'Bob Beta','price':1.85}]}]}]}


def test_api_quotes_are_french_prematch_complete_and_timestamped():
    event=api_event()
    assert len(engine.french_quotes([event],NOW))==1
    for mutation in ['missing_time','future_time','started','foreign','draw','wta']:
        e=deepcopy(event)
        if mutation=='missing_time': del e['bookmakers'][0]['last_update']
        if mutation=='future_time': e['bookmakers'][0]['last_update']='2026-09-16T10:01:00Z'
        if mutation=='started': e['commence_time']='2026-09-16T09:00:00Z'
        if mutation=='foreign': e['bookmakers'][0]['key']='winamax'
        if mutation=='draw': e['bookmakers'][0]['markets'][0]['outcomes'].append({'name':'Draw','price':10})
        if mutation=='wta': e['sport_key']='tennis_wta_test'
        assert engine.french_quotes([e],NOW)==[],mutation


def test_api_mapping_cannot_transfer_prices_to_other_players():
    f=fixture(); f['api_pair']=engine.french_quotes([api_event()],NOW)[0]
    engine.validate_fixture(f,NOW)
    f['player_1']='Different D.'
    with pytest.raises(ValueError): engine.validate_fixture(f,NOW)


def historical_row():
    r={'match_id':'past', 'match_date':pd.Timestamp('2026-09-01'), 'match_status':'completed',
       'surface':'Hard', 'tourney_name':'Test', 'round':'R32', 'indoor':'O', 'best_of':3,
       'tourney_level':'250', 'player_1_won':1, 'minutes':90.}
    for side,(name,key) in enumerate([('Alpha A.','alpha|a'),('Beta B.','beta|b')],1):
        r.update({f'player_{side}_name':name,f'player_{side}_id':key,
                  f'player_{side}_odds':2.,f'player_{side}_rank':10*side,
                  f'player_{side}_rank_points':1000,f'player_{side}_age':25.,
                  f'player_{side}_ht':185.,f'player_{side}_hand':'R'})
        for field,value in {'svpt':60,'1stIn':35,'1stWon':25,'2ndWon':15,'ace':5,
                            'df':2,'bpSaved':3,'bpFaced':5}.items():
            r[f'postmatch_player_{side}_{field}']=value
    return r


def test_live_features_use_exact_builder_and_exclude_same_day_and_future(tmp_path):
    past=historical_row()
    f=fixture()
    historical=pd.DataFrame([past])
    clean=engine.fixture_feature_row(historical,f)
    poisoned=[]
    for i,d in enumerate(['2026-09-16','2026-09-17']):
        row=dict(past,match_id=f'future{i}',match_date=pd.Timestamp(d),player_1_won=0,minutes=1000)
        poisoned.append(row)
    changed=engine.fixture_feature_row(pd.DataFrame([past,*poisoned]),f)
    pd.testing.assert_frame_equal(clean,changed)
    target=dict(past,match_id='prospective:fixture',match_date=pd.Timestamp('2026-09-16'),
                match_status='upcoming',player_1_won=0,minutes=np.nan,round='F',indoor='I',
                player_1_odds=2.,player_2_odds=1.85,player_1_rank_points=1000,player_2_rank_points=500,
                player_1_age=25+15/365.25,player_2_age=25+15/365.25)
    path=tmp_path/'historical.csv'
    pd.DataFrame([past,target]).to_csv(path,index=False)
    expected,_=build_phase4_features(path,progress=lambda _:None)
    expected=expected.iloc[[-1]].reset_index(drop=True)
    # Same frozen feature formulas, including corrected live context encoding.
    for col in [*SURFACE_FEATURES,*MLP_INVARIANT_FEATURES,'_p1_serve_sample','_p2_serve_sample']:
        np.testing.assert_allclose(clean[col],expected[col])
    assert clean['is_indoor'].iloc[0]==1
    assert clean['round_progress'].iloc[0]==1


def test_ledger_is_private_reserves_funds_and_prevents_duplicates(tmp_path):
    path=tmp_path/'paper.sqlite3'
    ledger.initialise(path,'a',1000,NOW);ledger.initialise(path,'b',200,NOW)
    assert ledger.record(path,'a',candidate(),NOW)==2.5
    s=ledger.state(path,'a',NOW)
    assert s['reserved_cents']==250 and s['available_cents']==99750
    assert not ledger.state(path,'b',NOW)['bets']
    with pytest.raises(ValueError): ledger.record(path,'a',candidate(),NOW)
    with pytest.raises(ValueError): ledger.settle(path,'b',s['bets'][0]['id'],'won',NOW+pd.Timedelta(hours=9))
    with pytest.raises(ValueError): ledger.initialise(path,'a',2000,NOW)


def test_daily_cap_includes_settled_bets_and_never_reinvests_same_day_gains(tmp_path):
    path=tmp_path/'paper.sqlite3';ledger.initialise(path,'a',1000,NOW)
    for i in range(8): ledger.record(path,'a',candidate(str(i)),NOW)
    with pytest.raises(ValueError): ledger.record(path,'a',candidate('ninth'),NOW)
    s=ledger.state(path,'a',NOW)
    later=NOW+pd.Timedelta(hours=9)
    ledger.settle(path,'a',s['bets'][0]['id'],'won',later)
    s=ledger.state(path,'a',later)
    assert s['balance_cents']==100245 and s['day_base_cents']==100000
    assert s['day_remaining_cents']==0
    assert ledger.proposed_stake(s)==0
    with pytest.raises(ValueError): ledger.settle(path,'a',s['bets'][0]['id'],'lost',later)
    tomorrow=ledger.state(path,'a',NOW+pd.Timedelta(days=1))
    assert tomorrow['day_base_cents']==100245


def test_void_refunds_without_erasing_exposure_or_future_settlement(tmp_path):
    path=tmp_path/'paper.sqlite3';ledger.initialise(path,'a',1000,NOW)
    ledger.record(path,'a',candidate(),NOW)
    with pytest.raises(ValueError): ledger.settle(path,'a',1,'won',NOW)
    ledger.settle(path,'a',1,'void',NOW+pd.Timedelta(hours=9))
    s=ledger.state(path,'a',NOW+pd.Timedelta(hours=9))
    assert s['reserved_cents']==0 and s['balance_cents']==100000
    assert s['day_used_cents']==250 and s['roi'] is None


def test_expired_or_changed_quote_cannot_be_recorded(tmp_path):
    path=tmp_path/'paper.sqlite3';ledger.initialise(path,'a',1000,NOW)
    with pytest.raises(ValueError): ledger.record(path,'a',candidate(),NOW+pd.Timedelta(minutes=16))
    c=candidate();c['odds']=2.1
    with pytest.raises(ValueError): ledger.record(path,'a',c,NOW)
    assert not ledger.state(path,'a',NOW)['bets']


def test_backup_restore_round_trip_is_isolated_and_never_overwrites(tmp_path):
    path=tmp_path/'paper.sqlite3';ledger.initialise(path,'a',1000,NOW)
    ledger.record(path,'a',candidate(),NOW)
    ledger.settle(path,'a',1,'lost',NOW+pd.Timedelta(hours=9))
    backup=ledger.export_backup(path,'a',NOW+pd.Timedelta(hours=9))
    ledger.restore_backup(path,'b',backup,NOW+pd.Timedelta(hours=9))
    restored=ledger.state(path,'b',NOW+pd.Timedelta(hours=9))
    assert restored['balance_cents']==99750
    assert all(b['owner']=='b' for b in restored['bets'])
    with pytest.raises(ValueError): ledger.restore_backup(path,'a',backup,NOW)
    broken=json.loads(backup);broken['bets'][0]['profit_cents']=999999
    with pytest.raises(ValueError): ledger.restore_backup(path,'c',json.dumps(broken),NOW)
    with pytest.raises(ValueError): ledger.state(path,'c',NOW)


def test_section_is_wired_without_changing_legacy_predictions():
    import ast
    root=Path(__file__).resolve().parents[1]
    source=(root/'unified_app.py').read_text()
    ast.parse(source)
    ast.parse((root/'src/app/tennis_strategy_page.py').read_text())
    assert '"Stratégie ATP"' in source
    assert 'render_tennis_strategy_page(PROJECT_ROOT' in source


def test_concurrent_records_cannot_exceed_daily_budget(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'paper.sqlite3';ledger.initialise(path,'a',1000,NOW)
    def attempt(i):
        try: return ledger.record(path,'a',candidate(str(i)),NOW)
        except ValueError: return 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        amounts=list(pool.map(attempt,range(12)))
    assert sum(amounts)==20
    s=ledger.state(path,'a',NOW)
    assert len(s['bets'])==8 and s['day_used_cents']==2000


def test_packaged_native_model_can_score_without_training_or_history_results(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    if not (root/'models/tennis_strategy/metadata.json').exists():
        pytest.skip('Portable runtime artefacts are not installed')
    meta,_,protocol,model=engine.load_bundle(root)
    meta=dict(meta,history_last_date='2026-09-15',model_year=2026)
    features=engine.fixture_feature_row(pd.DataFrame([historical_row()]),fixture())
    monkeypatch.setattr(engine,'fixture_feature_row',lambda *a,**k:features)
    result=engine.score_fixture((meta,pd.DataFrame(),protocol,model),fixture(),NOW)
    assert sum(result['probabilities'])==pytest.approx(1)
    assert all(0<p<1 for p in result['probabilities'])
    assert result['real_money_authorised'] is False
    assert result['model_sha256']==meta['files']['booster.ubj']
