"""Run with Streamlit installed; all accounts and writes are isolated in tmp_path."""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from src.app import tennis_strategy_page as page
from src.app import tennis_strategy as engine
from src.app import tennis_strategy_ledger as ledger


def test_ui_initialise_select_record_and_show_bankroll(tmp_path, monkeypatch):
    folder=tmp_path/'models/tennis_strategy';folder.mkdir(parents=True)
    (folder/'metadata.json').write_text('{}')
    today=engine.utc().tz_localize(None).normalize()
    history=pd.DataFrame([{'match_date':today-pd.Timedelta(days=1),'tourney_name':'Test',
                          **{f'player_{s}_{k}':v for s,n in [(1,'Alpha A.'),(2,'Beta B.')]
                             for k,v in {'name':n,'id':n,'rank':10,'rank_points':1000,'age':25,'ht':185,'hand':'R'}.items()}}])
    meta={'model_year':engine.utc().year,'history_last_date':str((today-pd.Timedelta(days=1)).date()),
          'files':{'booster.ubj':'model','history.csv.gz':'history'},
          'evidence':{'roi':{'0.02':.149681},'settled':54,'uncertainty':{'ci95':[-.1188,.4746]}}}
    monkeypatch.setattr(page,'_bundle',lambda *_:(meta,history,{},None))
    def score(*args):
        fixture=json.loads(args[-1])
        return {'strategy_id':engine.STRATEGY_ID,'model_sha256':'model','history_sha256':'history',
                'fixture':fixture,'event_key':'ui-event','computed_at':engine.utc().isoformat(),
                'selected_side':0,'pick':fixture['player_1'],'odds':fixture['odds_1'],
                'probability':.6,'probabilities':[.6,.4],'expected_returns':[.188,-.208],
                'eligible':True,'reason':'Simulation de test','real_money_authorised':False}
    monkeypatch.setattr(page,'_score',score)
    app=AppTest.from_string('from pathlib import Path\nfrom src.app.tennis_strategy_page import render_tennis_strategy_page\n'
                           f'render_tennis_strategy_page(Path({str(tmp_path)!r}),7,"test")',default_timeout=15)
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label=='Initialiser la simulation').click().run()
    assert not app.exception
    assert app.metric[0].value=='1000.00 €'
    next(b for b in app.button if b.label=='Calculer selon la stratégie figée').click().run()
    assert any('Confirmer les identités' in e.value for e in app.error)
    assert not any(b.label=='Enregistrer ce pari en simulation' for b in app.button)
    confirmation=next(c for c in app.checkbox if 'simple ATP' in c.label)
    confirmation.check().run()
    next(b for b in app.button if b.label=='Calculer selon la stratégie figée').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='Enregistrer ce pari en simulation').click().run()
    assert not app.exception
    state=ledger.state(tmp_path/'bets/tennis_strategy.sqlite3','7:test')
    assert len(state['bets'])==1 and state['reserved_cents']==250
    assert any(m.value=='997.50 €' for m in app.metric)


def test_stale_ui_keeps_journal_but_never_offers_calculation(tmp_path,monkeypatch):
    folder=tmp_path/'models/tennis_strategy';folder.mkdir(parents=True)
    (folder/'metadata.json').write_text('{}')
    meta={'model_year':engine.utc().year,'history_last_date':'2020-01-01',
          'evidence':{'roi':{'0.02':.1},'settled':54,'uncertainty':{'ci95':[-.1,.4]}}}
    monkeypatch.setattr(page,'_bundle',lambda *_:(meta,pd.DataFrame(),{},None))
    ledger.initialise(tmp_path/'bets/tennis_strategy.sqlite3','7:test',1000)
    app=AppTest.from_string('from pathlib import Path\nfrom src.app.tennis_strategy_page import render_tennis_strategy_page\n'
                           f'render_tennis_strategy_page(Path({str(tmp_path)!r}),7,"test")',default_timeout=15).run()
    assert not app.exception
    assert any('trop ancien' in e.value for e in app.error)
    assert not any('Calculer selon' in b.label for b in app.button)
    assert app.metric[0].value=='1000.00 €'
    calls = []
    def fail_refresh(root, key):
        calls.append(key)
        return {'ok': False, 'output': 'HTTP 503 fournisseur'}
    monkeypatch.setattr(page, 'run_task', fail_refresh)
    next(b for b in app.button if b.label=='Actualiser les données de la stratégie ATP').click().run()
    assert calls == ['tennis_strategy_refresh']
    assert not app.exception
    assert any('Actualisation non validée' in e.value for e in app.error)
    assert app.metric[0].value=='1000.00 €'


def test_refresh_success_reloads_metadata_without_touching_bankroll(tmp_path, monkeypatch):
    folder=tmp_path/'models/tennis_strategy';folder.mkdir(parents=True)
    (folder/'metadata.json').write_text('{}')
    meta={'model_year':engine.utc().year,'history_last_date':'2020-01-01',
          'evidence':{'roi':{'0.02':.1},'settled':54,'uncertainty':{'ci95':[-.1,.4]}}}
    def bundle(*args):
        return meta, pd.DataFrame(), {}, None
    cleared=[]
    bundle.clear=lambda: cleared.append(True)
    monkeypatch.setattr(page, '_bundle', bundle)
    monkeypatch.setattr(engine, 'latest_profiles', lambda *a: pd.DataFrame())
    def refresh(root, key):
        meta['history_last_date']=str(engine.utc().tz_convert('Europe/Paris').date())
        return {'ok':True, 'output':'OK'}
    monkeypatch.setattr(page,'run_task',refresh)
    ledger.initialise(tmp_path/'bets/tennis_strategy.sqlite3','7:test',1000)
    app=AppTest.from_string('from pathlib import Path\nfrom src.app.tennis_strategy_page import render_tennis_strategy_page\n'
                           f'render_tennis_strategy_page(Path({str(tmp_path)!r}),7,"test")',default_timeout=15).run()
    next(b for b in app.button if b.label=='Actualiser les données de la stratégie ATP').click().run()
    assert not app.exception
    assert cleared == [True]
    assert not any('trop ancien' in e.value for e in app.error)
    assert app.metric[0].value=='1000.00 €'
