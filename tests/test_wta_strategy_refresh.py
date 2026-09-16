import json
from types import SimpleNamespace
import pandas as pd
import pytest

from src.app import wta_strategy_refresh as module
from src.data.tennis_pipeline import DataQualityError

NOW = pd.Timestamp('2026-09-16T10:00:00Z')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    folder = tmp_path/'models/wta_live_strategy'; folder.mkdir(parents=True)
    old = pd.DataFrame([{'_source_row_id': str(i), '_date': pd.Timestamp(day), '_p1': 'Alpha A.', '_p2': 'Beta B.',
        '_status': 'completed', '_label': 1, '_surface': 'Hard', '_series': 'International',
        '_round': '1st Round', '_tournament': 'Test'} for i, day in enumerate(['2025-12-01', '2026-08-29'])])
    current = pd.DataFrame([{'Date': day, 'Player_1': 'Alpha A.', 'Player_2': 'Beta B.', 'Winner': 'Alpha A.',
        'Status': 'completed', 'Surface': 'Hard', 'Series': 'International', 'Round': '1st Round', 'Tournament': 'Test'}
        for day in ['2026-08-29', '2026-09-13']])
    old.to_csv(folder/'history.csv.gz', index=False)
    (folder/'booster.ubj').write_bytes(b'frozen')
    meta = {'model_year': 2026, 'history_last_date': '2026-08-29', 'files': {'booster.ubj': 'frozen'}}
    (folder/'metadata.json').write_text(json.dumps(meta))
    monkeypatch.setattr(module, 'load_bundle', lambda *_: (meta, old, {}, None))
    def fetch(start, end, tour):
        assert start == end == 2026 and tour == 'wta'
        return current.copy(), SimpleNamespace(url='https://example.org', name='test')
    monkeypatch.setattr(module, 'fetch_odds_snapshot', fetch)
    monkeypatch.setattr(module, 'transform_tennis_data_raw', lambda frame: frame)
    monkeypatch.setattr(module, 'normalize_legacy_odds', lambda frame, today: frame)
    return tmp_path, folder, meta, current


def snapshot(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_wta_refresh_keeps_booster_and_historical_years(prepared, monkeypatch):
    root, folder, _, _ = prepared
    updated = module.refresh(root, NOW, progress=lambda _: None)
    result = pd.read_csv(folder/'history.csv.gz')
    assert len(result) == 3 and result.iloc[0]['_date'] == '2025-12-01'
    assert updated['history_last_date'] == '2026-09-13'
    assert (folder/'booster.ubj').read_bytes() == b'frozen'
    assert module.digest(folder/'history.csv.gz') == updated['files']['history.csv.gz']
    monkeypatch.setattr(module, 'load_bundle', lambda *_: (updated, result, {}, None))
    module.refresh(root, NOW, progress=lambda _: None)
    pd.testing.assert_frame_equal(result, pd.read_csv(folder/'history.csv.gz'))


@pytest.mark.parametrize('failure', ['supplier', 'stale', 'missing', 'year', 'disk'])
def test_failed_wta_refresh_keeps_previous_generation(prepared, monkeypatch, failure):
    root, folder, meta, current = prepared
    before = snapshot(folder)
    if failure == 'supplier':
        def fail(*args, **kwargs): raise DataQualityError('HTTP 503')
        monkeypatch.setattr(module, 'fetch_odds_snapshot', fail)
    if failure == 'stale': current.loc[1, 'Date'] = '2026-09-01'
    if failure == 'missing': current.drop(index=0, inplace=True)
    if failure == 'year': meta['model_year'] = 2025
    if failure == 'disk':
        replace = module.os.replace
        def fail(source, target):
            if source.name == 'metadata.json' and target == folder/'metadata.json': raise OSError('disk error')
            return replace(source, target)
        monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises((DataQualityError, ValueError, OSError)):
        module.refresh(root, NOW, progress=lambda _: None)
    assert snapshot(folder) == before
