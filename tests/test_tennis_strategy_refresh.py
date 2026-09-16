import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.app import tennis_strategy_refresh as refresh_module
from src.data.tennis_pipeline import DataQualityError

NOW = pd.Timestamp('2026-09-16T10:00:00Z')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    folder = tmp_path / 'models/tennis_strategy'
    folder.mkdir(parents=True)
    old = pd.DataFrame([
        dict(match_id='old', match_date='2025-12-01', player_1_id='a', player_2_id='b', match_status='completed'),
        dict(match_id='atp_td:1', match_date='2026-08-29', player_1_id='a', player_2_id='b', match_status='completed'),
    ])
    current = pd.DataFrame([
        dict(match_id='atp_td:0', match_date='2026-08-29', player_1_id='a', player_2_id='b', match_status='completed'),
        dict(match_id='atp_td:1', match_date='2026-09-13', player_1_id='a', player_2_id='c', match_status='source_conflict', minutes=90),
    ])
    old.to_csv(folder / 'history.csv.gz', index=False)
    (folder / 'booster.ubj').write_bytes(b'frozen-model')
    meta = {'model_year': 2026, 'history_last_date': '2026-08-29', 'files': {'booster.ubj': 'unchanged'}}
    (folder / 'metadata.json').write_text(json.dumps(meta))
    monkeypatch.setattr(refresh_module, 'load_bundle', lambda *_: (meta, old, {}, None))
    source = SimpleNamespace(name='Test source', url='https://example.org', updated_at=NOW.isoformat())
    legacy = pd.DataFrame({'Date': current.match_date, 'Player_1': ['A', 'A'], 'Player_2': ['B', 'C']})
    monkeypatch.setattr(refresh_module, 'fetch_odds_snapshot', lambda *a: (legacy, source))
    monkeypatch.setattr(refresh_module, 'transform_tennis_data_raw', lambda frame: frame)
    monkeypatch.setattr(refresh_module, 'normalize_legacy_odds', lambda frame, today: frame)
    monkeypatch.setattr(refresh_module, 'fetch_tennis_mylife', lambda *a: (current, None, source, {}))
    monkeypatch.setattr(refresh_module, 'normalize_rich_matches', lambda frame, today: frame)
    monkeypatch.setattr(refresh_module, 'attach_odds', lambda frame, legacy: frame)
    monkeypatch.setattr(refresh_module, 'add_stable_player_orientation', lambda frame: frame)
    monkeypatch.setattr(refresh_module, 'atp_master', lambda *a, **kw: (current.copy(), {'unique_rich_matches': len(current)}))
    return tmp_path, folder, meta, current


def snapshot(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_refresh_preserves_model_and_old_years(prepared, monkeypatch):
    root, folder, _, _ = prepared
    updated = refresh_module.refresh(root, NOW, progress=lambda _: None)
    assert updated['history_last_date'] == '2026-09-13'
    assert updated['history_rows'] == 3
    assert (folder / 'booster.ubj').read_bytes() == b'frozen-model'
    history = pd.read_csv(folder / 'history.csv.gz')
    assert history.iloc[0].match_id == 'old'
    assert history.iloc[1].match_id == 'atp_td:1'
    assert pd.isna(history.iloc[-1]['minutes'])
    assert refresh_module.digest(folder / 'history.csv.gz') == updated['files']['history.csv.gz']
    # A second refresh is idempotent in population and generated identifiers.
    assert history.match_id.is_unique
    monkeypatch.setattr(refresh_module, 'load_bundle', lambda *_: (updated, history, {}, None))
    refresh_module.refresh(root, NOW, progress=lambda _: None)
    pd.testing.assert_frame_equal(history, pd.read_csv(folder / 'history.csv.gz'))


def test_supplier_failure_changes_no_file(prepared, monkeypatch):
    root, folder, _, _ = prepared
    before = snapshot(folder)
    def unavailable(*args):
        raise DataQualityError('HTTP 503 fournisseur')
    monkeypatch.setattr(refresh_module, 'fetch_odds_snapshot', unavailable)
    with pytest.raises(DataQualityError, match='503'):
        refresh_module.refresh(root, NOW)
    assert snapshot(folder) == before


def test_missing_statistics_blocks_publication(prepared, monkeypatch):
    root, folder, _, current = prepared
    before = snapshot(folder)
    monkeypatch.setattr(refresh_module, 'atp_master', lambda *a, **kw: (current.copy(), {'unique_rich_matches': 0}))
    with pytest.raises(DataQualityError, match='90 %'):
        refresh_module.refresh(root, NOW)
    assert snapshot(folder) == before


@pytest.mark.parametrize('failure', ['stale', 'removed', 'changed_date', 'year'])
def test_invalid_refresh_changes_no_file(prepared, failure):
    root, folder, meta, current = prepared
    if failure == 'stale':
        current.loc[1, 'match_date'] = '2026-09-01'
    elif failure == 'removed':
        current.drop(index=[0, 1], inplace=True)
    elif failure == 'changed_date':
        current.loc[0, 'match_date'] = '2026-08-28'
    else:
        meta['model_year'] = 2025
    before = snapshot(folder)
    with pytest.raises((DataQualityError, ValueError)):
        refresh_module.refresh(root, NOW)
    assert snapshot(folder) == before


def test_publication_failure_restores_previous_generation(prepared, monkeypatch):
    root, folder, _, _ = prepared
    before = snapshot(folder)
    original = refresh_module.os.replace
    def fail_metadata(source, target):
        if source.name == 'metadata.json' and target == folder / 'metadata.json':
            raise OSError('Test disk error')
        return original(source, target)
    monkeypatch.setattr(refresh_module.os, 'replace', fail_metadata)
    with pytest.raises(OSError, match='disk error'):
        refresh_module.refresh(root, NOW)
    assert snapshot(folder) == before
