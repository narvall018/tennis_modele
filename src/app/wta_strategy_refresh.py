"""Refresh WTA live history only; keep the annual booster and research immutable."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from src.app.wta_strategy import HISTORY_COLUMNS, digest, freshness_reasons, load_bundle, utc
from src.data.tennis_pipeline import (DataQualityError, _atomic_csv, _atomic_json,
    fetch_odds_snapshot, normalize_legacy_odds, transform_tennis_data_raw)


def merge_current(previous, legacy, today):
    previous = previous.copy()
    previous['_date'] = pd.to_datetime(previous['_date'])
    dates = pd.to_datetime(legacy['Date'])
    legacy = legacy[dates.dt.date < today].copy()
    if legacy.empty or not pd.to_datetime(legacy['Date']).dt.year.eq(today.year).all():
        raise DataQualityError('Saison WTA absente ou incorrecte.')
    mapping = {'Date': '_date', 'Player_1': '_p1', 'Player_2': '_p2', 'Status': '_status',
               'Surface': '_surface', 'Series': '_series', 'Round': '_round', 'Tournament': '_tournament'}
    current = legacy[list(mapping)].rename(columns=mapping)
    current['_date'] = pd.to_datetime(current['_date'])
    current['_label'] = legacy['Winner'].eq(legacy['Player_1']).astype(int)
    if not current['_status'].isin(['completed', 'retired', 'walkover', 'defaulted', 'unfinished', 'unknown']).all():
        raise DataQualityError('Statut WTA inconnu : audit requis.')
    if not current['_surface'].isin(['Hard', 'Clay', 'Grass', 'Carpet']).all():
        raise DataQualityError('Surface WTA inconnue.')
    def keys(frame):
        return [(str(pd.Timestamp(d).date()), *sorted([a, b]))
                for d, a, b in frame[['_date', '_p1', '_p2']].itertuples(index=False, name=None)]
    old = previous[pd.to_datetime(previous['_date']).dt.year.eq(today.year)]
    old_ids = dict(zip(keys(old), old['_source_row_id']))
    new_keys = keys(current)
    if len(set(new_keys)) != len(new_keys) or not set(old_ids).issubset(new_keys):
        raise DataQualityError('Doublons ou matchs historiques WTA manquants ; ancien paquet conservé.')
    if current['_date'].max() < pd.to_datetime(previous['_date']).max():
        raise DataQualityError('Régression de date WTA.')
    current['_source_row_id'] = [old_ids.get(k, 'wta_td_live:' + ':'.join(k)) for k in new_keys]
    historical = previous[pd.to_datetime(previous['_date']).dt.year.lt(today.year)]
    result = pd.concat([historical[HISTORY_COLUMNS], current[HISTORY_COLUMNS]], ignore_index=True)
    if result['_source_row_id'].duplicated().any():
        raise DataQualityError('Identifiants WTA dupliqués.')
    return result


def refresh(root, now=None, progress=print):
    stamp = utc(now)
    today = stamp.tz_convert('Europe/Paris').date()
    meta, previous, _, _ = load_bundle(root)
    if meta['model_year'] != today.year:
        raise ValueError('Le changement d’année exige un nouvel audit du modèle WTA.')
    progress(f'Téléchargement Tennis-Data WTA {today.year}…')
    raw, source = fetch_odds_snapshot(today.year, today.year, tour='wta')
    legacy = normalize_legacy_odds(transform_tennis_data_raw(raw), today)
    history = merge_current(previous, legacy, today)
    latest = str(history['_date'].max().date())
    updated = {**meta, 'built_at': stamp.isoformat(), 'history_last_date': latest,
               'history_rows': len(history), 'live_refresh': {'retrieved_at': stamp.isoformat(),
               'source': vars(source),
               'workbook_urls': sorted(raw['_source_url'].dropna().unique().tolist()) if '_source_url' in raw else [],
               'policy': 'Prior-day history only; annual model unchanged'}}
    reasons = freshness_reasons(updated, stamp)
    if reasons:
        raise DataQualityError(' ; '.join(reasons))
    folder = Path(root) / 'models/wta_live_strategy'
    if json.loads((folder / 'metadata.json').read_text()) != meta:
        raise DataQualityError('Une autre actualisation a modifié le paquet WTA ; réessayer.')
    with tempfile.TemporaryDirectory(prefix='.refresh-', dir=folder) as temporary:
        staging = Path(temporary)
        _atomic_csv(history, staging / 'history.csv.gz', gzip=True)
        updated['files'] = {**meta['files'], 'history.csv.gz': digest(staging / 'history.csv.gz')}
        _atomic_json(updated, staging / 'metadata.json')
        for name in ['history.csv.gz', 'metadata.json']:
            shutil.copyfile(folder / name, staging / (name + '.previous'))
        try:
            os.replace(staging / 'history.csv.gz', folder / 'history.csv.gz')
            os.replace(staging / 'metadata.json', folder / 'metadata.json')
        except OSError:
            for name in ['history.csv.gz', 'metadata.json']:
                os.replace(staging / (name + '.previous'), folder / name)
            raise
    progress(f'Paquet WTA actualisé : {len(history)} matchs, jusqu’au {latest}. Modèle et bankroll inchangés.')
    return updated
