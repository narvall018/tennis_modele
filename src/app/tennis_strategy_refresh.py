"""Refresh only the live ATP year; never retrain or rewrite research archives."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.app.tennis_strategy import digest, freshness_reasons, load_bundle, utc
from src.backtesting.three_sport_residual import atp_master
from src.data.tennis_pipeline import (
    DataQualityError, _atomic_csv, _atomic_json, add_stable_player_orientation,
    attach_odds, fetch_odds_snapshot, fetch_tennis_mylife, normalize_legacy_odds,
    normalize_rich_matches, transform_tennis_data_raw,
)


def refresh(root: Path, now=None, progress=print):
    """Validate all inputs before publication; a supplier failure leaves files alone."""
    stamp = utc(now)
    today = stamp.tz_convert('Europe/Paris').date()
    meta, previous, _, _ = load_bundle(root)
    if meta['model_year'] != today.year:
        raise ValueError('Le modèle annuel doit être audité avant le changement d’année.')
    progress(f'Téléchargement Tennis-Data ATP {today.year}…')
    raw_odds, odds_source = fetch_odds_snapshot(today.year, today.year)
    legacy = normalize_legacy_odds(transform_tennis_data_raw(raw_odds), today)
    legacy = legacy[pd.to_datetime(legacy['Date']).dt.date < today].copy()
    if legacy.empty or not pd.to_datetime(legacy['Date']).dt.year.eq(today.year).all():
        raise DataQualityError('La source ne contient pas la saison ATP attendue.')
    if legacy.duplicated(['Date', 'Player_1', 'Player_2']).any():
        raise DataQualityError('Matchs ATP dupliqués dans la source courante.')
    progress(f'Téléchargement des statistiques TennisMyLife {today.year}…')
    raw_rich, _, stats_source, source_details = fetch_tennis_mylife(today.year, today.year)
    rich = normalize_rich_matches(raw_rich, today)
    rich = rich[pd.to_datetime(rich['match_date']).dt.date < today].copy()
    if rich.empty or (today - pd.to_datetime(rich['match_date']).max().date()).days > 7:
        raise DataQualityError('Statistiques ATP manquantes ou trop anciennes ; ancien paquet conservé.')
    enriched = add_stable_player_orientation(attach_odds(rich, legacy))
    current, audit = atp_master(legacy, enriched, quarantine_before=str(today))
    if audit['unique_rich_matches'] / max(len(current), 1) < .90:
        raise DataQualityError('Moins de 90 % des matchs reliés aux statistiques ; publication refusée.')
    current.loc[current['match_status'].eq('source_conflict'), 'minutes'] = float('nan')
    old_current = previous[pd.to_datetime(previous['match_date']).dt.year.eq(today.year)]
    if len(current) < len(old_current):
        raise DataQualityError('La source a perdu des matchs de la saison ; publication refusée.')
    if pd.to_datetime(current['match_date']).max() < pd.to_datetime(previous['match_date']).max():
        raise DataQualityError('Régression de la date des données ; publication refusée.')
    # Do not silently remove one old match while adding many new matches.
    def match_key(row):
        return (str(pd.Timestamp(row.match_date).date()), *sorted([str(row.player_1_id), str(row.player_2_id)]))
    previous_ids = {match_key(row): row.match_id for row in old_current.itertuples(index=False)}
    if not set(previous_ids).issubset({match_key(row) for row in current.itertuples(index=False)}):
        raise DataQualityError('Des identités ou dates historiques ont changé ; audit manuel nécessaire.')
    historical = previous[pd.to_datetime(previous['match_date']).dt.year.lt(today.year)]
    # Keep historical tie-breaking stable when replaying multiple matches on one day.
    current['match_id'] = [previous_ids.get(match_key(row), 'atp_td_live:' + ':'.join(match_key(row)))
                           for row in current.itertuples(index=False)]
    history = pd.concat([historical, current], ignore_index=True)
    latest = str(pd.to_datetime(history['match_date']).max().date())
    updated = {**meta, 'built_at': stamp.isoformat(), 'history_last_date': latest,
               'history_rows': len(history),
               'live_refresh': {'year': today.year, 'retrieved_at': stamp.isoformat(),
                   'odds_source': vars(odds_source), 'stats_source': vars(stats_source),
                   'source_details': source_details, 'current_year_audit': audit,
                   'historical_rows_preserved': len(historical),
                   'policy': 'Tennis-Data match dates; original model unchanged; source conflicts quarantined'}}
    reasons = freshness_reasons(updated, stamp)
    if reasons:
        raise DataQualityError(' ; '.join(reasons))
    folder = Path(root) / 'models/tennis_strategy'
    # Reject a concurrent replacement instead of publishing against another model.
    if json.loads((folder / 'metadata.json').read_text()) != meta:
        raise DataQualityError('Une autre mise à jour a modifié le paquet ; réessayer.')
    progress(f'Contrôles réussis : publication de {len(history)} matchs, jusqu’au {latest}.')
    with tempfile.TemporaryDirectory(prefix='.refresh-', dir=folder) as staging_text:
        staging = Path(staging_text)
        _atomic_csv(history, staging / 'history.csv.gz', gzip=True)
        updated['files'] = {**meta['files'], 'history.csv.gz': digest(staging / 'history.csv.gz')}
        _atomic_json(updated, staging / 'metadata.json')
        # Roll back a failed second replacement. Readers reject any brief hash
        # mismatch; no calculation is allowed with an inconsistent generation.
        for name in ['history.csv.gz', 'metadata.json']:
            shutil.copyfile(folder / name, staging / (name + '.previous'))
        try:
            os.replace(staging / 'history.csv.gz', folder / 'history.csv.gz')
            os.replace(staging / 'metadata.json', folder / 'metadata.json')
        except OSError:
            for name in ['history.csv.gz', 'metadata.json']:
                os.replace(staging / (name + '.previous'), folder / name)
            raise
    return updated
