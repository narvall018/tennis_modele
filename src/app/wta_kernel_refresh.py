"""Refresh public WTA stats without retraining or changing research/ledgers."""
from datetime import date
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from src.app.wta_kernel_strategy import (load_bundle, apply_identities, identity_map,
                                         prepare_history, FOLDER, digest)
from src.data.tennis_expansion import fetch_wta_matches


def refresh(root, today=None):
    today = today or date.today(); root = Path(root)
    folder = root/FOLDER; before = digest(folder/'metadata.json')
    meta, old, _ = load_bundle(root)
    old = old.drop(columns=['_winner_key', '_loser_key'], errors='ignore')
    if today.year != meta['model_year']: raise ValueError('Nouvelle année : reconstruire le modèle après audit.')
    raw, _, details = fetch_wta_matches(today.year, today.year)
    if f'{today.year}_wta.csv' in details['missing_files']: raise ValueError('Saison WTA annuelle absente.')
    # Annual and ongoing files may overlap. Keep the annual copy first; refuse
    # conflicting identities rather than resolving with known winners/returns.
    raw = raw.drop_duplicates()
    key = ['tourney_id', 'match_num']
    overlaps = raw[raw.duplicated(key, keep=False)]
    for _, group in overlaps.groupby(key):
        if group[['winner_id', 'loser_id', 'winner_name', 'loser_name']].drop_duplicates().shape[0] != 1:
            raise ValueError('Conflit d’identité entre sources WTA.')
    raw = raw.drop_duplicates(key, keep='first')
    # The annual file alone cannot outvote a canonical id it does not contain, so seed
    # the table with the one frozen at build time and carry any new merge forward.
    merges = identity_map(raw, meta.get('identity_merges'))
    new = prepare_history(raw, today, merges)
    new = new[new._start.dt.year.eq(today.year)]
    old = apply_identities(old, merges)  # A merge found this year also repairs earlier seasons.
    previous = old[old._start.dt.year.eq(today.year)]
    old_keys = set(zip(previous.tourney_id, previous.match_num))
    if not old_keys.issubset(set(zip(new.tourney_id, new.match_num))):
        raise ValueError('Historique WTA téléchargé incomplet : ancien paquet conservé.')
    valid_old = previous[previous._valid]; valid_new = new[new._valid]
    if not set(zip(valid_old.tourney_id, valid_old.match_num)).issubset(set(zip(valid_new.tourney_id, valid_new.match_num))):
        raise ValueError('Statistiques valides manquantes : ancien paquet conservé.')
    history = pd.concat([old[old._start.dt.year.lt(today.year)], new], ignore_index=True)
    latest = str(history.loc[history._valid, '_start'].max().date())
    if latest < meta['history_last_date']: raise ValueError('Régression de fraîcheur WTA.')
    with tempfile.TemporaryDirectory(prefix='wta-kernel-', dir=folder) as temp:
        staged = Path(temp)/'history.csv.gz'
        history.to_csv(staged, index=False, compression='gzip')
        meta.update(history_rows=len(history), history_last_date=latest,
                    identity_merges=merges, refreshed_at=pd.Timestamp.now(tz='UTC').isoformat())
        meta['files']['history.csv.gz'] = digest(staged)
        manifest = Path(temp)/'metadata.json'
        manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        if digest(folder/'metadata.json') != before: raise ValueError('Actualisation concurrente : réessayer.')
        os.replace(staged, folder/'history.csv.gz')
        os.replace(manifest, folder/'metadata.json')
    return latest
