"""Export the 2026 frozen-family ATP model and local history, never a new ROI test."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app.tennis_strategy import STRATEGY_ID, digest
from src.backtesting.nonlinear_market import OffsetTrees, inputs, recency_weights, repair_atp_context
from src.backtesting.three_sport_residual import atp_master


def prepare(root=ROOT):
    folder = root / 'models/tennis_strategy'
    folder.mkdir(exist_ok=True)
    research = root / 'models/nonlinear_market'
    now = pd.Timestamp.now(tz='UTC')
    if now.year != 2026:
        raise ValueError('Reconstruction annuelle à auditer : ce paquet est prévu pour 2026 uniquement.')
    existing = folder / 'metadata.json'
    if existing.exists():
        metadata = json.loads(existing.read_text())
        if metadata['strategy_id'] != STRATEGY_ID or metadata['model_year'] != 2026:
            raise ValueError('Paquet annuel incompatible')
        for name in ['booster.ubj', 'protocol.json']:
            if digest(folder / name) != metadata['files'][name]:
                raise ValueError('Modèle existant incohérent ; ne pas le remplacer silencieusement')
        print('Modèle annuel conservé ; actualisation de l’historique seulement.', flush=True)
    else:
        metadata = train_initial(root, folder, research)
    legacy = pd.read_csv(root / 'data/atp_tennis.csv', low_memory=False)
    rich = pd.read_csv(root / 'data/processed/atp_matches_enriched.csv.gz', low_memory=False)
    today = now.tz_localize(None).normalize()
    legacy = legacy[pd.to_datetime(legacy['Date']) < today]
    rich = rich[pd.to_datetime(rich['match_date']) < today]
    # Prospective-only history: quarantine EVERY unresolved past source conflict,
    # without choosing a winner or examining returns. Frozen training is unchanged.
    history, audit = atp_master(legacy, rich, quarantine_before=str(today.date()))
    history.loc[history['match_status'].eq('source_conflict'), 'minutes'] = float('nan')
    audit['prospective_conflict_policy'] = 'Unresolved past source conflicts do not update results, serve stats or workload; no winner selected'
    # Exact master matching and source-conflict checks precede publication.
    history.to_csv(folder / 'history.csv.gz', index=False, compression='gzip')
    metadata.update(built_at=datetime.now(timezone.utc).isoformat(),
                    history_last_date=str(pd.to_datetime(history['match_date']).max().date()),
                    history_rows=len(history), history_audit=audit,
                    files={name: digest(folder / name) for name in ['booster.ubj', 'history.csv.gz', 'protocol.json']})
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + '\n')
    print(f"Paquet prêt ; historique arrêté au {metadata['history_last_date']}. Fraîcheur contrôlée dans l'app.")


def train_initial(root, folder, research):
    protocol = json.loads((research / 'protocol.json').read_text())
    chosen = json.loads((research / 'atp/report.json').read_text())
    if chosen['selection']['chosen']['model'] != 'trees_recent' or chosen['selection']['chosen']['threshold'] != .02:
        raise ValueError('The deployed family must not drift silently from the declared ATP candidate')
    source = root / 'models/three_sport_residual/atp_features.parquet'
    if digest(source) != protocol['frozen_files'][str(source.relative_to(root))]:
        raise ValueError('Frozen training features changed')
    frame = pd.read_parquet(source)
    master_path = root / 'models/three_sport_residual/atp_matched_master.csv.gz'
    if digest(master_path) != protocol['frozen_files'][str(master_path.relative_to(root))]:
        raise ValueError('Frozen context metadata changed')
    frame, _ = repair_atp_context(frame, pd.read_csv(master_path, usecols=['match_id', 'indoor', 'round']))
    cutoff = pd.Timestamp('2026-01-01') - pd.Timedelta(days=7)
    train = frame['_date'].lt(cutoff) & frame['_status'].eq('completed')
    x, q, _, names, signs = inputs(frame.loc[train], 'atp', protocol)
    weights = recency_weights(frame.loc[train, '_date'], cutoff, 3)
    print(f'Entraînement ATP 2026 : {train.sum()} matchs, famille et paramètres figés.', flush=True)
    model = OffsetTrees(protocol['booster'], protocol['boost_rounds'], signs)
    model.fit(x, q, frame.loc[train, '_label'].to_numpy(int), weights)
    model.model.save_model(folder / 'booster.ubj')
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return {'strategy_id': STRATEGY_ID, 'candidate': 'trees_recent', 'model_year': 2026,
                'built_at': datetime.now(timezone.utc).isoformat(), 'train_cutoff_exclusive': str(cutoff.date()),
                'train_max': str(frame.loc[train, '_date'].max().date()), 'train_rows': int(train.sum()),
                'features': names, 'swap_signs': signs.tolist(),
                'evidence': chosen['evaluation']['flat'], 'tuning_admitted': False,
                'real_money_authorised': False}


if __name__ == '__main__':
    prepare()
