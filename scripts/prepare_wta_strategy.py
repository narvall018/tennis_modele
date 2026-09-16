"""One-time export of the registered recent-tree WTA candidate, not a new search."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.app.wta_strategy import HISTORY_COLUMNS, STRATEGY_ID, digest, model_inputs, utc
from src.backtesting.nonlinear_market import OffsetTrees, recency_weights
from src.data.tennis_pipeline import _atomic_csv, _atomic_json


def prepare(root, now=None):
    stamp = utc(now)
    year = stamp.tz_convert('Europe/Paris').year
    if year != 2026:
        raise ValueError('Export audité pour 2026 seulement ; nouvel audit annuel requis.')
    folder = root / 'models/wta_live_strategy'
    if folder.exists():
        raise FileExistsError('Paquet existant conservé ; utiliser refresh_wta_strategy.py.')
    study = root / 'models/consensus_research_2026_09_16'
    research = json.loads((study / 'protocol.json').read_text())
    registration = json.loads((study / 'registration.json').read_text())
    checks = {**research['sources'], **registration['implementations'],
              'models/consensus_research_2026_09_16/protocol.json': registration['protocol_sha256']}
    for name, expected in checks.items():
        if digest(root / name) != expected:
            raise ValueError(f'Source de recherche modifiée : {name}')
    wta = json.loads((root / 'models/wta_price_residual/protocol.json').read_text())
    protocol = {'wta_features': wta, 'research': research}
    frame = pd.read_parquet(study / 'wta/features.parquet')
    frame['_date'] = pd.to_datetime(frame['_date'])
    if frame['_date'].dt.year.gt(2025).any():
        raise ValueError('Données 2026 interdites pour entraîner le modèle annuel.')
    cutoff = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=research['embargo_days'])
    train = frame['_date'].lt(cutoff) & frame['_status'].eq('completed')
    if train.sum() < research['sports']['wta']['min_train']:
        raise ValueError('Historique entraînement insuffisant.')
    x, q, names, signs = model_inputs(frame.loc[train], protocol)
    model = OffsetTrees(research['booster'], research['boost_rounds'], signs)
    model.fit(x, q, frame.loc[train, '_label'].to_numpy(int),
              recency_weights(frame.loc[train, '_date'], cutoff, research['half_life_years']))
    history = pd.read_csv(root / wta['data_path'], usecols=HISTORY_COLUMNS)
    history['_date'] = pd.to_datetime(history['_date'])
    history = history[history['_date'] < stamp.tz_convert('Europe/Paris').tz_localize(None).normalize()]
    report = json.loads((study / 'wta/report.json').read_text())
    folder.mkdir()
    model.model.save_model(folder / 'booster.ubj')
    _atomic_csv(history[HISTORY_COLUMNS], folder / 'history.csv.gz', gzip=True)
    _atomic_json(protocol, folder / 'protocol.json')
    meta = {'strategy_id': STRATEGY_ID, 'candidate': 'trees_recent', 'model_year': year,
        'built_at': stamp.isoformat(), 'history_rows': len(history),
        'history_last_date': str(history['_date'].max().date()), 'features': names,
        'swap_signs': signs.tolist(), 'train_rows': int(train.sum()),
        'train_last_date': str(frame.loc[train, '_date'].max().date()), 'cutoff_exclusive': str(cutoff.date()),
        'training_features_sha256': digest(study / 'wta/features.parquet'),
        'registration_sha256': digest(study / 'registration.json'),
        'evidence': report['evaluation']['trees_recent'], 'tuning': report['tuning']['trees_recent'],
        'real_money_authorised': False,
        'files': {name: digest(folder / name) for name in ['booster.ubj', 'history.csv.gz', 'protocol.json']}}
    _atomic_json(meta, folder / 'metadata.json')
    return meta


if __name__ == '__main__':
    print(json.dumps(prepare(ROOT), ensure_ascii=False, indent=2))
