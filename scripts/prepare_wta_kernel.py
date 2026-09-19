"""Export the frozen WTA kernel family for prospective paper use, no2026 fit."""
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtesting.kernel_matchups import KernelResidual, arrays, load_frame, STUDY
from src.app.wta_kernel_strategy import identity_map, prepare_history, digest, STRATEGY_ID, FOLDER


def main():
    research = ROOT/'models'/STUDY
    protocol = json.loads((research/'protocol.json').read_text())
    registration = json.loads((research/'registration.json').read_text())
    for name, expected in {**protocol['sources'], **registration['implementations'],
            str((research/'protocol.json').relative_to(ROOT)): registration['protocol_sha256']}.items():
        if digest(ROOT/name) != expected: raise ValueError(f'Frozen research input changed: {name}')
    year = date.today().year
    if year != 2026: raise ValueError('A new year requires an explicit training-history audit.')
    cutoff = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=7)
    frame = load_frame(ROOT, 'wta', protocol)
    x, q, features, signs = arrays(frame, 'wta')
    train = frame._date.lt(cutoff) & frame._date.ge(cutoff-pd.DateOffset(years=6)) & frame._status.eq('completed')
    if train.sum() < 3000: raise ValueError('Insufficient training history')
    age = (cutoff-frame.loc[train, '_date']).dt.total_seconds().to_numpy()/86400
    model = KernelResidual(.001, 128, 20260919, signs).fit(
        x[train], q[train], frame.loc[train, '_label'].to_numpy(int), np.exp2(-age/(365.25*3)))
    raw = pd.read_csv(ROOT/'data/raw/tennis_mylife/wta_matches_2000_current.csv.gz', low_memory=False)
    identities, merges = identity_map(raw)
    history = prepare_history(raw, date.today(), identities)
    folder = ROOT/FOLDER
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'model.npz', mean=model.mean, scale=model.scale, signs=model.signs,
                        projection=model.projection, phase=model.phase, coefficients=model.coefficients)
    history.to_csv(folder/'history.csv.gz', index=False, compression='gzip')
    evidence = json.loads((research/'report.json').read_text())['evaluation']['wta']['kernel_flexible']
    meta = {'strategy_id': STRATEGY_ID, 'candidate': 'kernel_flexible', 'model_year': year,
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(), 'features': features,
        'training_rows': int(train.sum()), 'training_max_date': str(frame.loc[train, '_date'].max().date()),
        'training_cutoff_exclusive': str(cutoff.date()), 'research_protocol_sha256': digest(research/'protocol.json'),
        'history_rows': len(history), 'history_last_date': str(history.loc[history._valid, '_start'].max().date()),
        'history_source_sha256': digest(ROOT/'data/raw/tennis_mylife/wta_matches_2000_current.csv.gz'),
        'identities': {str(int(k)): int(v) for k, v in sorted(identities.items())}, 'identity_merges': merges,
        'files': {p: digest(folder/p) for p in ['model.npz', 'history.csv.gz']},
        'evidence': evidence, 'prospective_selection': 'USER_REQUEST_AFTER_EXPLORATORY_DIAGNOSTIC_NOT_TUNING_ADMITTED',
        'real_money_authorised': False}
    (folder/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    # Portable NumPy inference must match the trained research model exactly.
    from src.app.wta_kernel_strategy import load_bundle, probabilities
    _, _, saved = load_bundle(ROOT)
    np.testing.assert_allclose(probabilities(saved, x[:100], q[:100]), model.predict(x[:100], q[:100]), atol=1e-12)
    print(f'WTA kernel {year}: {train.sum()} training rows, statistics through {meta["history_last_date"]}; '
          f'{len(merges)} duplicate player ids merged; parity verified.')


if __name__ == '__main__': main()
