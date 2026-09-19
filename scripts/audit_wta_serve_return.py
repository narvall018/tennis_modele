"""Read-only audit; independently recompute a fixed sample of delayed statistics."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtesting.wta_serve_return import STUDY, SOURCES, STATS_COLUMNS, prepare_stats, file_hash
from src.backtesting.wta_price_residual import select


def main():
    folder = ROOT/'models'/STUDY
    protocol = json.loads((folder/'protocol.json').read_text())
    registration = json.loads((folder/'registration.json').read_text())
    report = json.loads((folder/'report.json').read_text())
    for name, digest in {**registration['implementations'], **protocol['sources'],
            str((folder/'protocol.json').relative_to(ROOT)): registration['protocol_sha256']}.items():
        assert file_hash(ROOT/name) == digest, name
    frame = pd.read_parquet(folder/'features.parquet')
    assert frame._source_row_id.is_unique and frame._date.dt.year.le(2025).all()
    assert (frame.stats_max_start+pd.Timedelta(days=28) < frame._date).all()
    assert frame[['history_count_1', 'history_count_2']].ge(5).all().all()
    raw = pd.read_csv(ROOT/SOURCES[0])
    original = raw.loc[frame._source_row_id]
    np.testing.assert_array_equal(original[['Player_1', 'Player_2']], frame[['_p1', '_p2']])
    np.testing.assert_array_equal(original[['B365_1', 'B365_2']], frame[['price_0', 'price_1']])
    np.testing.assert_array_equal(np.where(original.Winner.eq(original.Player_1), 0, 1), frame._label)
    stats = prepare_stats(pd.read_csv(ROOT/SOURCES[1], usecols=STATS_COLUMNS, low_memory=False), protocol)
    for row in frame.sample(n=min(100, len(frame)), random_state=20260918).to_dict('records'):
        date = row['_date']
        history = stats[stats._valid & (stats._start+pd.Timedelta(days=28) < date)
                        & (stats._start >= date-pd.Timedelta(days=365))]
        for prefix, surface in [('global', None), ('surface', row['Surface'])]:
            vectors = []
            for identity in [row['id_1'], row['id_2']]:
                sums = np.zeros(4)
                subset = history if surface is None else history[history.surface.eq(surface)]
                for side, opponent, field in [('w', 'l', 'winner_id'), ('l', 'w', 'loser_id')]:
                    own = subset[subset[field].eq(identity)]
                    weight = np.exp2(-(date-own._start).dt.days.to_numpy()/180.)
                    counts = np.column_stack([own[side+'_won'], own[side+'_points'],
                        own[opponent+'_points']-own[opponent+'_won'], own[opponent+'_points']])
                    sums += weight@counts
                vectors.append(np.array([logit((sums[0]+300)/(sums[1]+500)),
                                         logit((sums[2]+200)/(sums[3]+500)), np.log1p(sums[1]+sums[3])]))
            np.testing.assert_allclose([row[prefix+'_'+s] for s in ['serve', 'return', 'volume']],
                                      vectors[0]-vectors[1], atol=1e-12)
    for fold in report['folds']:
        cutoff = pd.Timestamp(fold['year'], 1, 1)-pd.Timedelta(days=7)
        assert pd.Timestamp(fold['train_max_date']) < cutoff
        assert cutoff == pd.Timestamp(fold['cutoff_exclusive'])
        assert ((frame._date < cutoff) & frame._status.eq('completed')).sum() == fold['train_rows']
    reference = None
    for name, summary in report['evaluation'].items():
        predictions = pd.read_parquet(folder/f'{name}_evaluation_predictions.parquet')
        if reference is None: reference = predictions._source_row_id.to_numpy()
        np.testing.assert_array_equal(predictions._source_row_id, reference)
        p = predictions[['p0', 'p1']].to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        choice = select(p, predictions[['price_0', 'price_1']].to_numpy(float), protocol['selection'])
        bets = pd.read_parquet(folder/f'{name}_evaluation_bets.parquet')
        # This experiment had no signals: explicitly verify, not assume ROI=0.
        assert (choice == -1).all() and bets.empty and summary['roi']['0.02'] is None
        assert summary['bankroll']['final_bankroll'] == 1000.
        print(f'{name}: {len(predictions)} common predictions, no threshold-qualified bets confirmed.')
    print('PASS: frozen inputs; 100 independently recomputed profiles; 15 chronological fits; source odds/outcomes.')


if __name__ == '__main__':
    main()
