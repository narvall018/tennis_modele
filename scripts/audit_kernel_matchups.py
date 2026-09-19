"""Read-only audit of frozen inputs, selections, settlement and cash ledgers."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtesting.kernel_matchups import STUDY, CANDIDATES, load_frame, file_hash, select


def main():
    folder = ROOT/'models'/STUDY
    protocol = json.loads((folder/'protocol.json').read_text())
    registration = json.loads((folder/'registration.json').read_text())
    report = json.loads((folder/'report.json').read_text())
    for path, digest in {**protocol['sources'], **registration['implementations'],
            str((folder/'protocol.json').relative_to(ROOT)): registration['protocol_sha256']}.items():
        assert file_hash(ROOT/path) == digest, path
    locks = json.loads((folder/'all_selection_locks.json').read_text())
    for sport in protocol['sports']:
        frame = load_frame(ROOT, sport, protocol)
        indexed = frame.set_index('_source_row_id')
        count = 3 if sport == 'football' else 2
        for fold in [f for f in report['folds'] if f['sport'] == sport]:
            cutoff = pd.Timestamp(fold['year'], 1, 1)-pd.Timedelta(days=7)
            begin = cutoff-pd.DateOffset(years=6)
            train = frame._date.lt(cutoff) & frame._date.ge(begin) & frame._status.eq('completed')
            assert fold['train_rows'] == int(train.sum())
            assert str(cutoff.date()) == fold['cutoff_exclusive']
            assert pd.Timestamp(fold['train_max']) < cutoff and pd.Timestamp(fold['train_min']) >= begin
        for candidate in CANDIDATES:
            for period in ['tuning', 'evaluation']:
                rows = pd.read_parquet(folder/f'{sport}_{candidate}_{period}_predictions.parquet')
                bets = pd.read_parquet(folder/f'{sport}_{candidate}_{period}_bets.parquet')
                summary = (locks[sport]['tuning'][candidate] if period == 'tuning'
                           else report['evaluation'][sport][candidate])
                assert rows._source_row_id.is_unique and bets._source_row_id.is_unique
                source = indexed.loc[rows._source_row_id]
                np.testing.assert_array_equal(source._label, rows._label)
                np.testing.assert_array_equal(source._status, rows._status)
                np.testing.assert_array_equal(source._date, rows._date)
                prices = rows[[f'price_{j}' for j in range(count)]].to_numpy(float)
                np.testing.assert_array_equal(source[[f'price_{j}' for j in range(count)]], prices)
                p = rows[[f'p{j}' for j in range(count)]].to_numpy(float)
                assert np.isfinite(p).all() and (p > 0).all()
                np.testing.assert_allclose(p.sum(axis=1), 1)
                chosen = select(p, prices, protocol['selection'])
                selection = rows.assign(choice=chosen, order=[
                    '|'.join(sorted([a, b])) for a, b in zip(rows._p1, rows._p2)])
                selection = selection[selection.choice >= 0].sort_values([
                    '_date', '_tournament', 'order', '_source_row_id'])
                assert len(selection) == summary['bankroll']['theoretical_signals']
                balance = 100000; record = 0; skipped = 0
                for _, day in selection.groupby('_date', sort=True):
                    base = balance; budget = int(np.floor(base*.02)); fixed = int(np.floor(base*.0025))
                    gain = 0
                    for raw in day.to_dict('records'):
                        stake = min(fixed, budget, base)
                        if stake <= 0: skipped += 1; continue
                        row = bets.iloc[record]; record += 1; budget -= stake
                        assert row._source_row_id == raw['_source_row_id']
                        outcome = int(raw['choice']); odds = raw[f'price_{outcome}']
                        settled = raw['_status'] == 'completed'; won = settled and outcome == raw['_label']
                        assert row.selected_original_outcome == outcome and row.original_label == raw['_label']
                        assert bool(row.won) == won and bool(row.settled) == settled
                        assert row.odds == odds and row.stake_cash == stake/100
                        assert row.bankroll_before == base/100
                        for haircut in [0., .02, .05]:
                            expected = 0. if not settled else (odds-1)*(1-haircut) if won else -1.
                            assert np.isclose(row[f'return_{haircut}'], expected)
                        profit = round(stake*row['return_0.02'])
                        assert row.profit_cash == profit/100
                        gain += profit
                    balance += gain
                assert record == len(bets) and skipped == summary['bankroll']['skipped_daily_budget']
                assert balance/100 == summary['bankroll']['final_bankroll']
                assert int(bets.settled.sum()) == summary['settled']
                print(f'{sport}/{candidate}/{period}: {len(bets)} selections, cash and void rules verified.')
    assert len(report['folds']) == 32
    print('PASS: all frozen hashes, 32 past-only folds, 12 ledgers and original outcome selections.')


if __name__ == '__main__':
    main()
