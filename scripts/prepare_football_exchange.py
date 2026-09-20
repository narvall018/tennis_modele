"""Freeze the football favourite-bias rule from the development seasons.

The rule is a table of bootstrap LOWER BOUNDS, not point estimates. A band whose
lower bound does not clear a full point is dropped, which is why the rule spans
only 0.70-0.85: the 0.60-0.70 bands stop at +0.14 and +0.15 points, which no
amount of wording makes different from no bias. Weeks are the bootstrap unit,
because matches on one weekend settle together.
"""
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app.football_exchange_strategy import STRATEGY_ID, FOLDER, digest, rule_digest

SOURCE = 'data/football/football_matches.csv.gz'
DEVELOPMENT = (2012, 2023)      # 2024-2026 stay closed as a preserved holdout
BANDS = [(.50, .55), (.55, .60), (.60, .65), (.65, .70), (.70, .75), (.75, .85), (.85, 1.01)]
DRAWS = 4000
SEED = 20260920
# A lower bound of a tenth of a point is not distinguishable from no bias at all.
# Demanding a full point keeps only bands the data actually supports, and drops
# the rest before they can dress up noise as a rule.
MINIMUM_LOWER_BOUND_POINTS = 1.0


def load():
    d = pd.read_csv(ROOT/SOURCE, low_memory=False)
    d['match_date'] = pd.to_datetime(d.match_date, errors='coerce')
    d = d[d.season_start.between(*DEVELOPMENT)]
    need = ['PSCH', 'PSCD', 'PSCA']
    keep = d[need].notna().all(axis=1) & (d[need] > 1).all(axis=1) & d.result.isin(['H', 'D', 'A'])
    return d[keep].copy()


def main():
    d = load()
    inverse = 1/d[['PSCH', 'PSCD', 'PSCA']].to_numpy(float)
    probabilities = inverse/inverse.sum(1, keepdims=True)
    outcomes = np.column_stack([d.result.eq('H'), d.result.eq('D'), d.result.eq('A')]).astype(int)
    row = np.arange(len(d))
    favourite = probabilities.argmax(1)
    implied = probabilities.max(1)
    won = outcomes[row, favourite]
    week = d.match_date.dt.to_period('W').astype(str).to_numpy()
    rng = np.random.default_rng(SEED)

    bands, dropped = [], []
    for low, high in BANDS:
        mask = (implied >= low) & (implied < high)
        if mask.sum() < 300:
            dropped.append({'from': low, 'to': high, 'reason': f'{int(mask.sum())} matchs'})
            continue
        frame = pd.DataFrame({'p': implied[mask], 'w': won[mask], 'week': week[mask]})
        groups = {k: v for k, v in frame.groupby('week')}
        keys = frame.week.unique()
        draws = np.array([(lambda s: s.w.mean()-s.p.mean())(
            pd.concat([groups[k] for k in rng.choice(keys, len(keys), True)])) for _ in range(DRAWS)])
        point = float(frame.w.mean()-frame.p.mean())
        lower = float(np.percentile(draws, 5))
        record = {'from': low, 'to': high, 'matches': int(mask.sum()),
                  'implied_mean': float(frame.p.mean()), 'realised_mean': float(frame.w.mean()),
                  'bias_points': 100*point, 'lower_bound_points': 100*lower}
        strong = 100*lower >= MINIMUM_LOWER_BOUND_POINTS
        (bands if strong else dropped).append(record if strong else {
            **record, 'reason': f'borne basse {100*lower:+.2f} pts sous le seuil '
                                f'de {MINIMUM_LOWER_BOUND_POINTS:.1f} pt'})
    if not bands:
        raise ValueError('Aucune bande ne prouve un biais : ne pas figer de règle.')

    rule = {'bias_bands': bands, 'commission': .05, 'min_expected_return': .01,
            'odds_min': 1.15, 'odds_max': 1.60, 'max_reference_overround': 1.02,
            'max_quote_age_seconds': 300, 'min_minutes_before_start': 10,
            'max_days_before_start': 7, 'bias_basis': 'week_cluster_bootstrap_5pct_lower_bound',
            'minimum_lower_bound_points': MINIMUM_LOWER_BOUND_POINTS}
    meta = {'strategy_id': STRATEGY_ID, 'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'rule': rule, 'rule_sha256': rule_digest(rule),
            'development': {'source': SOURCE, 'source_sha256': digest(ROOT/SOURCE),
                            'seasons': list(DEVELOPMENT), 'matches': int(len(d)),
                            'leagues': int(d.league.nunique()),
                            'reference_price': 'Pinnacle closing, proportional devig',
                            'median_overround': float(np.median(inverse.sum(1)))},
            'dropped_bands': dropped,
            'evidence': 'EXPLORATORY_PREVIOUSLY_EXPOSED_HISTORY; le holdout 2024-2026 n\'a pas '
                        'ete ouvert. La regle accumule des preuves prospectives, elle n\'en a pas.',
            'known_limits': [
                'Le biais est mesure sur des prix de cloture Pinnacle ; le scanner lit des prix '
                'exchange avant cloture. Sur les favoris la ligne ne bouge pas (0.6961 -> 0.6965 '
                'en developpement), mais ce report reste une hypothese.',
                'Un prix d exchange est une offre de profondeur inconnue : la cote affichee peut '
                'ne pas etre disponible pour la mise proposee.',
                'La commission reelle varie de 2 a 5 % ; la regle retient 5 %, le bout defavorable.',
                'Les saisons 2012-2023 ont deja servi a chercher : ce ne sont pas des preuves.'],
            'real_money_authorised': False}
    folder = ROOT/FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

    from src.app.football_exchange_strategy import load_bundle
    load_bundle(ROOT)
    print(f'Football {DEVELOPMENT[0]}-{DEVELOPMENT[1]} : {len(d)} matchs, {d.league.nunique()} ligues.')
    for b in bands:
        print(f"  bande {b['from']:.2f}-{b['to']:.2f} : n={b['matches']:5d} "
              f"biais {b['bias_points']:+.2f} pts, borne basse {b['lower_bound_points']:+.2f} pts")
    for b in dropped:
        print(f"  ecartee {b['from']:.2f}-{b['to']:.2f} : {b.get('reason')}")
    print(f"regle figee, sha256 {meta['rule_sha256'][:16]} ; argent reel non autorise.")


if __name__ == '__main__':
    main()
