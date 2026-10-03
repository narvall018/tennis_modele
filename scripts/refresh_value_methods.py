"""Daily automation of the two app methods; no wagers or promotions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app import value_methods as engine
from src.app import value_methods_data as data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-ufc-refresh', action='store_true')
    parser.add_argument('--skip-odds', action='store_true')
    args = parser.parse_args()
    status = {'at': engine.utc().isoformat(), 'ufc_refresh': 'skipped' if args.skip_ufc_refresh else 'pending',
              'errors': [], 'scans': {}}
    if not args.skip_ufc_refresh:
        try:
            data.refresh_ufc(ROOT, progress=lambda message: print(message, flush=True))
            status['ufc_refresh'] = 'ok'
        except Exception as error:
            status['ufc_refresh'] = 'error'
            status['errors'].append(f'UFCStats : {type(error).__name__}')
            print(status['errors'][-1], flush=True)
    bundle = cards = None
    try:
        bundle, cards = data.load_ufc(ROOT)
    except (ValueError, KeyError, TypeError):
        pass
    if not args.skip_odds:
        for sport in ('football', 'ufc'):
            snapshot = data.collect_live(ROOT, sport, force=True)
            results = [engine.analyse_event(sport, event, bundle=bundle, cards=cards) for event in snapshot['events']]
            status['scans'][sport] = {'at': snapshot['at'], 'requests': snapshot['requests'],
                'events': len(results), 'analysable': sum(r['status'] != 'blocked' for r in results),
                'signals': sum(r['candidate'] is not None for r in results),
                'quota_remaining': snapshot['remaining'], 'errors': snapshot['errors']}
            print(sport, status['scans'][sport], flush=True)
    data.atomic_json(ROOT / 'models/value_methods/automation_status.json', status)
    if status['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
