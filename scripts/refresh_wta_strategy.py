"""Refresh the portable WTA bundle without research caches or retraining."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app.wta_strategy_refresh import refresh

if __name__ == '__main__':
    try:
        refresh(ROOT, progress=lambda message: print(message, flush=True))
    except Exception as error:
        print(f'Actualisation WTA refusée : {error}', file=sys.stderr)
        raise SystemExit(1)
