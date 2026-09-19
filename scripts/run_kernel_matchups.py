"""Register first; execute the six predeclared variants once."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtesting.kernel_matchups import register, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register:
        print(register(ROOT))
    else:
        run(ROOT, progress=lambda message: print(message, flush=True))
