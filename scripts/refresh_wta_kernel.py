from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app.wta_kernel_refresh import refresh

if __name__ == '__main__':
    print('Statistiques WTA actualisées jusqu’au', refresh(ROOT))
