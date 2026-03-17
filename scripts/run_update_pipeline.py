import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app


def main() -> int:
    print("[update] Fetching new matches...")
    df_new = app.fetch_new_matches(progress_callback=lambda msg: print(f"[update] {msg}"))

    if df_new.empty:
        print("[update] No new matches found. Nothing to update.")
        return 0

    print(f"[update] New matches: {len(df_new)}")
    app.update_main_csv(df_new)

    print("[update] Recalculating Elo files...")
    result = app.recalculate_all_elo(progress_callback=lambda msg: print(f"[elo] {msg}"))

    print(
        "[update] Done: "
        f"matches={result['total_matches']}, "
        f"players={result['total_players']}, "
        f"recent={result['recent_matches']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[update] ERROR: {exc}", file=sys.stderr)
        raise
