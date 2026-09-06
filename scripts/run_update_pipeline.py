import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tennis_pipeline import recalculate_elo_artifacts, run_data_update


def main() -> int:
    print("[update] Downloading, enriching, and validating ATP snapshots...")
    report = run_data_update(PROJECT_ROOT)
    changes = report["publication_changes"]
    print(
        "[update] Data published: "
        f"rows={changes['current_rows']}, "
        f"added={changes['rows_added']}, "
        f"removed_or_revised={changes['rows_removed']}, "
        f"odds_coverage={report['legacy_odds_dataset']['odds']['coverage']:.2%}"
    )

    print("[update] Recalculating Elo files...")
    result = recalculate_elo_artifacts(
        PROJECT_ROOT,
        progress_callback=lambda msg: print(f"[elo] {msg}"),
    )

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
