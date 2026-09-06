#!/usr/bin/env python3
"""CLI du pipeline UFC rigoureux."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigorous.data_pipeline import update_dataset
from rigorous.challenger_pipeline import run_challenger_holdout, run_challenger_research
from rigorous.model_pipeline import run_final_holdout, run_research
from rigorous.prospective_collector import collect_current_odds
from rigorous.phase3_pipeline import run_phase3_holdout, run_phase3_research
from rigorous.method_market import analyse_method_market
from rigorous.secondary_odds import update_secondary_odds


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "update-data", "cross-check-odds", "method-market", "research", "final-holdout", "challengers",
            "challenger-holdout", "phase3", "phase3-holdout", "collect-odds", "all",
        ],
    )
    args = parser.parse_args()
    if args.command == "update-data":
        result = update_dataset(BASE_DIR)
        print(json.dumps(result["quality"], indent=2, ensure_ascii=False))
    elif args.command == "method-market":
        print(json.dumps(analyse_method_market(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "cross-check-odds":
        print(json.dumps(update_secondary_odds(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "research":
        print(json.dumps(run_research(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "final-holdout":
        print(json.dumps(run_final_holdout(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "challengers":
        print(json.dumps(run_challenger_research(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "challenger-holdout":
        print(json.dumps(run_challenger_holdout(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "collect-odds":
        print(json.dumps(collect_current_odds(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "phase3":
        print(json.dumps(run_phase3_research(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "phase3-holdout":
        print(json.dumps(run_phase3_holdout(BASE_DIR), indent=2, ensure_ascii=False, default=str))
    elif args.command == "all":
        update_dataset(BASE_DIR)
        research = run_research(BASE_DIR)
        final = run_final_holdout(BASE_DIR)
        print(json.dumps({"research": research, "final": final}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
