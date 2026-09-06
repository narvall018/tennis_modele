"""Second, independent snapshot of historical UFC moneylines, used as a control.

The moneylines already in ``moneyline_quotes.parquet`` come in two very different
grades: 316 lines that carry a bookmaker and a pre-fight timestamp, and 6 040
legacy lines that carry neither.  Those 6 040 lines are the entire economic
sample before 2025, and nothing in the dataset says whether they are right.

This module downloads a *different* public compilation of the same historical
market (the "Ultimate UFC Dataset", sourced from BestFightOdds) and matches it
to the same fights.  It does not improve the temporal quality of anything: the
second source is untimestamped too, and is published as
``legacy_unverified_secondary``.  What it buys is falsifiability — where the two
independent compilations quote the same price for a fight, the legacy line is
corroborated; where they disagree, the fight is flagged and can be dropped from
an economic claim instead of silently carrying an unknown error.

The secondary price is never allowed to replace the primary one and is never
selected per fight by which of the two is more favourable.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .data_pipeline import fight_key, normalise_name, sha256_bytes, sha256_file


ULTIMATE_DATASET_SLUG = "mdabbert/ultimate-ufc-dataset"
ULTIMATE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{ULTIMATE_DATASET_SLUG}"
ULTIMATE_METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{ULTIMATE_DATASET_SLUG}"
ULTIMATE_CSV_NAME = "ufc-master.csv"

# Two prices for the same fight are treated as the same line when the implied
# no-vig probabilities differ by less than this. 1.5 points is roughly the width
# of a normal bookmaker-to-bookmaker spread on a UFC moneyline.
AGREEMENT_TOLERANCE = 0.015


def american_to_decimal(value: object) -> float:
    """Convert an American price to a decimal one, rejecting the invalid range."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or -100 < number < 100:
        return np.nan
    number = float(number)
    return 1.0 + (number / 100.0 if number > 0 else 100.0 / abs(number))


def fetch_ultimate_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    session = requests.Session()
    metadata_response = session.get(ULTIMATE_METADATA_URL, timeout=60)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    download_response = session.get(ULTIMATE_DOWNLOAD_URL, timeout=180)
    download_response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(download_response.content)) as archive:
        csv_bytes = archive.read(ULTIMATE_CSV_NAME)
    source = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    provenance = {
        "dataset": f"https://www.kaggle.com/datasets/{ULTIMATE_DATASET_SLUG}",
        "version": metadata.get("currentVersionNumber"),
        "last_updated": metadata.get("lastUpdated"),
        "license": metadata.get("licenseName"),
        "csv_sha256": sha256_bytes(csv_bytes),
        "rows": int(len(source)),
    }
    return source, provenance


def canonicalise_secondary_odds(source: pd.DataFrame) -> pd.DataFrame:
    """One row per fight, with prices attached to fighters by name, not by corner.

    ``Winner`` is present in the source and is deliberately not read: the
    orientation below depends only on the fighter names, so a mistake in the
    result column cannot leak into which side carries which price.
    """
    frame = source.copy()
    event_date = pd.to_datetime(frame["date"], errors="coerce")
    red = frame["R_fighter"].astype(str)
    blue = frame["B_fighter"].astype(str)
    keep = event_date.notna() & red.str.strip().ne("") & blue.str.strip().ne("")
    keep &= red.map(normalise_name).ne(blue.map(normalise_name))
    frame = frame[keep].copy()
    event_date = event_date[keep]

    result = pd.DataFrame(
        {
            "event_date": event_date.dt.normalize().to_numpy(),
            "red_fighter": red[keep].to_numpy(),
            "blue_fighter": blue[keep].to_numpy(),
            "red_name_key": red[keep].map(normalise_name).to_numpy(),
            "blue_name_key": blue[keep].map(normalise_name).to_numpy(),
            "red_odds": [american_to_decimal(value) for value in frame["R_odds"]],
            "blue_odds": [american_to_decimal(value) for value in frame["B_odds"]],
        }
    )
    for corner, prefix in (("r", "red"), ("b", "blue")):
        for market in ("ko", "sub", "dec"):
            column = f"{corner}_{market}_odds"
            result[f"{prefix}_{market}_odds"] = (
                [american_to_decimal(value) for value in frame[column]]
                if column in frame.columns
                else np.nan
            )
    result["fight_key"] = [
        fight_key(day, left, right)
        for day, left, right in zip(result["event_date"], result["red_fighter"], result["blue_fighter"])
    ]
    # A repeated key is an ambiguous pairing, not a duplicate row to be resolved.
    counts = result["fight_key"].value_counts()
    result = result[result["fight_key"].map(counts).eq(1)].copy()
    valid = result["red_odds"].gt(1.0) & result["blue_odds"].gt(1.0)
    return result[valid].reset_index(drop=True)


def match_secondary_to_fights(
    secondary: pd.DataFrame, fights: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align the secondary prices with this project's fight identifiers."""
    reference = fights[["fight_id", "fight_key", "event_date", "fighter_1", "fighter_2"]].copy()
    reference["f1_name_key"] = reference["fighter_1"].map(normalise_name)
    reference["f2_name_key"] = reference["fighter_2"].map(normalise_name)
    merged = reference.merge(secondary, on="fight_key", how="inner", suffixes=("", "_secondary"))

    forward = merged["f1_name_key"].eq(merged["red_name_key"])
    backward = merged["f1_name_key"].eq(merged["blue_name_key"])
    merged = merged[forward | backward].copy()
    forward = merged["f1_name_key"].eq(merged["red_name_key"])
    merged["odds_fighter_1"] = np.where(forward, merged["red_odds"], merged["blue_odds"])
    merged["odds_fighter_2"] = np.where(forward, merged["blue_odds"], merged["red_odds"])
    for market in ("ko", "sub", "dec"):
        merged[f"f1_{market}_odds"] = np.where(
            forward, merged[f"red_{market}_odds"], merged[f"blue_{market}_odds"]
        )
        merged[f"f2_{market}_odds"] = np.where(
            forward, merged[f"blue_{market}_odds"], merged[f"red_{market}_odds"]
        )

    inverse_1 = 1.0 / merged["odds_fighter_1"]
    inverse_2 = 1.0 / merged["odds_fighter_2"]
    merged["market_p1"] = inverse_1 / (inverse_1 + inverse_2)
    merged["overround"] = inverse_1 + inverse_2
    merged["source"] = "bestfightodds_compilation"
    merged["temporal_quality"] = "legacy_unverified_secondary"
    merged["line_protocol"] = "secondary_untimestamped_single_line"

    columns = [
        "fight_id", "fight_key", "event_date", "fighter_1", "fighter_2",
        "odds_fighter_1", "odds_fighter_2",
        "f1_ko_odds", "f2_ko_odds", "f1_sub_odds", "f2_sub_odds", "f1_dec_odds", "f2_dec_odds",
        "market_p1", "overround", "source", "temporal_quality", "line_protocol",
    ]
    matched = merged[columns].sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    report = {
        "secondary_rows": int(len(secondary)),
        "matched_fights": int(len(matched)),
        "unmatched_secondary_rows": int(len(secondary) - len(matched)),
        "coverage_of_secondary": float(len(matched) / len(secondary)) if len(secondary) else 0.0,
        "date_min": str(matched["event_date"].min().date()) if len(matched) else None,
        "date_max": str(matched["event_date"].max().date()) if len(matched) else None,
    }
    return matched, report


def cross_check(primary: pd.DataFrame, secondary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the two compilations fight by fight without preferring either."""
    left = primary[
        ["fight_id", "event_date", "market_p1", "overround", "temporal_quality", "source"]
    ].rename(
        columns={
            "market_p1": "primary_market_p1",
            "overround": "primary_overround",
            "temporal_quality": "primary_temporal_quality",
            "source": "primary_source",
        }
    )
    right = secondary[["fight_id", "market_p1", "overround"]].rename(
        columns={"market_p1": "secondary_market_p1", "overround": "secondary_overround"}
    )
    comparison = left.merge(right, on="fight_id", how="inner")
    comparison["probability_gap"] = (
        comparison["primary_market_p1"] - comparison["secondary_market_p1"]
    ).abs()
    comparison["agrees"] = comparison["probability_gap"].le(AGREEMENT_TOLERANCE)

    def summarise(frame: pd.DataFrame) -> dict[str, Any]:
        if not len(frame):
            return {"fights": 0}
        return {
            "fights": int(len(frame)),
            "agreement_rate": float(frame["agrees"].mean()),
            "median_probability_gap": float(frame["probability_gap"].median()),
            "p95_probability_gap": float(frame["probability_gap"].quantile(0.95)),
            "gap_above_5_points": int(frame["probability_gap"].gt(0.05).sum()),
            "mean_signed_gap": float(
                (frame["primary_market_p1"] - frame["secondary_market_p1"]).mean()
            ),
        }

    legacy = comparison[comparison["primary_temporal_quality"].eq("legacy_unverified")]
    identical_rate = float(legacy["probability_gap"].le(1e-9).mean()) if len(legacy) else 0.0
    if identical_rate >= 0.95:
        verdict = (
            "sources_are_not_independent: les prix legacy sont bit-a-bit ceux de la compilation "
            "secondaire, donc aucune corroboration independante n'existe pour cette periode"
        )
    elif identical_rate >= 0.5:
        verdict = "sources_partially_share_an_origin"
    else:
        verdict = "sources_are_independent"

    report = {
        "tolerance_probability_points": AGREEMENT_TOLERANCE,
        "legacy_bit_identical_rate": identical_rate,
        "independence_verdict": verdict,
        "overall": summarise(comparison),
        "by_primary_temporal_quality": {
            str(quality): summarise(group)
            for quality, group in comparison.groupby("primary_temporal_quality")
        },
        "by_year": {
            str(year): summarise(group)
            for year, group in comparison.groupby(comparison["event_date"].dt.year)
        },
    }
    return comparison, report


def update_secondary_odds(base_dir: Path) -> dict[str, Any]:
    """Publish the secondary lines and the corroboration report next to the primary ones."""
    processed_dir = base_dir / "data" / "rigorous" / "processed"
    raw_dir = base_dir / "data" / "rigorous" / "raw"
    quality_dir = base_dir / "data" / "rigorous" / "quality"
    for directory in (processed_dir, raw_dir, quality_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fights = pd.read_parquet(processed_dir / "fights.parquet")
    primary = pd.read_parquet(processed_dir / "moneyline_quotes.parquet")

    print("Telechargement de la compilation de cotes secondaire...", flush=True)
    source, provenance = fetch_ultimate_dataset()
    secondary_raw_path = raw_dir / "ufc_secondary_odds_source.parquet"
    source.to_parquet(secondary_raw_path, index=False)

    secondary = canonicalise_secondary_odds(source)
    matched, match_report = match_secondary_to_fights(secondary, fights)
    comparison, agreement = cross_check(primary, matched)

    secondary_path = processed_dir / "secondary_moneyline_quotes.parquet"
    comparison_path = processed_dir / "moneyline_cross_check.parquet"
    matched.to_parquet(secondary_path, index=False)
    comparison.to_parquet(comparison_path, index=False)

    primary_without_secondary = int(
        (~primary["fight_id"].isin(matched["fight_id"])).sum()
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "matching": match_report,
        "agreement": agreement,
        "primary_lines_without_a_second_opinion": primary_without_secondary,
        "fights_priced_only_by_the_secondary_source": int(
            (~matched["fight_id"].isin(primary["fight_id"])).sum()
        ),
        "artifacts": {
            str(path.relative_to(base_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (secondary_raw_path, secondary_path, comparison_path)
        },
        "usage_rules": [
            "La source secondaire n'est pas horodatee: elle n'ameliore pas la qualite temporelle.",
            "Elle ne remplace jamais la ligne primaire et n'est jamais choisie parce qu'elle paie mieux.",
            "Son seul role est de corroborer ou de signaler une ligne legacy avant toute conclusion economique.",
        ],
    }
    (quality_dir / "odds_cross_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report
