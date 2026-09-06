from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from rigorous.data_pipeline import sha256_file  # noqa: E402
from rigorous.model_pipeline import simulate_bankroll  # noqa: E402
from rigorous.prospective_collector import normalise_odds_api_payload  # noqa: E402


def test_completed_data_is_current_and_unique():
    """The published table must agree with its own audit and still be recent.

    The last event date is deliberately not hardcoded: the UFC adds a card most
    weekends, so pinning a date turns every legitimate refresh into a failure.
    What has to hold is that the table matches the quality report it was
    published with, and that neither has gone stale.
    """
    fights = pd.read_parquet(BASE / "data/rigorous/processed/fights.parquet")
    quality = json.loads((BASE / "data/rigorous/quality/data_quality.json").read_text())
    latest = pd.to_datetime(fights["event_date"]).max()
    assert not fights["fight_id"].duplicated().any()
    assert latest == pd.Timestamp(quality["official_latest_completed_event"])
    assert latest <= pd.Timestamp.today().normalize()
    assert (pd.Timestamp.today().normalize() - latest).days <= 30
    assert quality["duplicate_fight_ids"] == 0


def test_timestamped_quotes_respect_frozen_cutoff():
    quotes = pd.read_parquet(BASE / "data/rigorous/processed/moneyline_quotes.parquet")
    timed = quotes[quotes["line_protocol"].str.startswith("pinnacle_else", na=False)].copy()
    event_utc = pd.to_datetime(timed["event_date"]).dt.tz_localize("UTC")
    collected = pd.to_datetime(timed["collected_at"], utc=True)
    cutoff = event_utc - pd.Timedelta(days=1)
    assert not timed.empty
    assert collected.le(cutoff).all()
    assert (cutoff - collected).le(pd.Timedelta(days=14)).all()
    assert set(timed["source"]).issubset({"Pinnacle", "BetOnline.ag"})


def test_lock_matches_protocol_and_holdout_stays_closed():
    protocol_path = BASE / "rigorous_protocol.json"
    reports = BASE / "data/rigorous/reports"
    lock = json.loads((reports / "locked_strategy.json").read_text())
    final = json.loads((reports / "final_holdout_report.json").read_text())
    assert lock["protocol_sha256"] == sha256_file(protocol_path)
    assert lock["approved_for_holdout"] is False
    assert lock["status"] == "REJECTED_NO_BET"
    assert final["status"] == "NOT_OPENED_VALIDATION_FAILED"
    assert not (BASE / "data/rigorous/processed/final_holdout_predictions.parquet").exists()
    assert not (reports / "final_holdout_bets.parquet").exists()


def test_one_bet_maximum_per_fight_in_validation():
    bets = pd.read_parquet(BASE / "data/rigorous/reports/validation_bets.parquet")
    assert not bets["fight_id"].duplicated().any()
    assert bets["odds"].between(1.25, 5.0).all()
    assert bets["edge"].ge(0.05 - 1e-12).all()


def test_event_stakes_use_same_bankroll_and_respect_exposure_cap():
    bets = pd.read_parquet(BASE / "data/rigorous/reports/validation_bets.parquet")
    protocol = json.loads((BASE / "rigorous_protocol.json").read_text())
    ledger, _ = simulate_bankroll(bets, protocol["staking"])
    assert ledger.groupby("event_id")["event_bankroll_start"].nunique().max() == 1
    exposure = ledger.groupby("event_id").agg(stake=("stake", "sum"), start=("event_bankroll_start", "first"))
    assert (exposure["stake"] / exposure["start"]).le(0.03 + 1e-12).all()


def test_challenger_lock_matches_protocol_and_holdout_stays_closed():
    protocol_path = BASE / "challenger_protocol.json"
    reports = BASE / "data/rigorous/reports"
    lock = json.loads((reports / "challenger_locked_strategy.json").read_text())
    final = json.loads((reports / "challenger_holdout_report.json").read_text())
    assert lock["protocol_sha256"] == sha256_file(protocol_path)
    assert lock["approved_for_pristine_holdout"] is False
    assert lock["status"] == "CHALLENGER_REJECTED_NO_BET"
    assert final["status"] == "NOT_OPENED_CHALLENGER_CONFIRMATION_FAILED"
    assert not (BASE / "data/rigorous/processed/challenger_holdout_predictions.parquet").exists()
    assert not (reports / "challenger_holdout_bets.parquet").exists()


def test_challenger_confirmation_uses_one_bet_per_fight_and_conservative_stakes():
    bets = pd.read_parquet(BASE / "data/rigorous/reports/challenger_confirmation_bets.parquet")
    protocol = json.loads((BASE / "challenger_protocol.json").read_text())
    assert not bets["fight_id"].duplicated().any()
    assert bets["odds"].between(1.25, 4.0).all()
    assert bets["edge"].ge(0.03 - 1e-12).all()
    ledger, _ = simulate_bankroll(bets, protocol["staking"])
    exposure = ledger.groupby("event_id").agg(stake=("stake", "sum"), start=("event_bankroll_start", "first"))
    assert (exposure["stake"] / exposure["start"]).le(0.02 + 1e-12).all()


def test_rankings_are_strictly_prefight_and_not_stale():
    rankings = pd.read_parquet(BASE / "data/rigorous/processed/prefight_rankings.parquet")
    event_date = pd.to_datetime(rankings["event_date"])
    assert rankings["fight_id"].is_unique
    assert rankings["division_rank_known_count"].gt(0).sum() >= 1900
    for family in ("division_rank", "p4p_rank"):
        for side in (1, 2):
            snapshot = pd.to_datetime(rankings[f"{family}_{side}_snapshot_date"])
            known = snapshot.notna()
            assert snapshot[known].lt(event_date[known]).all()
            assert rankings.loc[known, f"{family}_{side}_snapshot_age_days"].between(0, 14).all()


def test_line_trajectories_use_distinct_prefight_observations():
    trajectories = pd.read_parquet(BASE / "data/rigorous/processed/moneyline_trajectories.parquet")
    assert not trajectories.empty
    assert trajectories["fight_id"].is_unique
    assert set(trajectories["source"]).issubset({"Pinnacle", "BetOnline.ag"})
    for _, row in trajectories.iterrows():
        event_date = pd.Timestamp(row["event_date"]).tz_localize("UTC")
        observed = []
        for horizon in (14, 7, 3, 1):
            column = f"observed_at_tminus_{horizon}d"
            value = row.get(column)
            if pd.notna(value):
                timestamp = pd.Timestamp(value)
                observed.append(timestamp)
                cutoff = event_date - pd.Timedelta(days=horizon)
                assert timestamp <= cutoff
                assert cutoff - timestamp <= pd.Timedelta(days=3)
        assert len(set(observed)) >= 2


def test_phase3_failed_closed_without_opening_holdout():
    reports = BASE / "data/rigorous/reports"
    protocol = BASE / "phase3_protocol.json"
    lock = json.loads((reports / "phase3_locked_strategy.json").read_text())
    holdout = json.loads((reports / "phase3_holdout_report.json").read_text())
    assert lock["protocol_sha256"] == sha256_file(protocol)
    assert lock["approved_for_pristine_holdout"] is False
    assert lock["status"] == "PHASE3_REJECTED_NO_BET"
    assert holdout["status"] == "NOT_OPENED_PHASE3_DEVELOPMENT_GATE_FAILED"
    assert not (BASE / "data/rigorous/processed/phase3_holdout_predictions.parquet").exists()
    assert not (reports / "phase3_holdout_bets.parquet").exists()


def test_prospective_collector_preserves_timestamp_and_unverified_scope():
    payload = [
        {
            "id": "event-1",
            "sport_key": "mma_mixed_martial_arts",
            "sport_title": "MMA",
            "commence_time": "2026-09-10T20:00:00Z",
            "home_team": "Fighter B",
            "away_team": "Fighter A",
            "bookmakers": [
                {
                    "key": "testbook",
                    "title": "Test Book",
                    "last_update": "2026-09-02T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-09-02T12:00:00Z",
                            "outcomes": [
                                {"name": "Fighter B", "price": 2.1},
                                {"name": "Fighter A", "price": 1.8}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    fetched = pd.Timestamp("2026-09-02T12:01:00Z")
    result = normalise_odds_api_payload(payload, fetched)
    assert len(result) == 1
    assert result.loc[0, "fetched_at"] == fetched
    assert result.loc[0, "fighter_1"] == "Fighter A"
    assert result.loc[0, "ufc_match_status"] == "UNVERIFIED_MMA_EVENT"
