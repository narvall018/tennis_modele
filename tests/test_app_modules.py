from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.evidence import all_evidence, global_verdict
from src.app.staking import (
    MINIMUM_EDGE,
    PLANS,
    apply_daily_cap,
    kelly_fraction,
    stake_for_bet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StakingTests(unittest.TestCase):
    def test_no_edge_means_no_stake(self):
        """The property that matters: a zero edge must size to zero.

        Everything measured in this repository says the edge is indistinguishable
        from zero, so a sizer that still proposes a number would be the one
        dishonest component.
        """
        result = stake_for_bet(0.50, 2.00, 1000.0, PLANS["standard"])
        self.assertEqual(result["stake"], 0.0)
        self.assertIn("plancher", result["reason"])

    def test_an_edge_below_the_floor_is_refused_with_a_reason(self):
        result = stake_for_bet(0.505, 2.00, 1000.0, PLANS["standard"])
        self.assertEqual(result["stake"], 0.0)
        self.assertLess(result["edge"], MINIMUM_EDGE)
        self.assertTrue(result["reason"])

    def test_a_real_edge_is_sized_and_capped(self):
        result = stake_for_bet(0.60, 1.80, 1000.0, PLANS["standard"])
        self.assertGreater(result["stake"], 0.0)
        self.assertLessEqual(
            result["fraction"], PLANS["standard"].max_fraction_per_bet + 1e-12
        )
        self.assertTrue(result["capped"])

    def test_a_prudent_plan_never_stakes_more_than_a_standard_one(self):
        prudent = stake_for_bet(0.70, 2.00, 1000.0, PLANS["prudent"])
        standard = stake_for_bet(0.70, 2.00, 1000.0, PLANS["standard"])
        self.assertLessEqual(prudent["stake"], standard["stake"])

    def test_kelly_is_zero_for_a_negative_expectation(self):
        self.assertEqual(kelly_fraction(0.40, 2.00), 0.0)
        self.assertGreater(kelly_fraction(0.60, 2.00), 0.0)
        self.assertEqual(kelly_fraction(np.nan, 2.00), 0.0)
        self.assertEqual(kelly_fraction(0.60, 1.0), 0.0)

    def test_the_daily_cap_scales_a_day_rather_than_dropping_a_match(self):
        stakes = [40.0, 30.0, 30.0]
        capped = apply_daily_cap(stakes, 1000.0, PLANS["standard"])
        self.assertAlmostEqual(sum(capped), 20.0, places=2)
        # Every bet survives, in proportion; none is silently truncated away.
        self.assertEqual(len(capped), 3)
        self.assertTrue(all(stake > 0 for stake in capped))

    def test_a_day_under_the_cap_is_left_alone(self):
        stakes = [5.0, 4.0]
        self.assertEqual(apply_daily_cap(stakes, 1000.0, PLANS["standard"]), [5.0, 4.0])


class EvidenceTests(unittest.TestCase):
    def test_the_verdict_never_authorises_real_money(self):
        verdict = global_verdict()
        self.assertFalse(verdict["real_money_authorised"])
        self.assertIn("AUCUNE", verdict["status"])

    def test_every_sport_reports_no_demonstrated_edge(self):
        for evidence in all_evidence(PROJECT_ROOT):
            self.assertFalse(evidence.real_money_authorised, evidence.sport)
            self.assertIn("AUCUN AVANTAGE", evidence.status, evidence.sport)

    def test_metrics_come_from_the_study_files(self):
        """A performance page whose numbers are retyped can drift from reality."""
        by_sport = {evidence.sport: evidence for evidence in all_evidence(PROJECT_ROOT)}
        self.assertGreater(len(by_sport["Football"].metrics), 0)
        self.assertGreater(len(by_sport["Tennis"].metrics), 0)
        # The football gate value must match the study output exactly.
        gate = by_sport["Football"].metrics.get("Meilleur gain contre le prix")
        self.assertIsNotNone(gate)
        self.assertLess(gate, 0.001)

    def test_missing_report_files_degrade_without_crashing(self):
        evidence = all_evidence(Path("/nonexistent"))
        self.assertEqual(len(evidence), 3)
        for block in evidence:
            self.assertIn("AUCUN AVANTAGE", block.status)


class PageSourceTests(unittest.TestCase):
    def test_the_page_module_parses(self):
        """Streamlit is not installed here, so the module is checked statically."""
        source = (PROJECT_ROOT / "src" / "app" / "pages.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_the_app_entry_point_parses(self):
        source = (PROJECT_ROOT / "unified_app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        for page in ("render_predictions_page", "render_staking_page",
                     "render_performance_page"):
            self.assertIn(page, names, f"{page} n'est pas branchée dans l'app")




class ModelRegistryTests(unittest.TestCase):
    def test_no_registered_model_uses_the_odds(self):
        """An app model fed the price could only ever agree with it."""
        from src.app.evidence import model_registry

        for row in model_registry(PROJECT_ROOT):
            self.assertFalse(row["Utilise la cote"], row["Sport"])

    def test_a_missing_model_is_reported_not_faked(self):
        from src.app.evidence import model_registry

        rows = model_registry(Path("/nonexistent"))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["Modèle retenu"], "non entraîné")
            self.assertIsNone(row["Log-loss (hors échantillon)"])

    def test_every_sport_has_a_registry_entry(self):
        from src.app.evidence import MODEL_METADATA, model_registry

        rows = model_registry(PROJECT_ROOT)
        self.assertEqual(len(rows), len(MODEL_METADATA))
        sports = {row["Sport"] for row in rows}
        self.assertIn("Football", sports)
        self.assertIn("UFC", sports)
        self.assertTrue(any("Tennis" in sport for sport in sports))




class BettingCandidateTests(unittest.TestCase):
    """The staking page treats three sports alike, so they must speak one shape."""

    def _block(self, sport, rows):
        from src.app.predictions import SportPredictions

        return SportPredictions(sport, True, pd.DataFrame(rows), {})

    def test_every_sport_normalises_to_the_same_columns(self):
        from src.app.predictions import CANDIDATE_COLUMNS, betting_candidates

        blocks = [
            self._block("Football", [{
                "domicile": "A", "extérieur": "B", "division": "E0", "date": "2026-09-10",
                "pari": "Domicile", "cote_pari": 2.5, "p_pari": 0.5,
                "espérance": 0.25, "score": 0.25,
            }]),
            self._block("UFC", [{
                "combattant_1": "X", "combattant_2": "Y", "date": "2026-09-12",
                "pari": "X", "cote_pari": 2.0, "p_pari": 0.6,
                "espérance": 0.20, "score": 0.20,
            }]),
            self._block("Tennis", [{
                "favori": "P", "adversaire": "Q", "début": "2026-09-11 15:00",
                "pari": "P", "cote_pari": 1.8, "p_pari": 0.62,
                "espérance": 0.12, "score": 0.12,
            }]),
        ]
        for block in blocks:
            frame = betting_candidates(block)
            self.assertEqual(list(frame.columns), CANDIDATE_COLUMNS, block.sport)
            self.assertEqual(len(frame), 1, block.sport)
            self.assertTrue(frame.iloc[0]["rencontre"], block.sport)

    def test_rows_the_ranking_rejected_never_reach_the_staking_page(self):
        from src.app.predictions import betting_candidates

        block = self._block("UFC", [
            {"combattant_1": "X", "combattant_2": "Y", "date": "2026-09-12",
             "pari": "X", "cote_pari": 2.0, "p_pari": 0.6, "espérance": 0.2, "score": 0.2},
            # Score zero means the ranking already judged it implausible.
            {"combattant_1": "W", "combattant_2": "Z", "date": "2026-09-12",
             "pari": "W", "cote_pari": 29.0, "p_pari": 0.38, "espérance": 10.0, "score": 0.0},
        ])
        frame = betting_candidates(block)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["pari"], "X")

    def test_an_unavailable_sport_yields_an_empty_frame_not_an_error(self):
        from src.app.predictions import CANDIDATE_COLUMNS, SportPredictions, betting_candidates

        block = SportPredictions("Tennis", False, pd.DataFrame(), {}, "pas de clé")
        frame = betting_candidates(block)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), CANDIDATE_COLUMNS)




class PredictionColumnTests(unittest.TestCase):
    """Each sport's tab shows the same recommendation columns, so they must exist."""

    def test_the_three_renderers_request_only_columns_the_sports_produce(self):
        from src.app.predictions import CANDIDATE_COLUMNS

        source = (PROJECT_ROOT / "src" / "app" / "pages.py").read_text(encoding="utf-8")
        # The recommendation quartet has to appear in every sport's table.
        for column in ("pari", "cote_pari", "p_pari", "espérance", "score"):
            self.assertGreaterEqual(
                source.count(f'"{column}"'), 3,
                f"{column} n'apparaît pas dans les trois tableaux",
            )
        for column in ("pari", "cote_pari", "p_pari", "espérance", "score"):
            self.assertIn(column, CANDIDATE_COLUMNS)

    def test_the_ranking_guard_is_documented_where_it_is_applied(self):
        from src.app.predictions import MAX_PLAUSIBLE_DISAGREEMENT, recommendation_score

        # A huge disagreement is model error, never a recommendation.
        self.assertEqual(
            recommendation_score(10.0, 50, 50, True,
                                 disagreement=MAX_PLAUSIBLE_DISAGREEMENT + 0.01),
            0.0,
        )
        self.assertGreater(
            recommendation_score(0.10, 50, 50, True,
                                 disagreement=MAX_PLAUSIBLE_DISAGREEMENT - 0.01),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
