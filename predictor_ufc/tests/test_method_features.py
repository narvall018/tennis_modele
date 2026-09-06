from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rigorous.method_features import (
    SYMMETRIC_FEATURES,
    build_method_features,
    method_category,
)


def _fight(index: int, date: str, event: str, f1: str, f2: str, method: str, y: float = 1.0,
           seconds: float = 900.0) -> dict:
    return {
        "fight_id": f"fight{index}",
        "event_id": event,
        "event_date": pd.Timestamp(date),
        "weight_class": "Lightweight",
        "method": method,
        "duration_secs": seconds,
        "fighter_1": f1,
        "fighter_2": f2,
        "fighter_1_id": f1,
        "fighter_2_id": f2,
        "y": y,
        "p1_sig_lnd": 40.0, "p2_sig_lnd": 30.0,
        "p1_td_lnd": 2.0, "p2_td_lnd": 1.0,
        "p1_kd": 0.0, "p2_kd": 0.0,
        "p1_sub_att": 1.0, "p2_sub_att": 0.0,
        "p1_ctrl_secs": 120.0, "p2_ctrl_secs": 60.0,
        "age_1": 30.0, "age_2": 28.0,
    }


class MethodFeatureTests(unittest.TestCase):
    def test_method_categories_cover_the_real_labels(self):
        self.assertEqual(method_category("KO/TKO"), "ko")
        self.assertEqual(method_category("TKO - Doctor's Stoppage"), "ko")
        self.assertEqual(method_category("Submission"), "sub")
        self.assertEqual(method_category("Decision - Split"), "dec")
        # A no-contest is not a method and must not be forced into one.
        self.assertIsNone(method_category("Overturned"))
        self.assertIsNone(method_category("DQ"))

    def test_a_fighters_first_fight_carries_no_record(self):
        frame = pd.DataFrame([_fight(1, "2020-01-01", "e1", "A", "B", "KO/TKO")])
        built = build_method_features(frame)
        self.assertEqual(built.iloc[0]["both_experience"], 0.0)
        self.assertEqual(built.iloc[0]["both_decision_rate"], 0.0)

    def test_the_current_fight_never_enters_its_own_features(self):
        """The decisive leakage test.

        Fighter A's second fight must be described by the first one only. If the
        builder updated state before reading it, A's decision rate would already
        include the fight being predicted.
        """
        frame = pd.DataFrame([
            _fight(1, "2020-01-01", "e1", "A", "B", "Decision - Unanimous"),
            _fight(2, "2020-06-01", "e2", "A", "C", "Decision - Unanimous"),
        ])
        built = build_method_features(frame).set_index("fight_id")
        # After exactly one decision, A's rate is 1.0 and C is a debutant, so the
        # pair mean is 0.5. A leaking builder would count the second fight too.
        self.assertAlmostEqual(built.loc["fight2", "both_decision_rate"], 0.5)
        self.assertAlmostEqual(built.loc["fight2", "min_decision_rate"], 0.0)

    def test_fights_on_one_card_share_the_same_pre_card_state(self):
        frame = pd.DataFrame([
            _fight(1, "2020-01-01", "e1", "A", "B", "Decision - Unanimous"),
            _fight(2, "2020-01-01", "e1", "A", "C", "KO/TKO"),
        ])
        built = build_method_features(frame).set_index("fight_id")
        self.assertEqual(
            built.loc["fight1", "both_experience"], built.loc["fight2", "both_experience"]
        )

    def test_features_are_symmetric_under_corner_swap(self):
        """Swapping corners must not change how a fight is expected to end."""
        history = [
            _fight(1, "2019-01-01", "e1", "A", "X", "Decision - Unanimous"),
            _fight(2, "2019-02-01", "e2", "B", "Y", "KO/TKO"),
        ]
        straight = build_method_features(
            pd.DataFrame(history + [_fight(3, "2020-01-01", "e3", "A", "B", "KO/TKO")])
        ).set_index("fight_id")
        swapped = build_method_features(
            pd.DataFrame(history + [_fight(3, "2020-01-01", "e3", "B", "A", "KO/TKO")])
        ).set_index("fight_id")
        for column in SYMMETRIC_FEATURES:
            left = straight.loc["fight3", column]
            right = swapped.loc["fight3", column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertAlmostEqual(float(left), float(right), places=9, msg=column)

    def test_unresolved_fights_get_no_label(self):
        frame = pd.DataFrame([_fight(1, "2020-01-01", "e1", "A", "B", "Overturned")])
        built = build_method_features(frame)
        self.assertTrue(np.isnan(built.iloc[0]["goes_to_decision"]))


if __name__ == "__main__":
    unittest.main()
