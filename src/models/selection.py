"""Pick the best probability model for a sport, the same way for every sport.

"The best model" is a selection, and a selection made on the data it is then
scored on is worthless. So the procedure here is fixed and identical across
sports:

* candidates are compared **walk-forward** — a fold is trained only on seasons
  strictly before it, so no candidate ever sees its own test period;
* the winner is chosen on the **development window only**, by log-loss;
* the quality finally reported comes from an **evaluation window the selection
  never looked at**, which is what makes the number quotable.

Log-loss is the criterion rather than accuracy because the app shows
probabilities, and a model that is right slightly more often but overconfident is
worse for every downstream use — staking most of all.

None of this produces a betting edge, and a better model here does not change
that: the conditional tests in this repository measure the gain *against the
price*, and that is a different and much harder bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


MINIMUM_TRAIN_ROWS = 2000
MINIMUM_TEST_ROWS = 200


def _linear(regularisation: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=regularisation, max_iter=4000)),
    ])


def _boosted(depth: int, leaf: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer()),
        ("model", HistGradientBoostingClassifier(
            max_depth=depth, learning_rate=0.04, max_iter=400,
            min_samples_leaf=leaf, l2_regularization=1.0, random_state=0,
        )),
    ])


def _forest() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer()),
        ("model", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=40,
            random_state=0, n_jobs=-1,
        )),
    ])


def _xgboost(depth: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer()),
        ("model", XGBClassifier(
            max_depth=depth, learning_rate=0.04, n_estimators=400,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=0, tree_method="hist", eval_metric="logloss",
        )),
    ])


def candidate_factories(multiclass: bool) -> dict[str, Callable[[], Any]]:
    """The families compared, fixed in advance for every sport.

    Both a plain and an isotonically calibrated version of each family is
    offered: raw boosted trees are usually overconfident, and the app publishes
    probabilities, so a family that only wins after calibration should be allowed
    to win.
    """
    factories: dict[str, Callable[[], Any]] = {
        "logistique_forte": lambda: _linear(0.1),
        "logistique_douce": lambda: _linear(1.0),
        "gradient_boosting": lambda: _boosted(3, 60),
        "gradient_boosting_profond": lambda: _boosted(5, 30),
        "foret_aleatoire": _forest,
    }
    if HAS_XGBOOST and not multiclass:
        factories["xgboost"] = lambda: _xgboost(4)
    elif HAS_XGBOOST:
        factories["xgboost"] = lambda: _xgboost(4)

    calibrated = {
        f"{name}_calibre": (lambda factory=factory: CalibratedClassifierCV(
            factory(), method="isotonic", cv=3
        ))
        for name, factory in factories.items()
        if name.startswith(("gradient", "foret", "xgboost"))
    }
    factories.update(calibrated)
    return factories


@dataclass
class SelectionResult:
    winner: str
    comparison: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    fitted: Any = None


def _metrics(labels: np.ndarray, probabilities: np.ndarray, classes: Sequence[int]) -> dict[str, float]:
    metrics = {
        "n": int(len(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(classes))),
    }
    if len(classes) == 2:
        positive = probabilities[:, 1]
        metrics["brier"] = float(brier_score_loss(labels, positive))
        metrics["auc"] = float(roc_auc_score(labels, positive))
    else:
        metrics["auc"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", labels=list(classes))
        )
    return metrics


def walk_forward_scores(
    factory: Callable[[], Any],
    features: np.ndarray,
    labels: np.ndarray,
    periods: np.ndarray,
    test_periods: Sequence[Any],
    classes: Sequence[int],
) -> dict[str, float] | None:
    """Train strictly on the past of each test period, then score it."""
    predictions, truth = [], []
    for period in test_periods:
        train = periods < period
        test = periods == period
        if train.sum() < MINIMUM_TRAIN_ROWS or test.sum() < MINIMUM_TEST_ROWS:
            continue
        model = factory()
        model.fit(features[train], labels[train])
        predictions.append(model.predict_proba(features[test]))
        truth.append(labels[test])
    if not truth:
        return None
    return _metrics(np.concatenate(truth), np.vstack(predictions), classes)


def select_best_model(
    features: np.ndarray,
    labels: np.ndarray,
    periods: np.ndarray,
    development_periods: Sequence[Any],
    evaluation_periods: Sequence[Any],
    progress=None,
    families: Sequence[str] | None = None,
) -> SelectionResult:
    """Compare families on development, then score the winner on unseen periods.

    ``families`` restricts the search to named candidates. It exists so a caller
    can trade breadth for time deliberately; leaving it unset compares them all,
    which is what any published result should do.
    """
    classes = sorted(set(int(label) for label in labels))
    multiclass = len(classes) > 2
    comparison: list[dict[str, Any]] = []

    factories = candidate_factories(multiclass)
    if families is not None:
        unknown = sorted(set(families) - set(factories))
        if unknown:
            raise ValueError(f"Familles inconnues: {unknown}")
        factories = {name: factories[name] for name in families}
    for name, factory in factories.items():
        if progress:
            progress(f"  candidat {name}…")
        scores = walk_forward_scores(
            factory, features, labels, periods, development_periods, classes
        )
        if scores is None:
            continue
        comparison.append({"model": name, **scores})
    if not comparison:
        raise RuntimeError("Aucun candidat n'a pu être évalué: fenêtres trop courtes")

    comparison.sort(key=lambda row: row["log_loss"])
    winner = comparison[0]["model"]
    if progress:
        progress(f"  retenu: {winner} (log-loss {comparison[0]['log_loss']:.5f})")

    # The winner is refitted on everything before the evaluation window, then
    # scored there — a window no candidate was ranked on.
    evaluation = {}
    fitted = None
    if evaluation_periods:
        boundary = min(evaluation_periods)
        train = periods < boundary
        test = np.isin(periods, list(evaluation_periods))
        if train.sum() >= MINIMUM_TRAIN_ROWS and test.sum() >= MINIMUM_TEST_ROWS:
            fitted = factories[winner]()
            fitted.fit(features[train], labels[train])
            evaluation = _metrics(
                labels[test], fitted.predict_proba(features[test]), classes
            )
            evaluation["periods"] = [str(period) for period in evaluation_periods]
    if fitted is None:
        fitted = factories[winner]()
        fitted.fit(features, labels)
    return SelectionResult(winner, comparison, evaluation, fitted)
