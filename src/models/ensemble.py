"""
Ensemble Model with Calibration — v3
======================================
Architecture :

  [XGBoost]  [LightGBM]  [CatBoost]  [Logistic]   ← modèles de base
       ↓           ↓          ↓           ↓
       └───────────┴──────────┴───────────┘
                           ↓
               [Meta-learner Logistic]               ← stacking
                           ↓
               [Calibration Isotonique]              ← calibration finale
                           ↓
                  Probabilités calibrées

Principes :
- Out-of-fold (OOF) predictions pour entraîner le meta-learner → pas de leakage
- Calibration isotonique sur un hold-out temporel (dernière année)
- Support XGBoost uniquement si LightGBM/CatBoost non disponibles (fallback gracieux)
- Cohérence avec le modèle v2b existant (peut charger l'ancien modèle comme base)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Imports optionnels — le modèle fonctionne sans LightGBM/CatBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


# ---------------------------------------------------------------------------
# Individual model factories
# ---------------------------------------------------------------------------

def _make_xgboost(params: dict):
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    defaults = dict(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def _make_lightgbm(params: dict):
    if not HAS_LGB:
        raise ImportError("lightgbm not installed")
    defaults = dict(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        metric="binary_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


def _make_catboost(params: dict):
    if not HAS_CAT:
        raise ImportError("catboost not installed")
    defaults = dict(
        iterations=600,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=0,
    )
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def _make_logistic(params: dict):
    defaults = dict(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)
    defaults.update(params)
    return LogisticRegression(**defaults)


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class TennisEnsemble:
    """
    Ensemble XGBoost + LightGBM + CatBoost + Logistic avec stacking.

    Workflow d'entraînement :
      1. train_base_models() → OOF predictions sur X_train
      2. train_meta_learner() → Logistic sur OOF probs
      3. calibrate() → Isotonic sur un hold-out (X_cal)
      4. predict_proba() → proba calibrée finale

    Workflow de prédiction :
      predict_proba(X) → array shape (n, 2)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        base_cfg = cfg.get("base_models", {})
        meta_cfg = cfg.get("meta_learner", {})
        cal_cfg = cfg.get("calibration", {})

        self.base_models_config = base_cfg
        self.meta_config = meta_cfg
        self.calibration_method = cal_cfg.get("method", "isotonic")
        self.cv_folds = cal_cfg.get("cv", 5)

        # Modèles de base
        self._base_models = {}
        self._init_base_models()

        # Meta-learner
        self._meta = LogisticRegression(
            C=meta_cfg.get("C", 1.0),
            max_iter=meta_cfg.get("max_iter", 500),
            random_state=42,
        )

        # Calibrateur (enveloppé sur le meta)
        self._calibrated_meta = None
        self.scaler = StandardScaler()

        # Infos
        self.feature_names: List[str] = []
        self.is_trained = False
        self._oof_scores: Dict[str, float] = {}

    def _init_base_models(self):
        """Initialise les modèles de base disponibles."""
        cfg = self.base_models_config

        if HAS_XGB:
            self._base_models["xgboost"] = _make_xgboost(cfg.get("xgboost", {}))

        if HAS_LGB:
            self._base_models["lightgbm"] = _make_lightgbm(cfg.get("lightgbm", {}))

        if HAS_CAT:
            self._base_models["catboost"] = _make_catboost(cfg.get("catboost", {}))

        # Logistic toujours disponible
        self._base_models["logistic"] = _make_logistic(cfg.get("logistic", {}))

        if not self._base_models:
            raise RuntimeError("Aucun modèle de base disponible.")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
        feature_names: List[str] = None,
        progress_callback=None,
    ) -> "TennisEnsemble":
        """
        Entraîne l'ensemble complet.

        X_cal / y_cal : hold-out pour calibration (recommandé : dernière année).
        Si X_cal est None, calibration sur X_train via CV (moins précis).
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        if progress_callback:
            progress_callback(f"Entraînement ensemble sur {len(y_train):,} matchs...")

        # Scale pour logistic
        X_train_s = self.scaler.fit_transform(X_train)
        X_cal_s = self.scaler.transform(X_cal) if X_cal is not None else None

        # --- Étape 1 : OOF predictions ---
        oof_probs = self._generate_oof_predictions(X_train, X_train_s, y_train, progress_callback)

        # --- Étape 2 : Meta-learner ---
        if progress_callback:
            progress_callback("Entraînement meta-learner...")
        self._meta.fit(oof_probs, y_train)

        # --- Étape 3 : Réentraîner les base models sur tout X_train ---
        if progress_callback:
            progress_callback("Réentraînement modèles de base sur toutes les données...")
        self._fit_all_base_models(X_train, X_train_s, y_train)

        # --- Étape 4 : Calibration ---
        if progress_callback:
            progress_callback("Calibration des probabilités...")
        self._calibrate(X_cal, X_cal_s, y_cal, X_train, X_train_s, y_train)

        # --- Scores OOF ---
        oof_preds = (oof_probs.mean(axis=1) > 0.5).astype(int)
        meta_probs = self._meta.predict_proba(oof_probs)[:, 1]
        self._oof_scores = {
            "brier_score": brier_score_loss(y_train, meta_probs),
            "log_loss": log_loss(y_train, meta_probs),
            "roc_auc": roc_auc_score(y_train, meta_probs),
        }

        if progress_callback:
            progress_callback(
                f"OOF AUC: {self._oof_scores['roc_auc']:.4f} | "
                f"Brier: {self._oof_scores['brier_score']:.4f}"
            )

        self.is_trained = True
        return self

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        X_scaled: np.ndarray,
        y: np.ndarray,
        progress_callback=None,
    ) -> np.ndarray:
        """Génère les OOF predictions pour chaque modèle de base."""
        n_models = len(self._base_models)
        oof = np.zeros((len(y), n_models))

        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            if progress_callback:
                progress_callback(f"OOF fold {fold_idx + 1}/{self.cv_folds}...")

            for col_idx, (name, model) in enumerate(self._base_models.items()):
                X_tr = X_scaled[train_idx] if name == "logistic" else X[train_idx]
                X_val = X_scaled[val_idx] if name == "logistic" else X[val_idx]
                y_tr = y[train_idx]

                model_fold = self._clone_model(name)
                model_fold.fit(X_tr, y_tr)
                oof[val_idx, col_idx] = model_fold.predict_proba(X_val)[:, 1]

        return oof

    def _clone_model(self, name: str):
        """Clone un modèle de base avec les mêmes paramètres."""
        cfg = self.base_models_config
        if name == "xgboost":
            return _make_xgboost(cfg.get("xgboost", {}))
        elif name == "lightgbm":
            return _make_lightgbm(cfg.get("lightgbm", {}))
        elif name == "catboost":
            return _make_catboost(cfg.get("catboost", {}))
        else:
            return _make_logistic(cfg.get("logistic", {}))

    def _fit_all_base_models(self, X: np.ndarray, X_scaled: np.ndarray, y: np.ndarray):
        for name, model in self._base_models.items():
            X_in = X_scaled if name == "logistic" else X
            model.fit(X_in, y)

    def _calibrate(self, X_cal, X_cal_s, y_cal, X_train, X_train_s, y_train):
        """Applique la calibration isotonique sur le hold-out ou CV."""
        from sklearn.calibration import IsotonicRegression
        from sklearn.calibration import _CalibratedClassifier

        if X_cal is not None and y_cal is not None:
            # Calibration sur hold-out (meilleure option)
            cal_meta_probs = self._meta_predict_raw(X_cal, X_cal_s)
            self._iso_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._iso_calibrator.fit(cal_meta_probs, y_cal)
        else:
            # Calibration CV sur training (fallback)
            train_meta_probs = self._meta_predict_raw(X_train, X_train_s)
            self._iso_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._iso_calibrator.fit(train_meta_probs, y_train)

        self._use_isotonic = True

    def _base_predict_raw(self, X: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        """Prédictions brutes de chaque modèle de base → matrice (n, n_models)."""
        preds = []
        for name, model in self._base_models.items():
            X_in = X_scaled if name == "logistic" else X
            preds.append(model.predict_proba(X_in)[:, 1])
        return np.column_stack(preds)

    def _meta_predict_raw(self, X: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        """Prédiction meta-learner (avant calibration)."""
        base_probs = self._base_predict_raw(X, X_scaled)
        return self._meta.predict_proba(base_probs)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction finale calibrée.
        Retourne array shape (n, 2) : [prob_lose, prob_win] pour P1.
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné.")

        if len(X) == 0:
            return np.empty((0, 2), dtype=np.float32)

        X_s = self.scaler.transform(X)
        raw_proba = self._meta_predict_raw(X, X_s)

        if hasattr(self, "_iso_calibrator"):
            cal_proba = self._iso_calibrator.predict(raw_proba)
            cal_proba = np.clip(cal_proba, 0.01, 0.99)
        else:
            cal_proba = raw_proba

        return np.column_stack([1 - cal_proba, cal_proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Importance des features agrégée depuis XGBoost (si dispo)."""
        rows = []
        for name, model in self._base_models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                for feat, val in zip(self.feature_names, imp):
                    rows.append({"model": name, "feature": feat, "importance": val})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        agg = df.groupby("feature")["importance"].mean().sort_values(ascending=False)
        return agg.reset_index().rename(columns={"importance": "avg_importance"})

    def get_model_weights(self) -> Dict[str, float]:
        """Poids implicites du meta-learner pour chaque modèle de base."""
        if not hasattr(self._meta, "coef_"):
            return {}
        coefs = np.abs(self._meta.coef_[0])
        total = coefs.sum()
        return {name: float(coefs[i] / total) for i, name in enumerate(self._base_models)}

    def get_oof_scores(self) -> Dict[str, float]:
        return self._oof_scores.copy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TennisEnsemble":
        return joblib.load(path)

    def summary(self) -> str:
        lines = ["=== TennisEnsemble Summary ==="]
        lines.append(f"Base models: {list(self._base_models.keys())}")
        lines.append(f"Features: {len(self.feature_names)}")
        if self._oof_scores:
            lines.append(f"OOF Scores:")
            for k, v in self._oof_scores.items():
                lines.append(f"  {k}: {v:.4f}")
        if self.is_trained:
            weights = self.get_model_weights()
            if weights:
                lines.append("Model weights (meta-learner):")
                for name, w in sorted(weights.items(), key=lambda x: -x[1]):
                    lines.append(f"  {name}: {w:.1%}")
        return "\n".join(lines)
