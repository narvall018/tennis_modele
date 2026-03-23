"""
Pipeline d'Entraînement — v3 Haute Fréquence
=============================================
Orchestre l'ensemble du pipeline :

  1. Chargement des données ATP (+ WTA si disponible)
  2. Calcul des Elo multi-variantes (TennisEloEngine)
  3. Construction des features v3 (FeatureBuilder)
  4. Entraînement de l'ensemble XGBoost + LightGBM + CatBoost
  5. Calibration isotonique sur hold-out temporel
  6. Backtesting walk-forward (4 ans train / 1 an test)
  7. Sélection de la meilleure stratégie par ROI ajusté risque
  8. Sauvegarde du modèle, scaler, elo_engine, config

Lancement :
    python train_pipeline.py [--config config.yaml] [--backtest] [--no-wta]

Output :
    models/ensemble_v3.pkl
    models/elo_engine_v3.pkl
    models/model_config_v3.pkl
    models/backtest_report_v3.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ajoute le répertoire racine au path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.elo_system import TennisEloEngine
from src.features.feature_builder import FeatureBuilder, FEATURE_COLS_V3
from src.models.ensemble import TennisEnsemble
from src.strategies.strategy_manager import StrategyManager
from src.backtesting.walk_forward import WalkForwardBacktester

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_config(path: str = "config.yaml") -> dict:
    if HAS_YAML and Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    log("⚠️  config.yaml non trouvé ou pyyaml absent — utilisation config par défaut")
    return _default_config()


def _default_config() -> dict:
    return {
        "data": {
            "atp_csv": "data/atp_tennis.csv",
            "wta_csv": "data/wta_tennis.csv",
            "models_dir": "models",
        },
        "elo": {
            "k_global": 32, "k_surface": 40, "k_momentum": 24,
            "initial_rating": 1500,
            "time_decay": {"enabled": True, "half_life_days": 180, "min_factor": 0.85},
        },
        "models": {
            "base_models": {},
            "calibration": {"method": "isotonic"},
            "version": "v3",
        },
        "strategies": {
            "ultra_confidence": {
                "name": "Ultra Confiance (≥72%)", "enabled": True,
                "model_threshold": 0.72, "min_edge": 0.05, "min_ev": 0.03,
                "series_filter": [], "surface_filter": [], "round_filter": [],
                "min_odds": 1.30, "max_odds": 4.00,
                "stake_type": "kelly_quarter", "kelly_fraction": 4.0,
                "max_stake_pct": 0.05, "min_stake_pct": 0.01,
            },
            "standard_volume": {
                "name": "Standard Volume (≥62%)", "enabled": True,
                "model_threshold": 0.62, "min_edge": 0.03, "min_ev": 0.02,
                "series_filter": ["Grand Slam", "Masters Cup", "Masters 1000", "ATP500"],
                "surface_filter": [], "round_filter": [],
                "min_odds": 1.25, "max_odds": 5.00,
                "stake_type": "kelly_sixth", "kelly_fraction": 6.0,
                "max_stake_pct": 0.04, "min_stake_pct": 0.01,
            },
            "high_volume": {
                "name": "Haute Fréquence (≥57%)", "enabled": True,
                "model_threshold": 0.57, "min_edge": 0.02, "min_ev": 0.01,
                "series_filter": [], "surface_filter": [], "round_filter": [],
                "min_odds": 1.20, "max_odds": 6.00,
                "stake_type": "flat", "flat_pct": 0.015,
                "max_stake_pct": 0.03, "min_stake_pct": 0.01,
            },
            "portfolio": {
                "max_daily_exposure": 0.10,
                "max_concurrent_bets": 8,
                "max_single_tournament": 0.08,
                "drawdown_limit": 0.20,
            },
        },
        "backtesting": {
            "train_years": 4,
            "initial_bankroll": 1000.0,
            "bookmaker_margin": 0.05,
        },
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_atp_data(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["atp_csv"]
    if not path.exists():
        log(f"❌ Données ATP non trouvées : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player_1", "Player_2"]).copy()

    # Normaliser la colonne Winner
    if "Winner" not in df.columns:
        if "winner" in df.columns:
            df["Winner"] = df["winner"]
        else:
            df["Winner"] = df["Player_1"]  # fallback

    # Colonnes optionnelles
    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
        if col not in df.columns:
            df[col] = 0

    df["Source"] = "ATP"
    log(f"✅ ATP : {len(df):,} matchs chargés ({df['Date'].min().year}–{df['Date'].max().year})")
    return df


def load_wta_data(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"].get("wta_csv", "data/wta_tennis.csv")
    if not path.exists():
        log("⚠️  Données WTA non trouvées — entraînement ATP uniquement")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player_1", "Player_2"]).copy()
    df["Source"] = "WTA"
    log(f"✅ WTA : {len(df):,} matchs chargés")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str = "config.yaml",
    run_backtest: bool = True,
    use_wta: bool = True,
    atp_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Pipeline complet d'entraînement.
    Retourne un dict avec les objets entraînés.
    """

    t0 = time.time()
    cfg = load_config(config_path)
    models_dir = ROOT / cfg["data"]["models_dir"]
    models_dir.mkdir(exist_ok=True)

    cb = log if verbose else lambda msg: None

    # -----------------------------------------------------------------------
    # 1. Chargement données
    # -----------------------------------------------------------------------
    cb("=" * 60)
    cb("ÉTAPE 1 : Chargement des données")
    cb("=" * 60)

    df_atp = load_atp_data(cfg)
    df_wta = load_wta_data(cfg) if use_wta and not atp_only else pd.DataFrame()

    frames = [f for f in [df_atp, df_wta] if not f.empty]
    if not frames:
        raise RuntimeError("Aucune donnée disponible.")

    df_all = pd.concat(frames, ignore_index=True).sort_values("Date")
    cb(f"Total : {len(df_all):,} matchs | {df_all['Date'].min().year}–{df_all['Date'].max().year}")

    # -----------------------------------------------------------------------
    # 2. Calcul Elo
    # -----------------------------------------------------------------------
    cb("\n" + "=" * 60)
    cb("ÉTAPE 2 : Calcul des ratings Elo multi-variantes")
    cb("=" * 60)

    elo_cfg = cfg.get("elo", {})
    decay_cfg = elo_cfg.get("time_decay", {})

    elo_engine = TennisEloEngine(
        k_global=elo_cfg.get("k_global", 32),
        k_surface=elo_cfg.get("k_surface", 40),
        k_momentum=elo_cfg.get("k_momentum", 24),
        initial_rating=elo_cfg.get("initial_rating", 1500),
        half_life_days=decay_cfg.get("half_life_days", 180),
        decay_enabled=decay_cfg.get("enabled", True),
        min_decay_factor=decay_cfg.get("min_factor", 0.85),
    )

    elo_engine.fit(df_all, progress_callback=cb)
    elo_history = elo_engine.get_history()
    cb(f"✅ Elo calculé pour {len(elo_engine.get_all_ratings()):,} joueurs")

    # Afficher top 10
    top10 = elo_engine.get_ratings_dataframe().head(10)
    cb(f"\nTop 10 joueurs (Elo global) :")
    for _, row in top10.iterrows():
        cb(f"  {row['player']:<25} {row['global_elo']:.0f}")

    # -----------------------------------------------------------------------
    # 3. Feature Engineering
    # -----------------------------------------------------------------------
    cb("\n" + "=" * 60)
    cb("ÉTAPE 3 : Construction des features (v3 — 48 features)")
    cb("=" * 60)

    # Ajouter colonnes Rank/Pts depuis df_all si disponibles
    elo_history_merged = elo_history.copy()
    rank_cols = [c for c in df_all.columns if c in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]]
    if rank_cols:
        df_ranks = df_all[["Date", "Player_1", "Player_2"] + rank_cols].copy()
        df_ranks["Date"] = pd.to_datetime(df_ranks["Date"])
        elo_history_merged = elo_history_merged.merge(
            df_ranks, on=["Date", "Player_1", "Player_2"], how="left"
        )

    builder = FeatureBuilder(feature_cols=FEATURE_COLS_V3)

    cb("Construction du dataset de features (peut prendre 5-15 min selon taille)...")
    feature_df = builder.build_dataset(elo_history_merged, progress_callback=cb)

    # Nettoyer les NaN
    n_before = len(feature_df)
    feature_df = feature_df.dropna(subset=FEATURE_COLS_V3).reset_index(drop=True)
    cb(f"✅ Features : {len(feature_df):,} matchs ({n_before - len(feature_df)} supprimés pour NaN)")

    # -----------------------------------------------------------------------
    # 4. Split train / hold-out calibration / test final
    # -----------------------------------------------------------------------
    cb("\n" + "=" * 60)
    cb("ÉTAPE 4 : Split temporel train / calibration / test")
    cb("=" * 60)

    feature_df["_year"] = pd.to_datetime(feature_df["_date"]).dt.year
    years = sorted(feature_df["_year"].unique())

    test_year = years[-1]
    cal_year = years[-2]
    train_mask = feature_df["_year"] < cal_year
    cal_mask = feature_df["_year"] == cal_year
    test_mask = feature_df["_year"] == test_year

    df_train = feature_df[train_mask]
    df_cal = feature_df[cal_mask]
    df_test = feature_df[test_mask]

    X_train = df_train[FEATURE_COLS_V3].values.astype(np.float32)
    y_train = df_train["_label"].values.astype(int)
    X_cal = df_cal[FEATURE_COLS_V3].values.astype(np.float32)
    y_cal = df_cal["_label"].values.astype(int)
    X_test = df_test[FEATURE_COLS_V3].values.astype(np.float32)
    y_test = df_test["_label"].values.astype(int)

    cb(f"Train : {len(X_train):,} | Calibration : {len(X_cal):,} | Test : {len(X_test):,}")

    # -----------------------------------------------------------------------
    # 5. Entraînement de l'ensemble
    # -----------------------------------------------------------------------
    cb("\n" + "=" * 60)
    cb("ÉTAPE 5 : Entraînement ensemble XGBoost + LightGBM + CatBoost")
    cb("=" * 60)

    model_cfg = cfg.get("models", {})
    ensemble = TennisEnsemble(config=model_cfg)

    ensemble.fit(
        X_train, y_train,
        X_cal=X_cal, y_cal=y_cal,
        feature_names=FEATURE_COLS_V3,
        progress_callback=cb,
    )

    cb(f"\n{ensemble.summary()}")

    # Évaluation sur test final
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    test_probs = ensemble.predict_proba(X_test)[:, 1]
    cb(f"\n📊 Performance sur test ({test_year}) :")
    cb(f"  AUC ROC   : {roc_auc_score(y_test, test_probs):.4f}")
    cb(f"  Brier     : {brier_score_loss(y_test, test_probs):.4f}")
    cb(f"  Log Loss  : {log_loss(y_test, test_probs):.4f}")

    # Feature importance
    fi = ensemble.get_feature_importance()
    if not fi.empty:
        cb("\n🔑 Top 10 features (importance moyenne) :")
        for _, row in fi.head(10).iterrows():
            cb(f"  {row['feature']:<30} {row['avg_importance']:.4f}")

    # -----------------------------------------------------------------------
    # 6. Backtesting walk-forward
    # -----------------------------------------------------------------------
    backtest_report = None
    if run_backtest:
        cb("\n" + "=" * 60)
        cb("ÉTAPE 6 : Backtesting walk-forward")
        cb("=" * 60)

        bt_cfg = cfg.get("backtesting", {})
        strat_cfg = cfg.get("strategies", {})

        def model_factory():
            return TennisEnsemble(config=model_cfg)

        strategy_manager = StrategyManager.from_config(strat_cfg)

        backtester = WalkForwardBacktester(
            model_factory=model_factory,
            strategy_manager=strategy_manager,
            feature_cols=FEATURE_COLS_V3,
            train_years=bt_cfg.get("train_years", 4),
            initial_bankroll=bt_cfg.get("initial_bankroll", 1000.0),
            bookmaker_margin=bt_cfg.get("bookmaker_margin", 0.05),
            progress_callback=cb,
        )

        # Backtest sur la fenêtre temporelle complète disponible
        backtest_df = feature_df[feature_df["_year"] < test_year].copy()
        backtest_report = backtester.run(backtest_df)

        # Sauvegarder le rapport
        report_path = models_dir / "backtest_report_v3.pkl"
        joblib.dump(backtest_report, report_path)
        cb(f"\n✅ Rapport backtest sauvegardé : {report_path}")

    # -----------------------------------------------------------------------
    # 7. Sauvegarde
    # -----------------------------------------------------------------------
    cb("\n" + "=" * 60)
    cb("ÉTAPE 7 : Sauvegarde des modèles")
    cb("=" * 60)

    # Ensemble
    ensemble_path = models_dir / "ensemble_v3.pkl"
    ensemble.save(str(ensemble_path))
    cb(f"✅ Ensemble sauvegardé : {ensemble_path}")

    # Elo engine
    elo_path = models_dir / "elo_engine_v3.pkl"
    elo_engine.save(str(elo_path))
    cb(f"✅ Elo engine sauvegardé : {elo_path}")

    # Config modèle (pour production)
    model_config = {
        "version": "v3",
        "feature_cols": FEATURE_COLS_V3,
        "n_features": len(FEATURE_COLS_V3),
        "trained_on": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_year": test_year,
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "test_brier": float(brier_score_loss(y_test, test_probs)),
        "strategies": cfg.get("strategies", {}),
        "base_models": list(ensemble._base_models.keys()),
        "model_weights": ensemble.get_model_weights(),
    }

    # Sélection de la meilleure stratégie
    if backtest_report and backtest_report.aggregate:
        best_strat = max(
            backtest_report.aggregate.items(),
            key=lambda x: x[1].get("roi", 0) * (1 - x[1].get("max_drawdown", 1)),
        )
        model_config["best_strategy"] = best_strat[0]
        model_config["best_strategy_roi"] = best_strat[1].get("roi", 0)
        cb(f"\n🏆 Meilleure stratégie : {best_strat[0]} (ROI: {best_strat[1].get('roi', 0):.2%})")

    config_path_out = models_dir / "model_config_v3.pkl"
    joblib.dump(model_config, config_path_out)
    cb(f"✅ Config sauvegardée : {config_path_out}")

    elapsed = time.time() - t0
    cb(f"\n{'='*60}")
    cb(f"✅ Pipeline terminé en {elapsed:.0f}s")
    cb(f"{'='*60}")

    return {
        "ensemble": ensemble,
        "elo_engine": elo_engine,
        "feature_df": feature_df,
        "model_config": model_config,
        "backtest_report": backtest_report,
    }


# ---------------------------------------------------------------------------
# Prédiction live (pour intégration app.py)
# ---------------------------------------------------------------------------

def predict_match_v3(
    player1: str,
    player2: str,
    surface: str,
    series: str,
    round_name: str,
    best_of: int,
    rank1: float,
    rank2: float,
    pts1: float,
    pts2: float,
    odds_p1: float,
    odds_p2: float,
    match_date: datetime,
    tournament: str,
    ensemble: TennisEnsemble,
    elo_engine: TennisEloEngine,
    recent_history: pd.DataFrame,
    model_config: dict,
    bankroll: float = 1000.0,
) -> dict:
    """
    Prédit le résultat d'un match et retourne les recommandations de paris.
    Compatible avec l'interface app.py existante.
    """
    # Features Elo
    elo_features = elo_engine.get_matchup_features(player1, player2, surface)

    # Feature vector
    builder = FeatureBuilder(feature_cols=model_config["feature_cols"])
    X, features_dict = builder.build_single(
        player1=player1,
        player2=player2,
        surface=surface,
        series=series,
        round_name=round_name,
        best_of=best_of,
        match_date=match_date,
        rank1=rank1,
        rank2=rank2,
        pts1=pts1,
        pts2=pts2,
        elo_features=elo_features,
        recent_history=recent_history,
        tournament_name=tournament,
    )

    # Prédiction
    proba = ensemble.predict_proba(X)
    prob_p1 = float(proba[0, 1])
    prob_p2 = 1.0 - prob_p1

    # Calcul edge & EV
    market_prob_p1 = 1.0 / odds_p1 if odds_p1 > 1 else 0.5
    market_prob_p2 = 1.0 / odds_p2 if odds_p2 > 1 else 0.5
    edge_p1 = prob_p1 - market_prob_p1
    edge_p2 = prob_p2 - market_prob_p2
    ev_p1 = prob_p1 * odds_p1 - 1.0
    ev_p2 = prob_p2 * odds_p2 - 1.0

    # Stratégies
    strat_cfg = model_config.get("strategies", {})
    strategy_manager = StrategyManager.from_config(strat_cfg)

    recommendations = strategy_manager.evaluate_match(
        player1=player1,
        player2=player2,
        model_prob_p1=prob_p1,
        odds_p1=odds_p1,
        odds_p2=odds_p2,
        surface=surface,
        series=series,
        round_name=round_name,
        tournament=tournament,
        bankroll=bankroll,
        match_date=match_date.date() if hasattr(match_date, "date") else match_date,
    )

    return {
        "player1": player1,
        "player2": player2,
        "prob_p1": prob_p1,
        "prob_p2": prob_p2,
        "edge_p1": edge_p1,
        "edge_p2": edge_p2,
        "ev_p1": ev_p1,
        "ev_p2": ev_p2,
        "odds_p1": odds_p1,
        "odds_p2": odds_p2,
        "recommendations": recommendations,
        "features": features_dict,
        "model_version": "v3",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline entraînement tennis v3")
    parser.add_argument("--config", default="config.yaml", help="Chemin vers config.yaml")
    parser.add_argument("--no-backtest", action="store_true", help="Désactiver le backtesting")
    parser.add_argument("--no-wta", action="store_true", help="ATP uniquement")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    args = parser.parse_args()

    results = run_pipeline(
        config_path=args.config,
        run_backtest=not args.no_backtest,
        use_wta=not args.no_wta,
        verbose=not args.quiet,
    )
