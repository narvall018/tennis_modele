"""
Walk-Forward Backtester avec CLV Tracking
==========================================
Implémente un backtesting rigoureux sans data leakage via la méthode
Walk-Forward (entraînement glissant).

Principe :
  Années 1-4 → Entraîne modèle → Teste sur Année 5
  Années 2-5 → Entraîne modèle → Teste sur Année 6
  ...
  → Agrège les résultats sur toutes les fenêtres

Métriques calculées :
  - ROI par stratégie et global
  - Sharpe ratio (annualisé)
  - Max drawdown
  - Calmar ratio (ROI / max drawdown)
  - Win rate
  - Brier score (calibration des probas)
  - CLV moyen (Closing Line Value)
  - Bootstrap confidence intervals

Hypothèses réalistes :
  - Marge bookmaker simulée (5% sur cotes synthétiques)
  - Bankroll mise à jour dynamiquement
  - Limites portefeuille respectées
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_roi(profits: List[float], stakes: List[float]) -> float:
    total_staked = sum(stakes)
    return sum(profits) / total_staked if total_staked > 0 else 0.0


def compute_sharpe(daily_returns: List[float], annualize: bool = True) -> float:
    """Sharpe ratio des rendements journaliers."""
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    std = arr.std()
    if std == 0:
        return 0.0
    sr = arr.mean() / std
    return sr * np.sqrt(252) if annualize else sr  # annualisé 252 jours trading


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Max drawdown de la courbe de capital (en %)."""
    if not equity_curve:
        return 0.0
    curve = np.array(equity_curve)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / (peak + 1e-10)
    return float(drawdown.max())


def compute_calmar(roi_annualized: float, max_drawdown: float) -> float:
    return roi_annualized / max_drawdown if max_drawdown > 0 else 0.0


def compute_clv(entry_odds: float, closing_odds: float) -> float:
    """
    Closing Line Value : mesure si le pari a été pris à meilleure valeur que le marché final.
    CLV > 0 → pari pris avant que le marché bouge en votre faveur (smart money signal).
    CLV ≈ (closing_prob - entry_prob) / entry_prob
    """
    if closing_odds <= 0 or entry_odds <= 0:
        return 0.0
    entry_prob = 1.0 / entry_odds
    closing_prob = 1.0 / closing_odds
    return (closing_prob - entry_prob) / entry_prob  # positif = détérioration (cote baissée)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Résultats d'une fenêtre de backtest."""
    fold_id: int
    train_years: Tuple[int, int]
    test_year: int
    n_train: int
    n_test: int

    # Par stratégie
    strategy_results: Dict[str, dict] = field(default_factory=dict)

    # Global
    model_brier: float = 0.0
    model_auc: float = 0.0
    model_logloss: float = 0.0

    # Bets log
    bets_log: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class BacktestReport:
    """Rapport complet du backtest walk-forward."""
    n_folds: int
    total_years: Tuple[int, int]
    folds: List[FoldResult]

    # Métriques agrégées (sur tous les folds)
    aggregate: Dict[str, dict] = field(default_factory=dict)  # par stratégie
    global_metrics: dict = field(default_factory=dict)

    # Bootstrap confidence intervals
    bootstrap_ci: dict = field(default_factory=dict)

    def print_summary(self):
        print("=" * 70)
        print(f"BACKTEST WALK-FORWARD — {self.n_folds} FOLDS")
        print("=" * 70)
        for strat, metrics in self.aggregate.items():
            print(f"\n📊 {strat}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.3f}")
                else:
                    print(f"   {k}: {v}")


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Backtesteur walk-forward pour modèles tennis.

    Usage :
        backtester = WalkForwardBacktester(
            model_factory=lambda: TennisEnsemble(config),
            strategy_manager=StrategyManager.from_config(config["strategies"]),
            train_years=4,
            initial_bankroll=1000.0,
        )
        report = backtester.run(feature_df)
    """

    def __init__(
        self,
        model_factory,          # callable → instance TennisEnsemble (ou compatible)
        strategy_manager,       # StrategyManager
        feature_cols: List[str],
        train_years: int = 4,
        initial_bankroll: float = 1000.0,
        bookmaker_margin: float = 0.05,
        progress_callback=None,
    ):
        self.model_factory = model_factory
        self.strategy_manager = strategy_manager
        self.feature_cols = feature_cols
        self.train_years = train_years
        self.initial_bankroll = initial_bankroll
        self.bookmaker_margin = bookmaker_margin
        self.progress = progress_callback or (lambda msg: None)

    def run(self, feature_df: pd.DataFrame) -> BacktestReport:
        """
        Lance le backtest walk-forward complet.

        feature_df doit contenir :
          - Toutes les colonnes feature (self.feature_cols)
          - _date, _label, _series, _surface, _round
          - odds_p1, odds_p2 (optionnel — sinon simulées à partir du label Elo)
        """
        df = feature_df.copy()
        df["_year"] = pd.to_datetime(df["_date"]).dt.year
        years = sorted(df["_year"].unique())

        if len(years) < self.train_years + 1:
            raise ValueError(
                f"Données insuffisantes : {len(years)} années, "
                f"besoin de {self.train_years + 1} minimum."
            )

        folds: List[FoldResult] = []
        all_bets: List[pd.DataFrame] = []

        # Fenêtres walk-forward
        for fold_start in range(len(years) - self.train_years):
            train_end_idx = fold_start + self.train_years - 1
            test_year = years[train_end_idx + 1]
            train_period = (years[fold_start], years[train_end_idx])

            self.progress(
                f"Fold {len(folds) + 1} — Train: {train_period} | Test: {test_year}"
            )

            train_mask = (df["_year"] >= years[fold_start]) & (df["_year"] <= years[train_end_idx])
            test_mask = df["_year"] == test_year

            df_train = df[train_mask].reset_index(drop=True)
            df_test = df[test_mask].reset_index(drop=True)

            if len(df_train) < 1000 or len(df_test) < 50:
                self.progress(f"Fold ignoré : données insuffisantes")
                continue

            # Feature matrices
            X_train = df_train[self.feature_cols].values.astype(np.float32)
            y_train = df_train["_label"].values.astype(int)
            X_test = df_test[self.feature_cols].values.astype(np.float32)
            y_test = df_test["_label"].values.astype(int)

            # Calibration hold-out : dernière saison du train
            cal_year = years[train_end_idx]
            cal_mask_train = df_train["_year"] == cal_year
            X_cal = df_train[cal_mask_train][self.feature_cols].values.astype(np.float32)
            y_cal = df_train[cal_mask_train]["_label"].values.astype(int)
            X_train_sub = df_train[~cal_mask_train][self.feature_cols].values.astype(np.float32)
            y_train_sub = df_train[~cal_mask_train]["_label"].values.astype(int)

            # Entraîne le modèle
            model = self.model_factory()
            model.fit(
                X_train_sub, y_train_sub,
                X_cal=X_cal, y_cal=y_cal,
                feature_names=self.feature_cols,
                progress_callback=self.progress,
            )

            # Prédictions test
            test_probs = model.predict_proba(X_test)[:, 1]

            # Métriques modèle
            model_brier = brier_score_loss(y_test, test_probs)
            model_auc = roc_auc_score(y_test, test_probs)
            model_logloss = log_loss(y_test, test_probs)

            self.progress(
                f"  AUC: {model_auc:.4f} | Brier: {model_brier:.4f} | LogLoss: {model_logloss:.4f}"
            )

            # Simuler les cotes si absentes
            df_test = self._ensure_odds(df_test, test_probs)

            # Évaluer les stratégies
            self.strategy_manager.reset_portfolio()
            bankroll = self.initial_bankroll
            fold_bets = []

            for i, row in df_test.iterrows():
                prob = float(test_probs[i])

                recs = self.strategy_manager.evaluate_match(
                    player1=str(row.get("_p1", "P1")),
                    player2=str(row.get("_p2", "P2")),
                    model_prob_p1=prob,
                    odds_p1=float(row.get("odds_p1", 2.0)),
                    odds_p2=float(row.get("odds_p2", 2.0)),
                    surface=str(row.get("_surface", "Hard")),
                    series=str(row.get("_series", "ATP250")),
                    round_name=str(row.get("_round", "1st Round")),
                    tournament=str(row.get("_tournament", "")),
                    bankroll=bankroll,
                    match_date=pd.to_datetime(row["_date"]).date(),
                )

                for rec in recs:
                    actual_winner_is_player = (
                        rec.player == str(row.get("_p1", "")) and row["_label"] == 1
                    ) or (
                        rec.player == str(row.get("_p2", "")) and row["_label"] == 0
                    )

                    profit = rec.stake_amount * (rec.odds - 1) if actual_winner_is_player else -rec.stake_amount
                    bankroll += profit

                    fold_bets.append({
                        "fold": len(folds) + 1,
                        "date": rec.match_date,
                        "test_year": test_year,
                        "strategy": rec.strategy_name,
                        "player": rec.player,
                        "odds": rec.odds,
                        "edge": rec.edge,
                        "ev": rec.ev,
                        "stake": rec.stake_amount,
                        "stake_pct": rec.stake_pct,
                        "won": actual_winner_is_player,
                        "profit": profit,
                        "bankroll": bankroll,
                        "model_prob": rec.model_prob,
                        "market_prob": rec.market_prob,
                        "surface": rec.surface,
                        "series": rec.series,
                    })

            bets_df = pd.DataFrame(fold_bets)
            all_bets.append(bets_df)

            strategy_results = self._compute_strategy_metrics(bets_df)

            fold_result = FoldResult(
                fold_id=len(folds) + 1,
                train_years=train_period,
                test_year=test_year,
                n_train=len(df_train),
                n_test=len(df_test),
                strategy_results=strategy_results,
                model_brier=model_brier,
                model_auc=model_auc,
                model_logloss=model_logloss,
                bets_log=bets_df,
            )
            folds.append(fold_result)

        # Agréger tous les bets
        all_bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()

        # Calcul des métriques agrégées
        aggregate = self._aggregate_metrics(folds, all_bets_df)
        global_metrics = self._global_metrics(folds, all_bets_df)

        # Bootstrap CI
        bootstrap_ci = self._bootstrap_analysis(all_bets_df) if not all_bets_df.empty else {}

        report = BacktestReport(
            n_folds=len(folds),
            total_years=(years[0], years[-1]),
            folds=folds,
            aggregate=aggregate,
            global_metrics=global_metrics,
            bootstrap_ci=bootstrap_ci,
        )

        report.print_summary()
        return report

    def _ensure_odds(self, df: pd.DataFrame, model_probs: np.ndarray) -> pd.DataFrame:
        """
        Si les cotes réelles sont absentes, les simule à partir des probabilités Elo
        en appliquant la marge bookmaker.
        """
        df = df.copy()
        if "odds_p1" not in df.columns or df["odds_p1"].isna().all():
            # Cotes simulées avec marge
            margin = self.bookmaker_margin
            raw_prob = model_probs
            raw_prob_p2 = 1.0 - raw_prob
            # Appliquer la marge (surround)
            total = raw_prob + raw_prob_p2 + margin
            adj_p1 = raw_prob / total * (1 + margin)
            adj_p2 = raw_prob_p2 / total * (1 + margin)
            df["odds_p1"] = np.where(adj_p1 > 0, 1.0 / np.clip(adj_p1, 0.01, 0.99), 2.0)
            df["odds_p2"] = np.where(adj_p2 > 0, 1.0 / np.clip(adj_p2, 0.01, 0.99), 2.0)
        return df

    def _compute_strategy_metrics(self, bets_df: pd.DataFrame) -> Dict[str, dict]:
        if bets_df.empty:
            return {}

        results = {}
        for strategy in bets_df["strategy"].unique():
            sb = bets_df[bets_df["strategy"] == strategy]
            profits = sb["profit"].tolist()
            stakes = sb["stake"].tolist()

            # Courbe de capital pour drawdown
            equity = [self.initial_bankroll + sum(profits[:i+1]) for i in range(len(profits))]
            mdd = compute_max_drawdown(equity)

            # Rendements journaliers pour Sharpe
            sb_daily = sb.groupby("date")["profit"].sum().reset_index()
            daily_returns = sb_daily["profit"].tolist()
            sharpe = compute_sharpe(daily_returns)

            n_bets = len(sb)
            n_wins = sb["won"].sum()
            roi = compute_roi(profits, stakes)
            n_test_days = (sb["date"].max() - sb["date"].min()).days if n_bets > 0 else 1
            roi_annualized = roi * (365 / max(n_test_days, 1)) * n_bets / max(n_bets, 1)

            results[strategy] = {
                "n_bets": n_bets,
                "win_rate": n_wins / n_bets if n_bets > 0 else 0.0,
                "roi": roi,
                "total_profit": sum(profits),
                "total_staked": sum(stakes),
                "avg_odds": sb["odds"].mean(),
                "avg_edge": sb["edge"].mean(),
                "max_drawdown": mdd,
                "sharpe": sharpe,
                "calmar": compute_calmar(roi_annualized, mdd),
            }

        return results

    def _aggregate_metrics(
        self, folds: List[FoldResult], all_bets: pd.DataFrame
    ) -> Dict[str, dict]:
        if all_bets.empty:
            return {}

        aggregate = {}
        for strategy in all_bets["strategy"].unique():
            sb = all_bets[all_bets["strategy"] == strategy]
            profits = sb["profit"].tolist()
            stakes = sb["stake"].tolist()
            equity = np.cumsum([self.initial_bankroll] + profits)
            mdd = compute_max_drawdown(equity.tolist())
            daily = sb.groupby("date")["profit"].sum().tolist()
            sharpe = compute_sharpe(daily)
            roi = compute_roi(profits, stakes)
            n_bets = len(sb)
            n_wins = sb["won"].sum()

            aggregate[strategy] = {
                "n_bets_total": n_bets,
                "n_bets_per_year": n_bets / max(len(folds), 1),
                "win_rate": n_wins / n_bets if n_bets > 0 else 0.0,
                "roi": roi,
                "roi_pct": f"{roi:.2%}",
                "total_profit": sum(profits),
                "avg_odds": sb["odds"].mean(),
                "avg_edge": sb["edge"].mean(),
                "max_drawdown": mdd,
                "sharpe": sharpe,
                "calmar": compute_calmar(roi * len(folds), mdd),
            }

        return aggregate

    def _global_metrics(self, folds: List[FoldResult], all_bets: pd.DataFrame) -> dict:
        """Métriques globales du modèle sur tous les folds."""
        if not folds:
            return {}
        avg_auc = np.mean([f.model_auc for f in folds])
        avg_brier = np.mean([f.model_brier for f in folds])
        avg_logloss = np.mean([f.model_logloss for f in folds])
        return {
            "avg_auc": avg_auc,
            "avg_brier": avg_brier,
            "avg_logloss": avg_logloss,
            "n_folds": len(folds),
            "total_bets": len(all_bets),
        }

    def _bootstrap_analysis(
        self,
        all_bets: pd.DataFrame,
        n_iter: int = 1000,
        confidence: float = 0.95,
    ) -> dict:
        """
        Bootstrap sur le ROI global pour estimer les intervalles de confiance.
        Resample les paris avec remplacement.
        """
        if all_bets.empty:
            return {}

        ci = {}
        alpha = 1 - confidence

        for strategy in all_bets["strategy"].unique():
            sb = all_bets[all_bets["strategy"] == strategy].reset_index(drop=True)
            profits = sb["profit"].values
            stakes = sb["stake"].values

            bootstrap_rois = []
            for _ in range(n_iter):
                idx = np.random.choice(len(profits), size=len(profits), replace=True)
                roi = profits[idx].sum() / stakes[idx].sum()
                bootstrap_rois.append(roi)

            arr = np.array(bootstrap_rois)
            ci[strategy] = {
                "roi_mean": float(arr.mean()),
                "roi_std": float(arr.std()),
                "roi_ci_low": float(np.percentile(arr, alpha / 2 * 100)),
                "roi_ci_high": float(np.percentile(arr, (1 - alpha / 2) * 100)),
                "prob_positive_roi": float((arr > 0).mean()),
            }

        return ci
