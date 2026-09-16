"""Exploratory cross-market study with fixed rules and dated walk-forward fits.

Only Bet365 pre-closing columns enter predictions or selection. These are NOT
verified opening or executable quotes. Reused history cannot become a fresh
holdout by changing the model. No result from this module authorises deployment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

GROUPS = {
    "1x2": ("B365H", "B365D", "B365A"),
    "totals": ("B365>2.5", "B365<2.5"),
    "handicap": ("B365AHH", "B365AHA"),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_frame(frame: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict]:
    """Reject incomplete/malformed quotes, without choosing on realised returns."""
    frame = frame.copy()
    dates = pd.to_datetime(frame["match_date"], errors="raise")
    if frame["match_id"].duplicated().any():
        raise ValueError("Duplicate match IDs; resolve source data before modelling")
    valid = dates.notna().to_numpy(copy=True)
    low, high = protocol["overround_bounds"]
    quality = {"source_rows": len(frame), "source_date_max": str(dates.max().date())}
    for market, columns in GROUPS.items():
        prices = frame[list(columns)].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = (1 / prices).sum(axis=1)
        good = (np.isfinite(prices).all(axis=1) & (prices > 1).all(axis=1)
                & (margin >= low) & (margin <= high))
        quality[f"invalid_{market}_rows"] = int((~good).sum())
        valid &= good
    line = frame["AHh"].to_numpy(dtype=float)
    valid &= (np.isfinite(line) & (np.abs(line) <= protocol["handicap_abs_max"])
              & np.isclose(line * 4, np.round(line * 4)))
    valid &= frame["season_start"].between(
        protocol["first_train_season"], protocol["partial_season_diagnostic"]
    ).to_numpy()
    quality["eligible_rows"] = int(valid.sum())
    quality["excluded_rows"] = int((~valid).sum())
    frame = frame.loc[valid].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    quality["eligible_rows_by_season"] = {
        str(k): int(v) for k, v in frame.groupby("season_start").size().items()
    }
    return frame, quality


def build_inputs(frame: pd.DataFrame, target: str, cross_market: bool):
    """Price-only allowlist: no scores, current match statistics or closing odds."""
    features: dict[str, np.ndarray] = {}
    probabilities = {}
    prices_by_market = {}
    for market, columns in GROUPS.items():
        prices = frame[list(columns)].to_numpy(dtype=float)
        inv = 1 / prices
        probabilities[market] = inv / inv.sum(axis=1, keepdims=True)
        prices_by_market[market] = prices
        if market == target or cross_market:
            p = probabilities[market]
            for j in range(p.shape[1] - 1):
                features[f"{market}_log_ratio_{j}"] = np.log(p[:, j] / p[:, -1])
            features[f"{market}_overround"] = inv.sum(axis=1)
    if cross_market:
        line = frame["AHh"].to_numpy(dtype=float)
        imbalance = np.log(probabilities["1x2"][:, 0] / probabilities["1x2"][:, 2])
        total_logit = np.log(probabilities["totals"][:, 0] / probabilities["totals"][:, 1])
        features.update({
            "handicap_line": line,
            "handicap_line_squared": line ** 2,
            "1x2_imbalance_squared": imbalance ** 2,
            "total_x_imbalance": total_logit * imbalance,
            "total_x_handicap_line": total_logit * line,
        })
    return (np.column_stack(list(features.values())), probabilities[target],
            prices_by_market[target], list(features))


class MarketResidual:
    """Ridge-regularised multinomial residual around the market, zero = market.

    Scaling is fitted on training rows only. The intercept is also regularised;
    the final class is the fixed reference. Penalty is on average log-loss.
    """

    def __init__(self, penalty: float = 0.01):
        self.penalty = penalty

    def _design(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), (x - self.mean_) / self.scale_])

    def fit(self, x: np.ndarray, market: np.ndarray, labels: np.ndarray):
        if not np.isfinite(x).all() or not np.isfinite(market).all():
            raise ValueError("Non-finite training data")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-10] = 1
        design = self._design(x)
        offset = np.log(market)
        n, classes = market.shape
        shape = (design.shape[1], classes - 1)
        truth = np.eye(classes)[labels]

        def objective(flat):
            weights = flat.reshape(shape)
            logits = offset.copy()
            logits[:, :-1] += design @ weights
            logp = logits - logsumexp(logits, axis=1, keepdims=True)
            loss = -logp[np.arange(n), labels].mean() + self.penalty * (weights ** 2).sum() / 2
            gradient = (design.T @ (np.exp(logp) - truth)[:, :-1]) / n + self.penalty * weights
            return loss, gradient.ravel()

        result = minimize(objective, np.zeros(np.prod(shape)), jac=True,
                          method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"Fit did not converge: {result.message}")
        self.weights_ = result.x.reshape(shape)
        return self

    def predict(self, x: np.ndarray, market: np.ndarray) -> np.ndarray:
        logits = np.log(market).copy()
        logits[:, :-1] += self._design(x) @ self.weights_
        return softmax(logits, axis=1)


def training_mask(frame: pd.DataFrame, season: int, embargo_days: int):
    test = frame["season_start"].eq(season).to_numpy()
    if not test.any():
        return np.zeros(len(frame), dtype=bool), test
    cutoff = frame.loc[test, "match_date"].min() - pd.Timedelta(days=embargo_days)
    train = (frame["season_start"].lt(season) & frame["match_date"].lt(cutoff)).to_numpy()
    return train, test


def select_outcomes(probabilities: np.ndarray, market: np.ndarray, odds: np.ndarray,
                    policy: dict) -> np.ndarray:
    effective = 1 + (odds - 1) * (1 - policy["primary_profit_haircut"])
    ev = probabilities * effective - 1
    valid = ((odds >= policy["odds_min"]) & (odds <= policy["odds_max"])
             & (probabilities - market >= policy["minimum_edge_probability"])
             & (ev >= policy["minimum_expected_return_after_haircut"]))
    scores = np.where(valid, ev, -np.inf)
    selected = scores.argmax(axis=1)
    return np.where(valid.any(axis=1), selected, -1)


def walk_forward(frame: pd.DataFrame, protocol: dict,
                 progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, list]:
    predictions, folds = [], []
    for candidate in protocol["candidates"]:
        target = "1x2" if candidate.startswith("1x2") else "totals"
        cross_market = candidate.endswith("cross_market")
        x, market, odds, names = build_inputs(frame, target, cross_market)
        labels = (frame["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
                  if target == "1x2" else (frame["total_goals"].to_numpy() <= 2.5).astype(int))
        for season in protocol["test_seasons"] + [protocol["partial_season_diagnostic"]]:
            train, test = training_mask(frame, season, protocol["embargo_days"])
            if not test.any() or train.sum() < protocol["minimum_training_rows"]:
                continue
            fitted = MarketResidual(protocol["l2_penalty_mean_loss"]).fit(
                x[train], market[train], labels[train])
            probability = fitted.predict(x[test], market[test])
            selected = select_outcomes(probability, market[test], odds[test], protocol["policy"])
            block = frame.loc[test, ["match_id", "match_date", "season_start", "league"]].copy()
            block["candidate"] = candidate
            block["selected_outcome"] = selected
            indices = np.arange(test.sum())
            chosen = np.maximum(selected, 0)
            block["model_loss"] = -np.log(probability[indices, labels[test]])
            block["market_loss"] = -np.log(market[test][indices, labels[test]])
            block["selected_probability"] = np.where(selected >= 0, probability[indices, chosen], np.nan)
            block["selected_market_probability"] = np.where(selected >= 0, market[test][indices, chosen], np.nan)
            block["selected_odds"] = np.where(selected >= 0, odds[test][indices, chosen], np.nan)
            block["won"] = (selected == labels[test]) & (selected >= 0)
            for j in range(probability.shape[1]):
                block[f"p_{j}"] = probability[:, j]
                block[f"market_p_{j}"] = market[test][:, j]
            predictions.append(block)
            folds.append({
                "candidate": candidate, "test_season": season,
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "train_max_date": str(frame.loc[train, "match_date"].max().date()),
                "test_min_date": str(frame.loc[test, "match_date"].min().date()),
                "features": names,
            })
            if progress:
                progress(f"{candidate}: saison {season}, {test.sum():,} matchs, {(selected >= 0).sum()} paris")
    if not predictions:
        raise ValueError("No eligible walk-forward fold")
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(["candidate", "match_id"]).any():
        raise ValueError("Duplicate out-of-sample predictions")
    return result, folds


def net_returns(bets: pd.DataFrame, haircut: float) -> np.ndarray:
    return np.where(bets["won"], (bets["selected_odds"] - 1) * (1 - haircut), -1.0)


def block_interval(bets: pd.DataFrame, all_dates: pd.Series, uncertainty: dict) -> dict:
    if bets.empty:
        return {"ci95": [None, None], "family_adjusted_lower": None, "calendar_months": 0}
    months = pd.period_range(all_dates.min(), all_dates.max(), freq="M")
    indexed = pd.DataFrame({"month": bets["match_date"].dt.to_period("M"),
                            "profit": bets["net_return"]})
    grouped = indexed.groupby("month")["profit"].agg(["sum", "count"]).reindex(months, fill_value=0)
    # Preserve within-month dependence and adjacent-month serial dependence.
    rng = np.random.default_rng(uncertainty["seed"])
    starts = rng.integers(0, len(months), size=(uncertainty["samples"], (len(months) + 2) // 3))
    picks = ((starts[:, :, None] + np.arange(3)) % len(months)).reshape(len(starts), -1)[:, :len(months)]
    counts = grouped["count"].to_numpy()[picks].sum(axis=1)
    sums = grouped["sum"].to_numpy()[picks].sum(axis=1)
    samples = sums[counts > 0] / counts[counts > 0]
    return {
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "family_adjusted_lower": float(np.quantile(samples, uncertainty["lower_quantile_per_candidate"])),
        "calendar_months": len(months), "nonempty_resamples": len(samples),
    }


def summarize_candidate(block: pd.DataFrame, protocol: dict) -> dict:
    bets = block[block["selected_outcome"] >= 0].copy()
    bets["net_return"] = net_returns(bets, protocol["policy"]["primary_profit_haircut"])
    roi = float(bets["net_return"].mean()) if len(bets) else None
    per_season = {}
    for season, rows in block.groupby("season_start"):
        selected = rows[rows["selected_outcome"] >= 0]
        returns = net_returns(selected, protocol["policy"]["primary_profit_haircut"])
        per_season[str(season)] = {
            "matches": len(rows), "bets": len(selected),
            "roi": float(returns.mean()) if len(returns) else None,
            "profit_units": float(returns.sum()),
            "log_loss_gain": float((rows["market_loss"] - rows["model_loss"]).mean()),
        }
    stress = {str(h): float(net_returns(bets, h).mean()) if len(bets) else None
              for h in protocol["policy"]["sensitivity_profit_haircuts"]}
    interval = block_interval(bets, block["match_date"], protocol["uncertainty"])
    best = max(per_season, key=lambda s: per_season[s]["profit_units"])
    others = bets[bets["season_start"].ne(int(best))]
    without_best = float(others["net_return"].mean()) if len(others) else None
    checks = {
        "at_least_500_bets": len(bets) >= 500,
        "roi_at_2pct_positive": roi is not None and roi > 0,
        "roi_at_5pct_positive": stress["0.05"] is not None and stress["0.05"] > 0,
        "family_adjusted_bootstrap_lower_bound_positive": interval["family_adjusted_lower"] is not None
            and interval["family_adjusted_lower"] > 0,
        "at_least_4_of_5_seasons_positive": sum(s["roi"] is not None and s["roi"] > 0
                                               for s in per_season.values()) >= 4,
        "roi_positive_when_best_season_removed": without_best is not None and without_best > 0,
    }
    return {
        "matches": len(block), "bets": len(bets), "roi": roi,
        "profit_units": float(bets["net_return"].sum()),
        "model_log_loss": float(block["model_loss"].mean()),
        "market_log_loss": float(block["market_loss"].mean()),
        "log_loss_gain": float((block["market_loss"] - block["model_loss"]).mean()),
        "roi_same_bets_by_profit_haircut": stress, "uncertainty": interval,
        "seasons": per_season, "best_season_by_profit": best,
        "roi_without_best_season": without_best, "research_checks": checks,
        "candidate_for_further_research": all(checks.values()),
        "real_money_authorised": False,
    }


def run_study(root: Path, progress=print) -> dict:
    directory = root / "models" / "football_cross_market"
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    data_path = root / protocol["data_path"]
    if file_hash(data_path) != protocol["data_sha256"]:
        raise ValueError("Data differs from frozen protocol. Archive it and register a separate study.")
    frame, quality = prepare_frame(pd.read_csv(data_path, low_memory=False), protocol)
    predictions, folds = walk_forward(frame, protocol, progress)
    predictions.to_parquet(directory / "oos_predictions.parquet", index=False)
    blocks = {}
    partial = {}
    for candidate, rows in predictions.groupby("candidate", sort=False):
        primary = rows[rows["season_start"].isin(protocol["test_seasons"])]
        blocks[candidate] = summarize_candidate(primary, protocol)
        recent = rows[rows["season_start"].eq(protocol["partial_season_diagnostic"])]
        if len(recent):
            selected = recent[recent["selected_outcome"] >= 0]
            partial[candidate] = {
                "matches": len(recent), "bets": len(selected),
                "roi": float(net_returns(selected, 0.02).mean()) if len(selected) else None,
                "excluded_from_primary_decision": True,
            }
    viable = [name for name, result in blocks.items() if result["candidate_for_further_research"]]
    report = {
        "study": protocol["study"], "protocol_sha256": file_hash(protocol_path),
        "data_sha256": protocol["data_sha256"], "source_quality": quality,
        "evidence_status": protocol["evidence_status"], "candidates": blocks,
        "partial_season_diagnostic": partial, "folds": folds,
        "status": "EXPLORATORY_CANDIDATE_ONLY" if viable else "NO_ROBUST_CANDIDATE",
        "candidates_for_further_research": viable,
        "real_money_authorised": False, "limitations": protocol["limitations"],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                      allow_nan=False) + "\n")
    lines = [
        "# Football : information croisée entre marchés — 11 septembre 2026", "",
        f"Statut : **{report['status']}**. Aucune autorisation de mise réelle.", "",
        "Historique déjà exploré : ces rendements sont un diagnostic, pas une validation indépendante.",
        "Quatre candidats fixés avant le calcul ; aucun seuil, pays ou saison sélectionné après coup.", "",
        "| Candidat | Matchs | Paris | Gain log-loss | ROI brut | ROI décote gains 2 % | Borne basse corrigée | ROI décote 5 % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def pct(value):
        return "n/a" if value is None else f"{value:+.2%}"
    for name, result in blocks.items():
        stress = result["roi_same_bets_by_profit_haircut"]
        lines.append(f"| {name} | {result['matches']} | {result['bets']} | {result['log_loss_gain']:+.6f} "
                     f"| {pct(stress['0.0'])} | {pct(result['roi'])} "
                     f"| {pct(result['uncertainty']['family_adjusted_lower'])} | {pct(stress['0.05'])} |")
    lines += ["", "La décote porte sur le gain net (cote − 1), pas sur toute la cote décimale.",
              "Bootstrap circulaire par blocs de trois mois ; borne unilatérale 98,75 % par candidat",
              "(correction Bonferroni pour quatre candidats). Cette correction ne couvre pas toutes",
              "les anciennes recherches du projet. Elle ne transforme pas un résultat rétrospectif en preuve.",
              "", "## Méthode et limites", "",
              "Modèles résiduels régularisés autour du prix de leur propre marché. Réentraînement par saison,",
              "avec résultats des saisons antérieures et embargo de sept jours. Les normalisations sont",
              "apprises sur l'entraînement seulement. Aucune cote de clôture n'entre dans les prédictions.",
              "Le handicap est une covariable ; ses prix ne sont pas interprétés comme des probabilités",
              "de victoire ordinaires (remboursements et demi-gains possibles).", "",
              "Une mise unité par sélection, au plus une issue par match et par candidat. Les quatre",
              "candidats sont des alternatives, leurs paris ne sont jamais additionnés en portefeuille.",
              "Les sensibilités à 0 % et 5 % réutilisent exactement les paris choisis à 2 %.", "",
              "Les cinq saisons commençant en 2021–2025 composent le diagnostic principal ; la saison",
              "2026 incomplète figure séparément dans report.json. Les données vont jusqu'au "
              + quality["source_date_max"] + ".", ""]
    lines += [f"- {item}" for item in protocol["limitations"]]
    lines += ["", "## Résultats par saison", "",
              "| Candidat | Saison débutant en | Paris | ROI après décote 2 % |",
              "|---|---:|---:|---:|"]
    for name, result in blocks.items():
        for season, metrics in result["seasons"].items():
            lines.append(f"| {name} | {season} | {metrics['bets']} | {pct(metrics['roi'])} |")
    lines += ["", "Reproduction : `python3 scripts/run_football_cross_market.py`.",
              "Les empreintes des données et du protocole sont vérifiées/enregistrées dans report.json.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
