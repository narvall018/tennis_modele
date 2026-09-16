"""Registered WTA price-residual experiment; repeated history is exploratory."""
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from src.backtesting.football_cross_market import MarketResidual, file_hash
from src.features.elo_system import ROUND_MAP, SURFACES, TennisEloEngine
from src.features.feature_builder import PlayerState


def rebuild_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """All same-day covariates share a prior-day state, regardless of status.

    Work on all rows, including unpriced matches. This replaces inherited
    row-sequential states but never changes prices, labels or the ranking input.
    """
    result = frame.copy()
    result["_date"] = pd.to_datetime(result["_date"])
    result["_round_order"] = result["_round"].map(ROUND_MAP).fillna(3)
    result = result.sort_values(["_date", "_tournament", "_round_order", "_p1"], kind="stable")
    engine = TennisEloEngine()
    states = defaultdict(PlayerState)
    for timestamp, day in result.groupby("_date", sort=True):
        date = timestamp.date()
        names = set(day["_p1"]) | set(day["_p2"])
        snapshots = {}
        for name in names:
            state = deepcopy(engine._get_or_create(name))
            engine._apply_decay(state, date)
            snapshots[name] = state
        values = []
        for row in day.to_dict("records"):
            p1, p2 = row["_p1"], row["_p2"]
            r1, r2 = snapshots[p1], snapshots[p2]
            s1, s2 = states[p1], states[p2]
            surface = row["_surface"] if row["_surface"] in SURFACES else "Hard"
            values.append([r1.global_elo-r2.global_elo,
                           r1.surface_elo[surface]-r2.surface_elo[surface],
                           s1.form(10)-s2.form(10), s1.days_rest(date)-s2.days_rest(date),
                           s1.fatigue(date)-s2.fatigue(date)])
        result.loc[day.index, ["elo_diff", "surface_elo_diff", "form_10_diff", "rest_diff", "fatigue_diff"]] = values
        completed = day[day["_status"].eq("completed")]
        for name in set(completed["_p1"]) | set(completed["_p2"]):
            engine._players[name] = snapshots[name]
        for row in completed.to_dict("records"):
            p1, p2, surface, series = row["_p1"], row["_p2"], row["_surface"], row["_series"]
            winner, loser = (p1, p2) if row["_label"] == 1 else (p2, p1)
            engine._update_pair(engine._players[winner], engine._players[loser],
                                surface, series, row["_round"], date)
            states[p1].update(date, row["_label"] == 1, surface, p2, series)
            states[p2].update(date, row["_label"] == 0, surface, p1, series)
    return result.drop(columns="_round_order").sort_index()


def prepare(frame: pd.DataFrame, protocol: dict):
    frame = frame.copy()
    frame["_date"] = pd.to_datetime(frame["_date"], errors="raise")
    if frame["_source_row_id"].duplicated().any():
        raise ValueError("Duplicate source IDs")
    valid = frame["_date"].notna().to_numpy(copy=True)
    for pair in protocol["required_pairs"]:
        prices = frame[pair].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = (1 / prices).sum(axis=1)
        low, high = protocol["overround_bounds"]
        valid &= (np.isfinite(prices).all(axis=1) & (prices > 1).all(axis=1)
                  & (margin >= low) & (margin <= high))
    valid &= frame["_date"].dt.year.between(protocol["first_train_year"], max(protocol["test_years"])).to_numpy()
    result = frame.loc[valid].sort_values(["_date", "_source_row_id"]).reset_index(drop=True)
    return result, {
        "input_rows": len(frame), "eligible_rows": len(result),
        "eligible_years": {str(y): int(n) for y, n in result.groupby(result["_date"].dt.year).size().items()},
        "input_latest_date": str(frame["_date"].max().date()),
    }


def features(frame: pd.DataFrame, candidate: str, protocol: dict):
    own_odds = frame[["B365_1", "B365_2"]].to_numpy(float)
    sharp_odds = frame[["Pinnacle_1", "Pinnacle_2"]].to_numpy(float)
    own_inv, sharp_inv = 1 / own_odds, 1 / sharp_odds
    own_margin, sharp_margin = own_inv.sum(axis=1), sharp_inv.sum(axis=1)
    q = own_inv / own_margin[:, None]
    sharp = sharp_inv[:, 0] / sharp_margin
    own_logit = logit(q[:, 0])
    delta = logit(sharp) - own_logit
    columns = {
        "B365_logit": own_logit,
        "B365_logit_x_abs_logit": own_logit * np.abs(own_logit),
        "B365_logit_x_overround": own_logit * own_margin,
    }
    if candidate != "book_calibration":
        columns.update({
            "Pinnacle_minus_B365_logit": delta,
            "delta_x_abs_B365_logit": delta * np.abs(own_logit),
            "delta_x_Pinnacle_overround": delta * sharp_margin,
        })
    if candidate == "cross_book_form":
        for column in protocol["lagged_form_features"]:
            columns[column] = np.nan_to_num(frame[column].to_numpy(float), nan=0, posinf=0, neginf=0)
    return np.column_stack(list(columns.values())), q, own_odds, list(columns)


def fit_symmetric(x, market, labels, penalty):
    return MarketResidual(penalty).fit(
        np.vstack([x, -x]), np.vstack([market, market[:, ::-1]]),
        np.concatenate([labels, 1 - labels]))


def predict_symmetric(model, x, market):
    left = (model.predict(x, market)[:, 0] + 1 - model.predict(-x, market[:, ::-1])[:, 0]) / 2
    return np.column_stack([left, 1 - left])


def select(probabilities, odds, rule):
    priced = 1 + (odds - 1) * (1 - rule["profit_haircut"])
    ev = probabilities * priced - 1
    eligible = ((odds >= rule["odds_min"]) & (odds <= rule["odds_max"])
                & (ev >= rule["minimum_ev_after_haircut"]))
    choice = np.where(eligible, ev, -np.inf).argmax(axis=1)
    return np.where(eligible.any(axis=1), choice, -1)


def walk_forward(frame: pd.DataFrame, protocol: dict, progress=None):
    outputs, folds = [], []
    labels = 1 - frame["_label"].to_numpy(int)  # class 0 = P1
    complete = frame["_status"].eq("completed").to_numpy()
    for candidate in protocol["candidates"]:
        x, market, odds, names = features(frame, candidate, protocol)
        for year in protocol["test_years"]:
            cutoff = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=protocol["embargo_days"])
            train = frame["_date"].lt(cutoff).to_numpy() & complete
            test = frame["_date"].dt.year.eq(year).to_numpy()
            if train.sum() < protocol["minimum_training_completed"] or not test.any():
                raise ValueError(f"Insufficient rows for frozen fold {candidate}/{year}")
            model = fit_symmetric(x[train], market[train], labels[train], protocol["l2_penalty_mean_loss"])
            predicted = predict_symmetric(model, x[test], market[test])
            selected = select(predicted, odds[test], protocol["selection"])
            rows = frame.loc[test, ["_source_row_id", "_date", "_status", "_label", "_p1", "_p2", "_surface"]].copy()
            rows["year"] = year
            rows["candidate"] = candidate
            rows["p1"] = predicted[:, 0]
            rows["market_p1"] = market[test, 0]
            rows["model_loss"] = np.where(complete[test], -np.log(predicted[np.arange(test.sum()), labels[test]]), np.nan)
            rows["market_loss"] = np.where(complete[test], -np.log(market[test][np.arange(test.sum()), labels[test]]), np.nan)
            index = np.arange(test.sum())
            safe = np.maximum(selected, 0)
            rows["selected"] = selected
            rows["odds"] = np.where(selected >= 0, odds[test][index, safe], np.nan)
            rows["probability"] = np.where(selected >= 0, predicted[index, safe], np.nan)
            rows["won"] = (selected == labels[test]) & complete[test]
            outputs.append(rows)
            folds.append({"candidate": candidate, "year": year, "train_rows": int(train.sum()),
                          "test_rows": int(test.sum()), "train_max_date": str(frame.loc[train, "_date"].max().date()),
                          "cutoff_exclusive": str(cutoff.date()), "features": names})
            if progress:
                progress(f"{candidate} {year}: {test.sum()} matchs, {(selected >= 0).sum()} paris")
    predictions = pd.concat(outputs, ignore_index=True)
    if predictions.duplicated(["candidate", "_source_row_id"]).any():
        raise ValueError("Duplicate OOS predictions")
    return predictions, folds


def ledger(predictions: pd.DataFrame, staking: str, rule: dict) -> pd.DataFrame:
    bets = predictions.loc[predictions["selected"] >= 0].copy()
    priced = 1 + (bets["odds"].to_numpy() - 1) * (1 - rule["profit_haircut"])
    ev = bets["probability"].to_numpy() * priced - 1
    fraction = (np.full(len(bets), rule["flat_fraction"]) if staking == "flat"
                else np.clip(rule["quarter_kelly_fraction"] * ev / (priced - 1),
                             0, rule["maximum_fraction_per_bet"]))
    bets["fraction"] = fraction
    total = bets.groupby("_date")["fraction"].transform("sum")
    bets["fraction"] *= np.minimum(1, rule["maximum_daily_exposure"] / total)
    bets["settled"] = bets["_status"].eq("completed")
    for haircut in rule["sensitivity_haircuts"]:
        bets[f"return_{haircut}"] = np.where(
            bets["settled"], np.where(bets["won"], (bets["odds"] - 1) * (1 - haircut), -1), 0)
    # All matches on a date are funded together, including later-voided bets.
    bankroll, peak, drawdown = 1000.0, 1000.0, 0.0
    for _, rows in bets.groupby("_date", sort=True):
        stakes = bankroll * rows["fraction"]
        profit = stakes * rows[f"return_{rule['profit_haircut']}"]
        bets.loc[rows.index, "bankroll_before"] = bankroll
        bets.loc[rows.index, "stake_cash"] = stakes
        bets.loc[rows.index, "profit_cash"] = profit
        bankroll += profit.sum()
        peak = max(peak, bankroll)
        drawdown = max(drawdown, 1 - bankroll / peak)
    bets.attrs.update(final_bankroll=bankroll, maximum_drawdown=drawdown)
    return bets


def uncertainty(bets, dates, settings):
    if bets.empty or not bets["settled"].any():
        return {"ci95": [None, None], "family_lower": None}
    months = pd.period_range(dates.min(), dates.max(), freq="M")
    settled = bets[bets["settled"]]
    values = pd.DataFrame({"month": settled["_date"].dt.to_period("M"),
                           "profit": settled["fraction"] * settled["return_0.02"],
                           "exposure": settled["fraction"]})
    sums = values.groupby("month")[["profit", "exposure"]].sum().reindex(months, fill_value=0)
    rng = np.random.default_rng(settings["seed"])
    starts = rng.integers(len(months), size=(settings["samples"], (len(months) + 2) // 3))
    indices = ((starts[:, :, None] + np.arange(3)) % len(months)).reshape(len(starts), -1)[:, :len(months)]
    denominator = sums["exposure"].to_numpy()[indices].sum(axis=1)
    numerator = sums["profit"].to_numpy()[indices].sum(axis=1)
    draws = numerator[denominator > 0] / denominator[denominator > 0]
    return {"ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "family_lower": float(np.quantile(draws, settings["lower_quantile_per_candidate"])),
            "months": len(months)}


def roi(bets, haircut):
    settled = bets[bets["settled"]]
    exposure = settled["fraction"].sum()
    return float((settled["fraction"] * settled[f"return_{haircut}"]).sum() / exposure) if exposure > 0 else None


def evaluate(predictions, staking, protocol):
    bets = ledger(predictions, staking, protocol["selection"])
    ci = uncertainty(bets, predictions["_date"], protocol["uncertainty"])
    yearly = {}
    for year in protocol["test_years"]:
        subset = bets[bets["year"] == year]
        yearly[str(year)] = {"bets": len(subset), "settled": int(subset["settled"].sum()),
                             "roi": roi(subset, 0.02),
                             "profit_fraction": float((subset["fraction"] * subset["return_0.02"]).sum())}
    best = max(yearly, key=lambda y: yearly[y]["profit_fraction"])
    without_best = roi(bets[bets["year"] != int(best)], 0.02)
    stressed = {str(h): roi(bets, h) for h in protocol["selection"]["sensitivity_haircuts"]}
    positive = sum(y["roi"] is not None and y["roi"] > 0 for y in yearly.values())
    checks = {
        "500_settled_bets": int(bets["settled"].sum()) >= 500,
        "roi_positive_2pct": stressed["0.02"] is not None and stressed["0.02"] > 0,
        "roi_positive_5pct": stressed["0.05"] is not None and stressed["0.05"] > 0,
        "family_adjusted_lower_positive": ci["family_lower"] is not None and ci["family_lower"] > 0,
        "at_least_6_of_9_years_positive": positive >= 6,
        "roi_without_best_profit_year_positive": without_best is not None and without_best > 0,
    }
    return {
        "matches": len(predictions), "bets": len(bets), "settled": int(bets["settled"].sum()),
        "void": int((~bets["settled"]).sum()), "roi": stressed["0.02"],
        "weighted_roi_by_haircut": stressed, "uncertainty": ci, "yearly": yearly,
        "positive_years": positive, "best_profit_year": best, "roi_without_best_year": without_best,
        "model_log_loss": float(predictions["model_loss"].mean()),
        "market_log_loss": float(predictions["market_loss"].mean()),
        "bankroll": bets.attrs, "checks": checks, "research_candidate": all(checks.values()),
        "real_money_authorised": False,
    }, bets


def run(root: Path, progress=print, strict_daily: bool = True):
    folder = root / "models/wta_price_residual"
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    path = root / protocol["data_path"]
    if file_hash(path) != protocol["data_sha256"]:
        raise ValueError("Input changed relative to the registered experiment")
    raw = pd.read_csv(path, low_memory=False)
    daily_quality = None
    if strict_daily:
        folder = folder / "daily_audit"
        amendment_path = folder / "amendment.json"
        if not amendment_path.exists():
            raise ValueError("Daily audit must be registered before execution")
        if progress:
            progress("Rebuilding strict prior-day features on every historical match")
        rebuilt = rebuild_daily_features(raw)
        columns = ["elo_diff", "surface_elo_diff", "form_10_diff", "rest_diff", "fatigue_diff"]
        changed = ~np.isclose(raw[columns].to_numpy(float), rebuilt[columns].to_numpy(float), equal_nan=True)
        daily_quality = {"rows_changed": int(changed.any(axis=1).sum()),
                         "rows_changed_by_feature": dict(zip(columns, changed.sum(axis=0).astype(int).tolist())),
                         "amendment_sha256": file_hash(amendment_path)}
        raw = rebuilt
    frame, quality = prepare(raw, protocol)
    predictions, folds = walk_forward(frame, protocol, progress)
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    reports, ledgers = {}, []
    for candidate, rows in predictions.groupby("candidate", sort=False):
        for staking in protocol["stakes"]:
            name = candidate + "/" + staking
            reports[name], bets = evaluate(rows, staking, protocol)
            bets["strategy"] = name
            bets.attrs = {}
            ledgers.append(bets)
            if progress:
                progress(f"{name}: n={reports[name]['settled']}, ROI={reports[name]['roi']}")
    pd.concat(ledgers, ignore_index=True).to_parquet(folder / "bets.parquet", index=False)
    report = {"study": protocol["study"], "data_sha256": file_hash(path),
              "protocol_sha256": file_hash(protocol_path), "implementation_sha256": file_hash(Path(__file__)),
              "evidence_status": protocol["evidence_status"], "quality": quality, "folds": folds,
              "strict_prior_day_features": strict_daily, "daily_feature_audit": daily_quality,
              "strategies": reports, "real_money_authorised": False,
              "status": "HYPOTHESIS_REQUIRES_PROSPECTIVE_REPLICATION" if any(r["research_candidate"] for r in reports.values())
                        else "NO_ROBUST_CANDIDATE"}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    def percent(value):
        return "n/a" if value is None else f"{value:+.2%}"
    lines = ["# WTA : désaccord entre bookmakers et pondération des mises", "",
             f"Statut : **{report['status']}**. Aucun argent réel autorisé.", "",
             "Historique déjà exploré ; ce test est un diagnostic chronologique, pas une validation indépendante.", "",
             "| Stratégie | Réglés | Annulés | ROI brut | ROI décote 2 % | ROI décote 5 % | Borne basse corrigée | Années positives |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in reports.items():
        s = result["weighted_roi_by_haircut"]
        lines.append(f"| {name} | {result['settled']} | {result['void']} | {percent(s['0.0'])} | {percent(s['0.02'])} "
                     f"| {percent(s['0.05'])} | {percent(result['uncertainty']['family_lower'])} | {result['positive_years']}/9 |")
    lines += ["", "Les ROI sont pondérés par les fractions de bankroll fixées avant chaque résultat ;",
              "le scénario de bankroll composé figure séparément dans report.json. Les six variantes",
              "sont alternatives et ne doivent pas être additionnées. Abandons et walkovers sont",
              "conservés lors de la sélection puis annulés au règlement dans le scénario déclaré.", "",
              "La décote porte sur (cote − 1). Les mêmes paris et fractions de mise sont conservés",
              "pour les trois décotes. Bootstrap de blocs de trois mois, correction pour six variantes",
              "seulement : elle ne couvre pas la multiplicité de toutes les recherches passées.", "",
              "L'entraînement utilise les matchs terminés avant le 1er janvier moins sept jours.",
              "Les transformations apprises utilisent seulement l'entraînement ; l'inversion des",
              "joueuses inverse les probabilités. Aucun filtre ne sélectionne sur le statut final.", "",
              "**Limites d'exécution** : prix historiques Bet365 et Pinnacle sans horodatage précis ;",
              "leur simultanéité et leur disponibilité en France ne sont pas prouvées. Le test ne",
              "contient aucune promotion, aucun bonus et aucune martingale.", "",
              "Le règlement des abandons est une hypothèse, pas une reproduction vérifiée des",
              "règles historiques d'un opérateur. Les dates ne prouvent pas les heures de fin des",
              "matchs reportés. Le financement quotidien suppose aussi que les prix et le budget",
              "soient disponibles ensemble ; faute d'horodatages, ce point n'est pas démontré.", ""]
    if strict_daily:
        lines += ["## Correction temporelle obligatoire", "",
                  "Ce résultat remplace celui basé sur les variables héritées. Les Elo, forme, repos",
                  "et fatigue sont reconstruits avec les jours strictement antérieurs. Le statut final",
                  "du match ne décide plus si son rating pré-match doit subir une décote temporelle.",
                  f"L'audit a modifié au moins une variable dans {daily_quality['rows_changed']} lignes.",
                  "Aucun choix n'a été fait entre les rendements des deux versions.", "",
                  "Reproduire : `python3 scripts/run_wta_price_residual.py --strict-daily`.", ""]
    else:
        lines[2:2] = ["**Version héritée remplacée après audit temporel.** Résultats conservés pour",
                      "traçabilité uniquement ; consulter le [recalcul strict](daily_audit/REPORT.md).", ""]
        lines += ["Reproduction de la version obsolète uniquement :",
                  "`python3 scripts/run_wta_price_residual.py --legacy-inherited-features`.", ""]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
