#!/usr/bin/env python3
"""Compare next-day ERCOT RTM residual models and a 2023 county-risk ablation."""

import argparse
import datetime as dt
import gzip
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "price"
MODEL_DIR = DATA / "models"
UTC = dt.timezone.utc
SEED = 43
FEATURES = ["dam_asinh", "local_hour", "day_of_year", "zone_code", "weekday", "quarter", "repeated"]
CATS = ["zone_code", "weekday", "quarter", "repeated"]
QUANTILES = (0.1, 0.5, 0.9)
CHANGE = pd.Timestamp("2025-12-05")


def load_prices():
    path = DATA / "price_rows_2023_2025.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Run build_price_model_data.py first: {path}")
    frame = pd.read_csv(path, parse_dates=["model_issue_utc", "input_cutoff_utc",
                                          "delivery_date", "interval_start_utc", "interval_end_utc"])
    expected = sum(x["matched_rows"] for x in json.loads((DATA / "source_audit.json").read_text())["years"].values())
    if len(frame) != expected:
        raise ValueError(f"Price row count {len(frame)} differs from source audit {expected}")
    if frame[["settlement_point", "interval_start_utc"]].duplicated().any():
        raise ValueError("Duplicate load-zone UTC interval")
    if not (frame.model_issue_utc < frame.interval_start_utc).all():
        raise ValueError("Issue time must precede target")
    return frame.sort_values(["interval_start_utc", "settlement_point"]).reset_index(drop=True)


def matrix(frame, zones):
    zone_codes = frame.settlement_point.map(zones)
    if zone_codes.isna().any():
        raise ValueError("Unknown settlement point")
    x = pd.DataFrame({
        "dam_asinh": np.arcsinh(frame.dam_spp_usd_mwh.to_numpy(dtype=np.float32) / 50),
        "local_hour": (frame.hour_ending.to_numpy(dtype=np.float32) - 1
                       + (frame.quarter.to_numpy(dtype=np.float32) - 0.5) / 4),
        "day_of_year": frame.delivery_date.dt.dayofyear.to_numpy(dtype=np.float32),
        "zone_code": zone_codes.to_numpy(dtype=np.int8),
        "weekday": frame.delivery_date.dt.dayofweek.to_numpy(dtype=np.int8),
        "quarter": frame.quarter.to_numpy(dtype=np.int8),
        "repeated": (frame.repeated_hour_flag == "Y").to_numpy(dtype=np.int8),
    }, index=frame.index)
    if "county_onset_risk_mean" in frame:
        x["county_onset_risk_mean"] = frame.county_onset_risk_mean.to_numpy(dtype=np.float32)
    return x


def spline_transformer():
    # One smooth term per continuous input, plus additive categorical effects.
    return ColumnTransformer([
        ("dam", SplineTransformer(n_knots=7, knots="quantile", include_bias=False,
                                   sparse_output=True), ["dam_asinh"]),
        ("hour", SplineTransformer(n_knots=7, knots="quantile", include_bias=False,
                                    sparse_output=True), ["local_hour"]),
        ("doy", SplineTransformer(n_knots=7, knots="quantile", include_bias=False,
                                   sparse_output=True), ["day_of_year"]),
        ("factor", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATS),
    ], sparse_threshold=1.0)


def spline_penalty(transformer):
    names = transformer.get_feature_names_out()
    penalty = np.eye(len(names)) * 1e-5
    for prefix in ("dam__", "hour__", "doy__"):
        block = [i for i, name in enumerate(names) if name.startswith(prefix)]
        for left, middle, right in zip(block, block[1:], block[2:]):
            indices = [left, middle, right]
            difference = np.array([1., -2., 1.])
            penalty[np.ix_(indices, indices)] += 1e-3 * np.outer(difference, difference)
    return penalty


def fit_spline_loss(design, y, penalty, quantile=None, initial=None):
    """Penalized additive spline fit; quantiles minimize smoothed pinball loss."""
    y = np.asarray(y, dtype=float)
    n_features = design.shape[1]
    if quantile is None:
        x_mean = np.asarray(design.mean(axis=0)).ravel()
        gram = (design.T @ design).toarray() / len(y) - np.outer(x_mean, x_mean)
        rhs = np.asarray(design.T @ y).ravel() / len(y) - x_mean * y.mean()
        coefficients = np.linalg.solve(gram + 2 * penalty, rhs)
        intercept = y.mean() - x_mean @ coefficients
        fitted = np.asarray(design @ coefficients).ravel() + intercept
        objective = 0.5 * np.mean((y - fitted) ** 2) + coefficients @ penalty @ coefficients
        return np.r_[coefficients, intercept], {"iterations": 0, "objective": float(objective),
                                                 "loss": "squared", "pinball_smoothing_usd_mwh": None}
    initial = np.zeros(n_features + 1) if initial is None else initial.copy()
    initial[-1] += np.quantile(y - np.asarray(design @ initial[:n_features]).ravel() - initial[-1], quantile)
    epsilon = 0.10  # USD/MWh, a small smoothing band around the pinball kink

    def objective(coef):
        fitted = design @ coef[:n_features] + coef[-1]
        error = y - fitted
        if quantile is None:
            loss = 0.5 * np.mean(error ** 2)
            slope = error
        else:
            q = quantile
            high = error >= epsilon
            low = error <= -epsilon
            middle = ~(high | low)
            values = np.empty_like(error)
            values[high] = q * error[high]
            values[low] = (q - 1) * error[low]
            values[middle] = ((q - 0.5) * error[middle]
                              + error[middle] ** 2 / (4 * epsilon) + epsilon / 4)
            loss = values.mean()
            slope = np.empty_like(error)
            slope[high] = q
            slope[low] = q - 1
            slope[middle] = q - 0.5 + error[middle] / (2 * epsilon)
        regularization = penalty @ coef[:n_features]
        gradient = np.r_[-np.asarray(design.T @ slope).ravel() / len(y) + 2 * regularization,
                         -slope.mean()]
        return loss + float(coef[:n_features] @ regularization), gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-8, "gtol": 1e-4})
    if not result.success:
        raise RuntimeError(f"Additive spline fit failed for q={quantile}: {result.message}")
    return result.x, {"iterations": int(result.nit), "objective": float(result.fun),
                      "loss": "squared" if quantile is None else f"smoothed pinball q={quantile}",
                      "pinball_smoothing_usd_mwh": None if quantile is None else epsilon}


def fit_family(family, x_train, y_train, x_val, y_val, max_gam_rows):
    models = {}
    started = time.monotonic()
    if family == "lightgbm":
        train = x_train.copy()
        val = x_val.copy()
        for col in CATS:
            train[col] = train[col].astype("category")
            val[col] = val[col].astype("category")
        for name, q in [("mean", None), *((f"p{int(p * 100)}", p) for p in QUANTILES)]:
            params = {"n_estimators": 450, "learning_rate": 0.045, "num_leaves": 23,
                      "min_child_samples": 100, "reg_lambda": 2., "verbosity": -1,
                      "n_jobs": 4, "random_state": SEED}
            params.update({"objective": "regression"} if q is None else {"objective": "quantile", "alpha": q})
            model = LGBMRegressor(**params)
            model.fit(train, y_train, eval_set=[(val, y_val)],
                      callbacks=[early_stopping(35, verbose=False)])
            models[name] = model
    elif family == "catboost":
        for name, q in (("mean", None), ("quantiles", True)):
            model = CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=4,
                loss_function="RMSE" if q is None else "MultiQuantile:alpha=0.1,0.5,0.9",
                random_seed=SEED, thread_count=4, allow_writing_files=False, verbose=False)
            model.fit(x_train, y_train, cat_features=CATS, eval_set=(x_val, y_val),
                      early_stopping_rounds=40)
            models[name] = model
    elif family == "quantile_gam":
        if "county_onset_risk_mean" in x_train:
            raise ValueError("GAM is used in the main family comparison only")
        if len(x_train) > max_gam_rows:
            raise ValueError("All family fits must use the same capped training rows")
        transformer = spline_transformer()
        design = transformer.fit_transform(x_train)
        penalty = spline_penalty(transformer)
        models["transformer"] = transformer
        models["optimizer"] = {}
        for name, q in [("mean", None), *((f"p{int(p * 100)}", p) for p in QUANTILES)]:
            models[name], models["optimizer"][name] = fit_spline_loss(
                design, y_train, penalty, q, models.get("mean"))
    else:
        raise ValueError(family)
    return models, round(time.monotonic() - started, 2)


def forecast(family, models, frame, zones):
    x = matrix(frame, zones)
    offset = frame.dam_spp_usd_mwh.to_numpy(dtype=float)
    if family == "lightgbm":
        for col in CATS:
            x[col] = x[col].astype("category")
        raw = {name: offset + model.predict(x) for name, model in models.items()}
    elif family == "catboost":
        raw = {"mean": offset + models["mean"].predict(x)}
        q = np.asarray(models["quantiles"].predict(x))
        if q.shape != (len(x), 3):
            raise ValueError(f"Unexpected CatBoost MultiQuantile shape: {q.shape}")
        raw.update({"p10": offset + q[:, 0], "p50": offset + q[:, 1], "p90": offset + q[:, 2]})
    else:
        design = models["transformer"].transform(x)
        raw = {name: offset + np.asarray(design @ models[name][:-1]).ravel() + models[name][-1]
               for name in ("mean", "p10", "p50", "p90")}
    values = np.column_stack((raw["p10"], raw["p50"], raw["p90"]))
    crossed = int(np.count_nonzero((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])))
    values.sort(axis=1)  # enforce the output contract without changing the mean
    return {"mean": np.asarray(raw["mean"]), "p10": values[:, 0],
            "p50": values[:, 1], "p90": values[:, 2]}, crossed


def score(y, pred):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if not n:
        return {"n": 0}
    mean = np.asarray(pred["mean"])
    p10, p50, p90 = (np.asarray(pred[key]) for key in ("p10", "p50", "p90"))
    high = y >= 200
    flagged_high = p90 >= 200
    negative = y < 0
    pinball = lambda q, p: float(np.mean(np.maximum(q * (y - p), (q - 1) * (y - p))))
    return {"n": n, "p50_mae_usd_mwh": float(np.mean(np.abs(y - p50))),
            "mean_mae_usd_mwh": float(np.mean(np.abs(y - mean))),
            "mean_rmse_usd_mwh": float(np.sqrt(np.mean((y - mean) ** 2))),
            "mean_bias_usd_mwh": float(np.mean(mean - y)),
            "pinball_p10": pinball(0.1, p10), "pinball_p50": pinball(0.5, p50),
            "pinball_p90": pinball(0.9, p90),
            "mean_pinball": float(np.mean([pinball(0.1, p10), pinball(0.5, p50), pinball(0.9, p90)])),
            "p10_p90_coverage": float(np.mean((p10 <= y) & (y <= p90))),
            "below_p10": float(np.mean(y < p10)), "above_p90": float(np.mean(y > p90)),
            "high_spike_threshold_usd_mwh": 200, "high_spike_n": int(high.sum()),
            "high_spike_p50_mae_usd_mwh": float(np.mean(np.abs(y[high] - p50[high]))) if high.any() else None,
            "high_spike_p90_recall_at_200": float(np.mean(p90[high] >= 200)) if high.any() else None,
            "high_spike_p90_flagged_n": int(flagged_high.sum()),
            "high_spike_p90_precision_at_200": float(np.mean(high[flagged_high])) if flagged_high.any() else None,
            "negative_price_n": int(negative.sum()),
            "negative_price_p50_mae_usd_mwh": float(np.mean(np.abs(y[negative] - p50[negative]))) if negative.any() else None}


def subset_prediction(pred, mask):
    return {name: values[mask] for name, values in pred.items()}


def paired_week_bootstrap(frame, without, with_risk, draws=3000):
    """Paired risk-minus-no-risk loss difference, resampling whole delivery weeks."""
    y = frame.rtm_spp_usd_mwh.to_numpy(dtype=float)
    def losses(pred):
        pinballs = [np.maximum(q * (y - pred[name]), (q - 1) * (y - pred[name]))
                    for q, name in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90"))]
        return {"p50_mae_usd_mwh": np.abs(y - pred["p50"]),
                "mean_pinball_usd_mwh": np.mean(pinballs, axis=0)}
    a, b = losses(without), losses(with_risk)
    week = frame.delivery_date.dt.to_period("W-SUN").astype(str)
    codes, unique = pd.factorize(week, sort=True)
    n = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    result = {"method": "paired whole-delivery-week bootstrap", "weeks": len(unique), "draws": draws,
              "sign": "with outage minus without outage; negative means improvement"}
    for metric in a:
        diff = b[metric] - a[metric]
        sums = np.bincount(codes, weights=diff, minlength=len(unique))
        estimates = sums[sampled].sum(axis=1) / n[sampled].sum(axis=1)
        result[metric] = {"observed_delta": float(diff.mean()),
                          "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])]}
    return result


def save_predictions(path, frame, forecasts):
    cols = ["model_version", "issued_at_utc", "input_cutoff_utc", "settlement_point",
            "delivery_date", "hour_ending", "quarter", "repeated_hour_flag",
            "interval_start_utc", "interval_end_utc", "dam_spp_usd_mwh",
            "actual_rtm_spp_usd_mwh", "rtm_mean_usd_mwh", "rtm_p10_usd_mwh",
            "rtm_p50_usd_mwh", "rtm_p90_usd_mwh"]
    if "county_onset_risk_mean" in frame:
        cols.append("county_onset_risk_mean")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as file:
        import csv
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        for name, pred in forecasts.items():
            for row, mean, p10, p50, p90 in zip(
                    frame.itertuples(index=False), pred["mean"], pred["p10"], pred["p50"], pred["p90"]):
                record = {"model_version": name, "issued_at_utc": row.model_issue_utc.isoformat(),
                          "input_cutoff_utc": row.input_cutoff_utc.isoformat(),
                          "settlement_point": row.settlement_point,
                          "delivery_date": row.delivery_date.date().isoformat(),
                          "hour_ending": row.hour_ending, "quarter": row.quarter,
                          "repeated_hour_flag": row.repeated_hour_flag,
                          "interval_start_utc": row.interval_start_utc.isoformat(),
                          "interval_end_utc": row.interval_end_utc.isoformat(),
                          "dam_spp_usd_mwh": row.dam_spp_usd_mwh,
                          "actual_rtm_spp_usd_mwh": row.rtm_spp_usd_mwh,
                          "rtm_mean_usd_mwh": float(mean), "rtm_p10_usd_mwh": float(p10),
                          "rtm_p50_usd_mwh": float(p50), "rtm_p90_usd_mwh": float(p90)}
                if "county_onset_risk_mean" in frame:
                    record["county_onset_risk_mean"] = row.county_onset_risk_mean
                writer.writerow(record)


def main_experiment(prices, max_train_rows):
    train = prices[prices.delivery_date < "2024-12-31"]
    val = prices[(prices.delivery_date >= "2025-01-01") & (prices.delivery_date < "2025-06-30")]
    test = prices[prices.delivery_date >= "2025-07-01"]
    if not (train.delivery_date.max() < val.delivery_date.min() < test.delivery_date.min()):
        raise ValueError("Nonchronological main split")
    if not (train.interval_end_utc.max() < val.model_issue_utc.min()
            and val.interval_end_utc.max() < test.model_issue_utc.min()):
        raise ValueError("Training/validation outcomes not complete before next forecast issue")
    rng = np.random.default_rng(SEED)
    if len(train) > max_train_rows:
        train = train.iloc[np.sort(rng.choice(len(train), max_train_rows, replace=False))]
    zones = {zone: n for n, zone in enumerate(sorted(prices.settlement_point.unique()))}
    x_train, x_val = matrix(train, zones), matrix(val, zones)
    y_train = (train.rtm_spp_usd_mwh - train.dam_spp_usd_mwh).to_numpy()
    y_val = (val.rtm_spp_usd_mwh - val.dam_spp_usd_mwh).to_numpy()
    y_test = test.rtm_spp_usd_mwh.to_numpy()
    periods = {"all_heldout": np.ones(len(test), bool),
               "before_2025_12_05": (test.delivery_date < CHANGE).to_numpy(),
               "from_2025_12_05": (test.delivery_date >= CHANGE).to_numpy()}
    result = {"status": "retrospective", "train_period": "2023-01-01 through 2024-12-30",
              "validation_period": "2025-01-01 through 2025-06-29",
              "heldout_period": "2025-07-01 through 2025-12-31",
              "split_embargo": "omit 2024-12-31 and 2025-06-30 delivery labels before following-day forecast issues",
              "train_rows_fit": len(train), "validation_rows": len(val), "heldout_rows": len(test),
              "training_sampled": len(train) < len(prices[prices.delivery_date < "2024-12-31"]),
              "random_training_sample_seed": SEED, "zone_codes": zones,
              "feature_inputs": ["DAM SPP", "load zone", "delivery calendar", "DST repeated-hour flag"],
              "model_training": {}, "validation": {}, "heldout": {}}
    benchmark = {key: test.dam_spp_usd_mwh.to_numpy() for key in ("mean", "p10", "p50", "p90")}
    result["heldout"]["dam_degenerate"] = {p: score(y_test[m], subset_prediction(benchmark, m)) for p, m in periods.items()}
    result["validation"]["dam_degenerate"] = score(val.rtm_spp_usd_mwh, {
        key: val.dam_spp_usd_mwh.to_numpy() for key in benchmark})
    forecasts = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for family in ("lightgbm", "catboost", "quantile_gam"):
        print(f"Fitting main {family}", flush=True)
        models, duration = fit_family(family, x_train, y_train, x_val, y_val, max_train_rows)
        joblib.dump({"family": family, "models": models, "zone_codes": zones,
                     "features": FEATURES, "target": "RTM minus DAM, USD/MWh",
                     "status": "retrospective"}, MODEL_DIR / f"main_{family}.joblib", compress=3)
        pred_val, crossed_val = forecast(family, models, val, zones)
        pred_test, crossed_test = forecast(family, models, test, zones)
        result["model_training"][family] = {"seconds": duration, "quantile_crossings_validation": crossed_val,
                                             "quantile_crossings_heldout": crossed_test,
                                             "artifact": f"data/price/models/main_{family}.joblib"}
        if family == "quantile_gam":
            result["model_training"][family]["optimizer"] = models["optimizer"]
        result["validation"][family] = score(val.rtm_spp_usd_mwh, pred_val)
        result["heldout"][family] = {p: score(y_test[m], subset_prediction(pred_test, m))
                                      for p, m in periods.items()}
        forecasts[family] = pred_test
    save_predictions(DATA / "heldout_predictions_2025.csv.gz", test, forecasts)
    return result


def outage_experiment(prices):
    path = DATA / "outage_risk_2023.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Run build_price_outage_feature.py first: {path}")
    risk = pd.read_csv(path, parse_dates=["issue_utc", "valid_start_utc", "valid_end_utc"]).rename(
        columns={"issue_utc": "outage_issue_utc", "valid_start_utc": "valid_hour_start_utc",
                 "mean_county_first_onset_probability": "county_onset_risk_mean"})
    if risk[["settlement_point", "valid_hour_start_utc"]].duplicated().any():
        raise ValueError("Duplicate zone-hour outage forecast")
    source = prices[prices.delivery_date.dt.year == 2023].copy()
    source["valid_hour_start_utc"] = source.interval_start_utc.dt.floor("h")
    joined = source.merge(risk, on=["settlement_point", "valid_hour_start_utc"], how="left",
                          validate="many_to_one")
    later_issue = joined.outage_issue_utc.notna() & (joined.outage_issue_utc > joined.input_cutoff_utc)
    joined.loc[later_issue, "county_onset_risk_mean"] = np.nan
    has_risk = joined.county_onset_risk_mean.notna()
    covered = joined[has_risk].copy()
    if not ((covered.outage_issue_utc <= covered.input_cutoff_utc)
            & (covered.valid_hour_start_utc >= covered.outage_issue_utc)
            & (covered.valid_hour_start_utc < covered.outage_issue_utc + pd.Timedelta(hours=24))
            & (covered.valid_end_utc == covered.valid_hour_start_utc + pd.Timedelta(hours=1))).all():
        raise ValueError("Outage forecast timing leakage")
    zones = {zone: n for n, zone in enumerate(sorted(prices.settlement_point.unique()))}
    train = covered[covered.delivery_date < "2023-06-30"]
    val = covered[(covered.delivery_date >= "2023-07-01") & (covered.delivery_date < "2023-09-30")]
    test = covered[covered.delivery_date >= "2023-10-01"]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Outage ablation split has no coverage")
    if not (train.interval_end_utc.max() < val.model_issue_utc.min()
            and val.interval_end_utc.max() < test.model_issue_utc.min()):
        raise ValueError("Ablation outcomes not complete before next forecast issue")
    base_x = [matrix(x.drop(columns="county_onset_risk_mean"), zones) for x in (train, val)]
    risk_x = [matrix(x, zones) for x in (train, val)]
    y_train = (train.rtm_spp_usd_mwh - train.dam_spp_usd_mwh).to_numpy()
    y_val = (val.rtm_spp_usd_mwh - val.dam_spp_usd_mwh).to_numpy()
    result = {"status": "retrospective 2023-only ablation; frozen outage model held out in 2023",
              "train_period": "2023-01-02 through 2023-06-29",
              "validation_period": "2023-07-01 through 2023-09-29",
              "heldout_period": "2023-10-01 through 2023-12-31",
              "split_embargo": "omit 2023-06-30 and 2023-09-30 delivery labels before following-day forecast issues",
              "risk_interpretation": "Equal-county mean probability of a first recorded county episode in the hour; not homes or lost MW",
              "all_2023_price_rows": len(joined), "covered_2023_price_rows": len(covered),
              "coverage_fraction": float(has_risk.mean()),
              "later_outage_issue_rejected_rows": int(later_issue.sum()),
              "coverage_by_zone": {zone: {"total": len(g), "covered": int(g.county_onset_risk_mean.notna().sum())}
                                   for zone, g in joined.groupby("settlement_point")},
              "train_rows": len(train), "validation_rows": len(val), "heldout_rows": len(test),
              "models": {}}
    forecasts = {}
    for name, pair in (("without_outage", base_x), ("with_outage", risk_x)):
        print(f"Fitting ablation {name}", flush=True)
        models, duration = fit_family("lightgbm", pair[0], y_train, pair[1], y_val, len(train))
        pred, crossed = forecast("lightgbm", models,
                                 test if name == "with_outage" else test.drop(columns="county_onset_risk_mean"), zones)
        joblib.dump({"family": "lightgbm", "models": models, "zone_codes": zones,
                     "features": FEATURES + (["county_onset_risk_mean"] if name == "with_outage" else []),
                     "status": "retrospective"}, MODEL_DIR / f"ablation_{name}.joblib", compress=3)
        result["models"][name] = {"fit_seconds": duration, "quantile_crossings": crossed,
                                   "heldout": score(test.rtm_spp_usd_mwh, pred),
                                   "artifact": f"data/price/models/ablation_{name}.joblib"}
        forecasts[name] = pred
    baseline = {key: test.dam_spp_usd_mwh.to_numpy() for key in ("mean", "p10", "p50", "p90")}
    result["dam_degenerate_heldout"] = score(test.rtm_spp_usd_mwh, baseline)
    result["paired_week_bootstrap"] = paired_week_bootstrap(
        test, forecasts["without_outage"], forecasts["with_outage"])
    save_predictions(DATA / "outage_ablation_predictions_2023.csv.gz", test, forecasts)
    return result


def self_test():
    y = np.array([0., 10.])
    p = {key: np.array([0., 0.]) for key in ("mean", "p10", "p50", "p90")}
    m = score(y, p)
    assert m["n"] == 2 and m["p50_mae_usd_mwh"] == 5 and m["p10_p90_coverage"] == 0.5
    assert m["pinball_p50"] == 2.5 and m["high_spike_n"] == 0
    assert m["high_spike_p90_flagged_n"] == 0 and m["high_spike_p90_precision_at_200"] is None
    frame = pd.DataFrame({"delivery_date": pd.to_datetime(["2023-10-01", "2023-10-09"]),
                          "rtm_spp_usd_mwh": y})
    assert paired_week_bootstrap(frame, p, p, 20)["p50_mae_usd_mwh"]["ci95"] == [0., 0.]
    toy_x = csr_matrix(np.repeat([-1., 1.], 50).reshape(-1, 1))
    toy_y = np.r_[np.tile(np.arange(10.), 5), 20 + np.tile(np.arange(10.), 5)]
    penalty = np.eye(1) * 1e-5
    mean, _ = fit_spline_loss(toy_x, toy_y, penalty)
    toy_quantiles = [fit_spline_loss(toy_x, toy_y, penalty, q, mean)[0] for q in QUANTILES]
    fitted = np.column_stack([np.asarray(toy_x @ c[:-1]).ravel() + c[-1] for c in toy_quantiles])
    assert np.all(fitted[:, 0] <= fitted[:, 1]) and np.all(fitted[:, 1] <= fitted[:, 2])
    assert np.allclose(fitted[[0, 50]], [[0, 4.5, 9], [20, 24.5, 29]], atol=1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--experiment", choices=("main", "outage", "all"), default="all")
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
    else:
        prices = load_prices()
        path = DATA / "metrics.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        result.update({"prediction_target": "15-minute ERCOT LZ RTM SPP, USD/MWh",
                       "point_type": "LZ only; LZEW excluded", "source_audit": "data/price/source_audit.json",
                       "vintage_status": "retrospective; original DAM runs unavailable",
                       "package_versions": {p: importlib.metadata.version(p) for p in
                                            ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost")}})
        if args.experiment in ("main", "all"):
            result["main"] = main_experiment(prices, args.max_train_rows)
        if args.experiment in ("outage", "all"):
            result["outage_ablation"] = outage_experiment(prices)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(path)
