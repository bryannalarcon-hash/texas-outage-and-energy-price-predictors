#!/usr/bin/env python3
"""Run the immutable, retrospectively approved EXP001 six-fold price experiment."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from build_price_model_data import expected_intervals
from train_price_models import CATS, FEATURES, ROOT, SEED, load_prices, matrix, score


OUT = ROOT / "data/price/experiments/EXP001"
TABLE = ROOT / "data/price/price_rows_2023_2025.csv.gz"
PREFLIGHT = ROOT / "data/price/research/master_ds_preflight_001.md"
EXPECTED_HASH = "425df06f0ed056cb7ff909af32799c7972ec17465e317cbfa42fa37cd1fb067e"
EXPECTED_FEATURES = ["dam_asinh", "local_hour", "day_of_year", "zone_code", "weekday", "quarter", "repeated"]
OUTPUTS = {"mean": None, "p10": .1, "p50": .5, "p90": .9}
GROUPS = {"dam_input": ["dam_asinh"], "date": ["day_of_year", "weekday"],
          "intraday": ["local_hour", "quarter"], "zone": ["zone_code"]}
BASELINES = ("raw_dam", "global_residual", "zone_hour_residual")
FOLDS = [
    ("2024Q1", "2024-01-01", "2024-04-01", 257280, 21504, 279552, 69856),
    ("2024Q2", "2024-04-01", "2024-07-01", 327168, 21472, 349408, 69888),
    ("2024Q3", "2024-07-01", "2024-10-01", 397024, 21504, 419296, 70656),
    ("2024Q4", "2024-10-01", "2025-01-01", 467680, 21504, 489952, 70688),
    ("2025Q1", "2025-01-01", "2025-04-01", 538368, 21504, 560640, 69088),
    ("2025Q2", "2025-04-01", "2025-07-01", 607488, 21472, 629728, 69888),
]
PARAMS = dict(n_estimators=450, learning_rate=.045, num_leaves=23,
              min_child_samples=100, reg_lambda=2., verbosity=-1, n_jobs=4, random_state=SEED)
KEYS = ["settlement_point", "interval_start_utc"]
ROW_COLS = ["model_issue_utc", "input_cutoff_utc", "delivery_date", "settlement_point",
            "hour_ending", "quarter", "repeated_hour_flag", "interval_start_utc",
            "interval_end_utc", "dam_spp_usd_mwh", "rtm_spp_usd_mwh"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value, exclusive=False):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text)
        temp.replace(path)


def validate_prices(frame):
    assert list(FEATURES) == EXPECTED_FEATURES, "Original feature helper changed"
    assert len(frame) == 841728 and frame.delivery_date.min() == pd.Timestamp("2023-01-01")
    assert frame.delivery_date.max() == pd.Timestamp("2025-12-31")
    assert frame.settlement_point.nunique() == 8
    assert not frame[KEYS].duplicated().any(), "Duplicate UTC zone-interval key"
    assert not frame[ROW_COLS].isna().any().any(), "Missing source value"
    assert np.isfinite(frame[["dam_spp_usd_mwh", "rtm_spp_usd_mwh"]]).all().all()
    assert frame.quarter.isin([1, 2, 3, 4]).all() and frame.hour_ending.between(1, 24).all()
    assert frame.repeated_hour_flag.isin(["N", "Y"]).all()
    assert (frame.interval_end_utc - frame.interval_start_utc == pd.Timedelta(minutes=15)).all()
    local = frame.interval_start_utc.dt.tz_convert("America/Chicago")
    assert (local.dt.tz_localize(None).dt.normalize() == frame.delivery_date).all()
    assert (local.dt.hour + 1 == frame.hour_ending).all()
    assert (local.dt.minute == (frame.quarter - 1) * 15).all()
    assert (local.dt.second == 0).all()
    clock = frame[["settlement_point"]].assign(local=local.dt.tz_localize(None))
    assert np.array_equal(clock.duplicated().to_numpy(), (frame.repeated_hour_flag == "Y").to_numpy())
    issue = frame.delivery_date.dt.tz_localize("UTC") - pd.Timedelta(hours=2)
    assert (frame.model_issue_utc == issue).all() and (frame.input_cutoff_utc == issue).all()
    assert (issue < frame.interval_start_utc).all()
    counts = frame.groupby(["delivery_date", "settlement_point"]).size()
    assert len(counts) == 1096 * 8
    expected = counts.index.get_level_values(0).map(lambda x: expected_intervals(x.date()))
    assert np.array_equal(counts.to_numpy(), expected.to_numpy()), "Incomplete operating day"
    source = json.loads((ROOT / "data/price/source_audit.json").read_text())
    assert source["price_unit"] == "USD/MWh"
    return {"rows": len(frame), "zone_count": 8, "unique_utc_keys": True,
            "finite_prices": True, "utc_local_dst_mapping": True,
            "complete_days_92_96_100": True, "price_units": "USD/MWh"}


def split(frame, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    day = frame.delivery_date
    parts = (
        frame[day < start - pd.Timedelta(days=30)],
        frame[(day >= start - pd.Timedelta(days=29)) & (day < start - pd.Timedelta(days=1))],
        frame[day < start - pd.Timedelta(days=1)],
        frame[(day >= start) & (day < end)],
    )
    fit, stop, refit, outer = parts
    assert all(len(x) for x in parts)
    assert fit.interval_end_utc.max() < stop.model_issue_utc.min(), "Inner embargo violated"
    assert stop.interval_end_utc.max() < outer.model_issue_utc.min(), "Stopping embargo violated"
    assert refit.interval_end_utc.max() < outer.model_issue_utc.min(), "Refit embargo violated"
    return parts


def design(frame, zones):
    x = matrix(frame, zones)
    assert x.columns.tolist() == EXPECTED_FEATURES
    assert np.isfinite(x.to_numpy(dtype=float)).all()
    for col in CATS:
        x[col] = pd.Categorical(x[col], categories=(list(zones.values()) if col == "zone_code"
                                                   else list(range(7)) if col == "weekday"
                                                   else [1, 2, 3, 4] if col == "quarter" else [0, 1]))
    return x


def residual(frame):
    return (frame.rtm_spp_usd_mwh - frame.dam_spp_usd_mwh).to_numpy(dtype=float)


def fit_models(frame, start, end, zones, test_rounds=None):
    fit, stop, refit, _ = split(frame, start, end)
    x_fit, x_stop, x_refit = (design(x, zones) for x in (fit, stop, refit))
    y_fit, y_stop, y_refit = (residual(x) for x in (fit, stop, refit))
    models, selected, timing = {}, {}, {}
    for name, q in OUTPUTS.items():
        begun = time.monotonic()
        params = dict(PARAMS, **({"objective": "regression"} if q is None
                                else {"objective": "quantile", "alpha": q}))
        if test_rounds is not None:
            params["n_estimators"] = test_rounds
        inner = LGBMRegressor(**params)
        inner.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)],
                  callbacks=[early_stopping(35, verbose=False)])
        selected[name] = int(inner.best_iteration_)
        assert 1 <= selected[name] <= params["n_estimators"]
        models[name] = LGBMRegressor(**dict(params, n_estimators=selected[name]))
        models[name].fit(x_refit, y_refit)
        timing[name] = round(time.monotonic() - begun, 4)
    return models, selected, timing


def predict(models, x, offset):
    raw = {name: np.asarray(model.predict(x)) + offset for name, model in models.items()}
    quantiles = np.column_stack([raw[name] for name in ("p10", "p50", "p90")])
    crossed = int(np.any(np.diff(quantiles, axis=1) < 0, axis=1).sum())
    quantiles.sort(axis=1)
    out = dict(mean=raw["mean"], p10=quantiles[:, 0], p50=quantiles[:, 1], p90=quantiles[:, 2])
    assert np.isfinite(np.column_stack(list(out.values()))).all()
    return out, crossed


def benchmarks(refit, outer):
    y = residual(refit)
    offset = outer.dam_spp_usd_mwh.to_numpy(dtype=float)
    global_values = {name: float(y.mean()) if q is None else float(np.quantile(y, q))
                     for name, q in OUTPUTS.items()}
    result = {"raw_dam": {name: offset.copy() for name in OUTPUTS},
              "global_residual": {name: offset + value for name, value in global_values.items()}}
    reference = refit[["settlement_point", "hour_ending"]].assign(residual=y)
    groups = reference.groupby(["settlement_point", "hour_ending"]).residual
    index = pd.MultiIndex.from_frame(outer[["settlement_point", "hour_ending"]])
    cell = {}
    estimates = {}
    for name, q in OUTPUTS.items():
        values = groups.mean() if q is None else groups.quantile(q)
        cell[name] = offset + values.reindex(index).fillna(global_values[name]).to_numpy()
        estimates[name] = {f"{zone}|{hour}": float(value) for (zone, hour), value in values.items()}
    result["zone_hour_residual"] = cell
    return result, {"global_residual": global_values, "zone_hour_residual": estimates,
                    "cell_keys": "settlement point | hour ending (both fall-back copies share a cell)",
                    "missing_cell_fallback": "global historical residual"}


def get_pred(frame, method):
    return {name: frame[f"{method}_{name}"].to_numpy(dtype=float) for name in OUTPUTS}


def scores(frame, method):
    pred = get_pred(frame, method)
    y = frame.rtm_spp_usd_mwh.to_numpy(dtype=float)
    result = score(y, pred)
    if not len(frame):
        return result
    high, alarm = y >= 200, pred["p90"] >= 200
    result.update({"mean_mse_usd_mwh_squared": float(np.mean((y - pred["mean"]) ** 2)),
                   "p10_p90_mean_width_usd_mwh": float(np.mean(pred["p90"] - pred["p10"])),
                   "high_spike_false_alarms_n": int((alarm & ~high).sum()),
                   "high_spike_false_alarm_rate": float(alarm[~high].mean()) if (~high).any() else None,
                   "high_spike_delivery_dates": sorted(frame.loc[high, "delivery_date"].dt.strftime("%Y-%m-%d").unique().tolist())})
    return result


def losses(frame, method):
    y = frame.rtm_spp_usd_mwh.to_numpy(dtype=float)
    p = get_pred(frame, method)
    return {"p50_mae_usd_mwh": np.abs(y - p["p50"]),
            "mean_mse_usd_mwh_squared": (y - p["mean"]) ** 2,
            "mean_pinball_usd_mwh": np.mean([np.maximum(q * (y - p[name]), (q - 1) * (y - p[name]))
                                              for name, q in OUTPUTS.items() if q is not None], axis=0)}


def paired_uncertainty(frame, baseline, draws=3000):
    a, b = losses(frame, baseline), losses(frame, "lightgbm")
    result = {"sign": "model minus benchmark; negative delta is improvement",
              "interpretation": "descriptive uncertainty on exposed, repeatedly usable development data",
              "draws": draws, "seed": SEED}
    for label, groups in (("whole_delivery_week", frame.delivery_date.dt.to_period("W-SUN")),
                           ("moving_seven_day", frame.delivery_date)):
        codes, dates = pd.factorize(groups, sort=True)
        n = np.bincount(codes)
        rng = np.random.default_rng(SEED)
        if label == "whole_delivery_week":
            picked = rng.integers(0, len(n), size=(draws, len(n)))
        else:
            assert len(dates) >= 7 and (pd.Series(dates).diff().dropna() == pd.Timedelta(days=1)).all()
            starts = rng.integers(0, len(n) - 6, size=(draws, (len(n) + 6) // 7))
            picked = (starts[:, :, None] + np.arange(7)).reshape(draws, -1)[:, :len(n)]
        denominator = n[picked].sum(axis=1)
        estimates = {}
        for metric in a:
            sums_a = np.bincount(codes, weights=a[metric])
            sums_b = np.bincount(codes, weights=b[metric])
            delta = (sums_b[picked].sum(axis=1) - sums_a[picked].sum(axis=1)) / denominator
            estimates[metric] = {"observed_delta": float((b[metric] - a[metric]).mean()),
                                 "ci95": np.quantile(delta, [.025, .975]).tolist()}
            if label == "whole_delivery_week":
                estimates[metric].update(improved_weeks=int((sums_b < sums_a).sum()), weeks=len(n))
            if metric == "mean_mse_usd_mwh_squared":
                rmse_delta = np.sqrt(sums_b[picked].sum(axis=1) / denominator) - np.sqrt(sums_a[picked].sum(axis=1) / denominator)
                estimates["mean_rmse_usd_mwh"] = {
                    "observed_delta": float(np.sqrt(b[metric].mean()) - np.sqrt(a[metric].mean())),
                    "ci95": np.quantile(rmse_delta, [.025, .975]).tolist()}
        result[label] = {"resampling_units": len(n), "metrics": estimates}
    return result


def day_blocks(frame):
    blocks = []
    strata = {}
    for day, group in frame.groupby("delivery_date", sort=True):
        ix = group.index.to_numpy()
        assert len(ix) % 8 == 0 and np.all(np.diff(ix) == 1)
        assert group.settlement_point.nunique() == 8
        slots = group.interval_start_utc.nunique()
        assert len(ix) == slots * 8 and slots in (92, 96, 100)
        strata.setdefault((day.month, slots), []).append(len(blocks))
        blocks.append(ix)
    return blocks, strata


def perturb(x, group, rng, blocks, strata):
    result = x.copy()
    if group in ("dam_input", "date"):
        donors = np.arange(len(x))
        for members in strata.values():
            for target, source in zip(members, rng.permutation(members)):
                donors[blocks[target]] = blocks[source]
        for col in GROUPS[group]:
            result[col] = x[col].iloc[donors].to_numpy()
    elif group == "intraday":
        donors = np.arange(len(x))
        for ix in blocks:
            slots = ix.reshape(-1, 8)
            hours = len(slots) // 4
            donors[ix] = np.roll(slots, int(rng.integers(1, hours)) * 4, axis=0).ravel()
        for col in GROUPS[group]:
            result[col] = x[col].iloc[donors].to_numpy()
    elif group == "zone":
        mapping = rng.permutation(8)
        result["zone_code"] = mapping[x.zone_code.to_numpy(dtype=int)]
    else:
        raise ValueError(group)
    changed = {col: float(np.mean(result[col].to_numpy() != x[col].to_numpy())) for col in GROUPS[group]}
    changed["any_group_value"] = float(np.any(result[GROUPS[group]].to_numpy() != x[GROUPS[group]].to_numpy(), axis=1).mean())
    for col in CATS:
        result[col] = pd.Categorical(result[col], categories=x[col].cat.categories)
    return result, changed


def importance(fold, outer, bundle, base):
    x = design(outer, bundle["zone_codes"])
    offset = outer.dam_spp_usd_mwh.to_numpy(dtype=float)
    blocks, strata = day_blocks(outer)
    original = score(outer.rtm_spp_usd_mwh, base)
    records = []
    for group in GROUPS:
        for repeat in range(10):
            # Reset per group so DAM and date diagnostics use the same donor maps.
            rng = np.random.default_rng(np.random.SeedSequence([SEED, list(x for x, *_ in FOLDS).index(fold), repeat]))
            xp, changed = perturb(x, group, rng, blocks, strata)
            pred, crossings = predict(bundle["models"], xp, offset)
            trial = score(outer.rtm_spp_usd_mwh, pred)
            records.append({"fold": fold, "group": group, "repeat": repeat, "rows": len(outer),
                            "changed_fraction": changed, "raw_quantile_crossings": crossings,
                            "p50_mae_increase_usd_mwh": trial["p50_mae_usd_mwh"] - original["p50_mae_usd_mwh"],
                            "mean_mse_increase_usd_mwh_squared": trial["mean_rmse_usd_mwh"] ** 2 - original["mean_rmse_usd_mwh"] ** 2,
                            "mean_rmse_increase_usd_mwh": trial["mean_rmse_usd_mwh"] - original["mean_rmse_usd_mwh"],
                            "mean_pinball_increase_usd_mwh": trial["mean_pinball"] - original["mean_pinball"]})
        print(f"{fold}: completed grouped importance {group}", flush=True)
    return records


def summarize_importance(records, pooled):
    keys = ["p50_mae_increase_usd_mwh", "mean_mse_increase_usd_mwh_squared", "mean_pinball_increase_usd_mwh"]
    results = {}
    for group in GROUPS:
        repeats = []
        for repeat in range(10):
            items = [r for r in records if r["group"] == group and r["repeat"] == repeat]
            assert len(items) == 6
            row = {key: sum(r[key] * r["rows"] for r in items) / sum(r["rows"] for r in items) for key in keys}
            row["mean_rmse_increase_usd_mwh"] = float(np.sqrt(pooled["mean_rmse_usd_mwh"] ** 2
                                                              + row["mean_mse_increase_usd_mwh_squared"])
                                                     - pooled["mean_rmse_usd_mwh"])
            row["fraction_rows_changed"] = sum(r["changed_fraction"]["any_group_value"] * r["rows"] for r in items) / sum(r["rows"] for r in items)
            repeats.append(row)
        results[group] = {key: {"mean": float(np.mean([r[key] for r in repeats])),
                                "monte_carlo_std": float(np.std([r[key] for r in repeats], ddof=1)),
                                "minimum": float(min(r[key] for r in repeats)),
                                "maximum": float(max(r[key] for r in repeats))} for key in repeats[0]}
    return {"seed": SEED, "repeats": 10, "estimand": "Correction reliance; original additive DAM offset is held fixed",
            "methods": {"dam_input": "Whole operating-day swaps within outer fold, calendar month, and 92/96/100-quarter shape; same donor mapping across all zones",
                        "date": "Same coherent donor-day method; day-of-year and weekday moved together",
                        "intraday": "Nonzero whole-hour cyclic rotation within operating day, shared across zones; hour and quarter moved together. Calendar sensitivity, not a plausible physical counterfactual",
                        "zone": "One permutation of eight zone codes for the entire outer fold per repeat"},
            "untested": {"repeated": "Non-identifiable by meaningful blocked individual importance with one fall transition per year"},
            "interpretation": "Repeat variation is Monte Carlo variation, not population confidence; no importance threshold selected features",
            "pooled": results, "fold_repeats": records}


def read_predictions(path):
    return pd.read_csv(path, parse_dates=["model_issue_utc", "input_cutoff_utc", "delivery_date",
                                          "interval_start_utc", "interval_end_utc"])


def verify_predictions(saved, source):
    assert len(saved) == 420064 and not saved[KEYS].duplicated().any()
    expected = source[(source.delivery_date >= "2024-01-01") & (source.delivery_date < "2025-07-01")]
    assert saved[KEYS].reset_index(drop=True).equals(expected[KEYS].reset_index(drop=True))
    for col in ROW_COLS:
        if col in ("dam_spp_usd_mwh", "rtm_spp_usd_mwh"):
            np.testing.assert_allclose(saved[col], expected[col], rtol=0, atol=1e-9)
        else:
            assert saved[col].reset_index(drop=True).equals(expected[col].reset_index(drop=True)), col
    for method in ("lightgbm", *BASELINES):
        p = get_pred(saved, method)
        assert np.isfinite(np.column_stack(list(p.values()))).all()
        assert ((p["p10"] <= p["p50"]) & (p["p50"] <= p["p90"])).all()
    return {"rows": len(saved), "same_row_keys_and_values_as_source": True,
            "four_forecasts_each_model_and_benchmark": True, "finite_ordered_quantiles": True}


def self_test():
    # Exercise the actual nested fit/refit entry point, altering only future outcomes.
    times = pd.date_range("2023-11-01 15:00", "2024-01-03 18:45", freq="15min", tz="UTC")
    times = times[(times.hour >= 15) & (times.hour < 19)]
    local = times.tz_convert("America/Chicago")
    toy = pd.DataFrame({"interval_start_utc": times, "interval_end_utc": times + pd.Timedelta(minutes=15),
                        "delivery_date": local.tz_localize(None).normalize(), "settlement_point": "LZ_TEST",
                        "hour_ending": local.hour + 1, "quarter": local.minute // 15 + 1,
                        "repeated_hour_flag": "N", "dam_spp_usd_mwh": 30 + 2 * np.sin(np.arange(len(times)) / 10)})
    toy["model_issue_utc"] = toy.delivery_date.dt.tz_localize("UTC") - pd.Timedelta(hours=2)
    toy["input_cutoff_utc"] = toy.model_issue_utc
    toy["rtm_spp_usd_mwh"] = toy.dam_spp_usd_mwh + 3 * np.sin(np.arange(len(times)) / 50)
    original = fit_models(toy, "2024-01-01", "2024-04-01", {"LZ_TEST": 0}, test_rounds=40)
    changed = toy.copy()
    changed.loc[changed.delivery_date >= "2024-01-01", "rtm_spp_usd_mwh"] += 1_000_000
    altered = fit_models(changed, "2024-01-01", "2024-04-01", {"LZ_TEST": 0}, test_rounds=40)
    assert original[1] == altered[1], "Future labels changed iteration selection"
    assert all(original[0][key].booster_.model_to_string() == altered[0][key].booster_.model_to_string()
               for key in OUTPUTS), "Future labels changed the fitted model"
    bad = toy.copy()
    bad.loc[bad.delivery_date == "2023-12-30", "interval_end_utc"] = pd.Timestamp("2024-01-02", tz="UTC")
    try:
        split(bad, "2024-01-01", "2024-04-01")
    except AssertionError:
        pass
    else:
        raise AssertionError("Injected embargo breach accepted")
    return {"status": "passed", "outer_label_mutation": "All four serialized model bodies and selected iterations identical",
            "injected_embargo_violation": "rejected", "test_fixture_round_cap": 40,
            "test_fixture_only": "Toy fits are a guard; experiment configuration remains 450 maximum rounds"}


def permutation_check(prices):
    sample = prices[(prices.delivery_date >= "2024-03-08") & (prices.delivery_date < "2024-03-13")].reset_index(drop=True)
    zones = {zone: n for n, zone in enumerate(sorted(sample.settlement_point.unique()))}
    x = design(sample, zones)
    blocks, strata = day_blocks(sample)
    assert {len(block) // 8 for block in blocks} == {92, 96}
    encoded = x.copy()
    encoded["dam_asinh"] = np.arange(len(encoded), dtype=float)
    for group in GROUPS:
        changed, _ = perturb(encoded, group, np.random.default_rng(SEED), blocks, strata)
        for column in encoded.columns.difference(GROUPS[group]):
            assert changed[column].equals(encoded[column]), (group, column)
        if group == "dam_input":
            donor = changed.dam_asinh.to_numpy(dtype=int)
            for block in blocks:
                assert np.all(np.diff(donor[block]) == 1), "Donor day fragmented"
                assert sample.iloc[donor[block]].delivery_date.nunique() == 1
                assert np.array_equal(sample.iloc[donor[block]].settlement_point, sample.iloc[block].settlement_point)
        if group == "intraday":
            assert changed.quarter.equals(encoded.quarter), "Whole-hour rotation changed quarter"
            assert ((changed.local_hour - encoded.local_hour) % 1 == 0).all()
        if group == "zone":
            for block in blocks:
                zone_matrix = changed.zone_code.iloc[block].to_numpy().reshape(-1, 8)
                assert np.all(zone_matrix == zone_matrix[0]), "Zone mapping changed within fold"
                assert len(np.unique(zone_matrix[0])) == 8
    return {"status": "passed", "spring_dst_day_shape_preserved": True,
            "donor_days_unfragmented": True, "cross_zone_donor_mapping_shared": True,
            "non_group_features_unchanged": True}


def make_manifest(prices, checks):
    zones = {zone: n for n, zone in enumerate(sorted(prices.settlement_point.unique()))}
    folds = []
    for name, start, end, *counts in FOLDS:
        parts = split(prices, start, end)
        assert [len(x) for x in parts] == counts, (name, [len(x) for x in parts], counts)
        folds.append({"fold": name, "score_start_inclusive": start, "score_end_exclusive": end,
                      "parts": {label: {"rows": len(part), "delivery_start": str(part.delivery_date.min().date()),
                                         "delivery_end": str(part.delivery_date.max().date()),
                                         "max_label_interval_end_utc": part.interval_end_utc.max().isoformat(),
                                         "min_issue_utc": part.model_issue_utc.min().isoformat()}
                                for label, part in zip(("inner_fit", "early_stop", "final_refit", "outer_score"), parts)}})
    paths = [Path(__file__), ROOT / "train_price_models.py", ROOT / "build_price_model_data.py",
             TABLE, PREFLIGHT, ROOT / "data/price/source_audit.json", ROOT / "data/ercot/source_manifest.json"]
    return {"experiment_id": "EXP001", "created_at_utc": now(), "preflight_path": str(PREFLIGHT.relative_to(ROOT)),
            "hashes_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths},
            "features": FEATURES, "categorical": CATS, "zone_codes": zones, "row_keys": KEYS,
            "folds": folds, "total_outer_rows": 420064, "configuration": PARAMS,
            "objectives": OUTPUTS, "early_stopping_rounds": 35, "seed": SEED,
            "issue_rule": "22:00 UTC on previous calendar day; cutoff equals issue",
            "units": "USD/MWh", "target": "Untransformed and unclipped RTM minus DAM",
            "data_vintage_caveats": ["Final-value annual price archives; original DAM and RTM publication/revision vintages unverified",
                                     "Embargo proves completed intervals, not historical label publication",
                                     "All score periods are exposed research data, not confirmatory holdouts"],
            "preprocessing": "Fixed asinh(DAM/50); predetermined zone category vocabulary; no learned scaling or imputation",
            "package_versions": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")},
            "importance": {"groups": GROUPS, "seed": SEED, "repeats": 10,
                           "original_additive_dam_offset_fixed": True, "untested": ["repeated"]},
            "checks_before_fit": checks}


def run():
    begun = time.monotonic()
    assert digest(TABLE) == EXPECTED_HASH, "Source table differs from reviewed immutable input"
    prices = load_prices()
    checks = {"source_validation": validate_prices(prices), "leakage_self_test": self_test(),
              "permutation_check": permutation_check(prices)}
    manifest = make_manifest(prices, checks)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "manifest.json", manifest, exclusive=True)
    zones = manifest["zone_codes"]
    record = {"experiment_id": "EXP001", "status": "running", "hypothesis": "Fixed DAM/calendar correction has repeatable benefit across six forward development quarters",
              "parent": "existing seven-input LightGBM baseline", "model_family": "LightGBM residual mean and quantiles",
              "feature_groups": list(GROUPS) + ["dst_repeated"], "feature_columns": FEATURES,
              "techniques": ["purged expanding-window quarters", "nested per-objective early stopping", "purged full-history refit",
                             "three same-row benchmarks", "paired week and moving-seven-day bootstrap", "coherent grouped permutation"],
              "evaluation_role": "retrospective exposed development-validation", "preflight_path": manifest["preflight_path"],
              "started_at_utc": now(), "completed_at_utc": None, "data_vintage_caveats": manifest["data_vintage_caveats"],
              "process": {"pid": os.getpid(), "command": [sys.executable, *sys.argv], "log": "run.log"},
              "artifact_links": {"manifest": "manifest.json", "predictions": "predictions.csv.gz", "metrics": "metrics.json",
                                 "importance": "importance.json", "verification": "verification.json", "models": "folds/"},
              "completed_folds": []}
    save_json(OUT / "record.json", record)
    pooled_frames, permutation_records, fit_records = [], [], []
    try:
        for name, start, end, *_ in FOLDS:
            fold_started = time.monotonic()
            fold_dir = OUT / "folds" / name
            fold_dir.mkdir(parents=True, exist_ok=False)
            print(f"{now()} fitting {name}", flush=True)
            _, _, refit, outer = split(prices, start, end)
            outer = outer.reset_index(drop=True)
            models, iterations, timing = fit_models(prices, start, end, zones)
            bundle = {"models": models, "zone_codes": zones, "features": FEATURES,
                      "fold": name, "manifest_sha256": digest(OUT / "manifest.json"),
                      "source_sha256": EXPECTED_HASH, "selected_iterations": iterations}
            model_path = fold_dir / "models.joblib"
            joblib.dump(bundle, model_path, compress=3)
            bundle = joblib.load(model_path)
            pred, crossed = predict(bundle["models"], design(outer, zones), outer.dam_spp_usd_mwh.to_numpy(dtype=float))
            reference, baseline_fit = benchmarks(refit, outer)
            save_json(fold_dir / "baseline_fit.json", baseline_fit)
            saved = outer[ROW_COLS].copy()
            saved["fold"] = name
            for method, forecasts in {"lightgbm": pred, **reference}.items():
                for output, values in forecasts.items():
                    saved[f"{method}_{output}"] = values
            saved.to_csv(fold_dir / "predictions.csv.gz", index=False, compression="gzip")
            frozen = read_predictions(fold_dir / "predictions.csv.gz")
            assert frozen[KEYS].equals(saved[KEYS])
            for method in ("lightgbm", *BASELINES):
                np.testing.assert_allclose(list(scores(frozen, method)[key] for key in ("p50_mae_usd_mwh", "mean_rmse_usd_mwh", "mean_pinball")),
                                           list(scores(saved, method)[key] for key in ("p50_mae_usd_mwh", "mean_rmse_usd_mwh", "mean_pinball")), atol=1e-10)
            print(f"{name}: fit iterations {iterations}; P50 MAE {scores(frozen, 'lightgbm')['p50_mae_usd_mwh']:.6f}", flush=True)
            perms = importance(name, frozen, bundle, get_pred(frozen, "lightgbm"))
            save_json(fold_dir / "importance.json", perms)
            fit_record = {"fold": name, "selected_iterations": iterations, "objective_fit_seconds": timing,
                          "raw_quantile_crossings": crossed, "raw_quantile_crossing_fraction": crossed / len(outer),
                          "total_seconds_with_importance": round(time.monotonic() - fold_started, 3),
                          "model_sha256": digest(model_path), "prediction_sha256": digest(fold_dir / "predictions.csv.gz")}
            save_json(fold_dir / "fit.json", fit_record)
            fit_records.append(fit_record)
            pooled_frames.append(frozen)
            permutation_records.extend(perms)
            record["completed_folds"].append(name)
            save_json(OUT / "record.json", record)
        saved = pd.concat(pooled_frames, ignore_index=True)
        saved.to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
        saved = read_predictions(OUT / "predictions.csv.gz")
        verification = verify_predictions(saved, prices)
        methods = ("lightgbm", *BASELINES)
        pooled = {method: scores(saved, method) for method in methods}
        slices = {"fold": saved.fold, "zone": saved.settlement_point,
                  "month": saved.delivery_date.dt.strftime("%Y-%m"),
                  "lead_hour": ((saved.interval_start_utc - saved.model_issue_utc).dt.total_seconds() // 3600).astype(int)}
        metrics = {"pooled": pooled, "model_training": fit_records,
                   "slices": {name: {str(value): {method: scores(saved.loc[ids], method) for method in methods}
                                     for value, ids in values.groupby(values).groups.items()} for name, values in slices.items()},
                   "paired_benchmark_uncertainty": {method: paired_uncertainty(saved, method) for method in BASELINES},
                   "raw_dam_distribution_note": "Raw DAM point price repeated in quantile fields only for same-row diagnostics; it supplies no predictive distribution"}
        save_json(OUT / "metrics.json", metrics)
        save_json(OUT / "importance.json", summarize_importance(permutation_records, pooled["lightgbm"]))
        rows = []
        for method, values in pooled.items():
            rows.append(dict(experiment_id="EXP001", slice_type="pooled", slice_value="all", method=method,
                             **{key: value for key, value in values.items() if not isinstance(value, list)}))
        for kind, values in metrics["slices"].items():
            for value, methods_scores in values.items():
                for method, metrics_scores in methods_scores.items():
                    rows.append(dict(experiment_id="EXP001", slice_type=kind, slice_value=value, method=method,
                                     **{key: v for key, v in metrics_scores.items() if not isinstance(v, list)}))
        pd.DataFrame(rows).to_csv(OUT / "chart_metrics.csv", index=False)
        verification.update({"source_sha256_unchanged": digest(TABLE) == EXPECTED_HASH,
                             "manifest_sha256": digest(OUT / "manifest.json"),
                             "model_and_prediction_hashes": fit_records, "saved_metrics_recomputed": True})
        assert verification["source_sha256_unchanged"]
        save_json(OUT / "verification.json", verification)
        record.update(status="complete", completed_at_utc=now(), elapsed_seconds=round(time.monotonic() - begun, 3),
                      pooled_metrics=pooled["lightgbm"], baselines={key: pooled[key] for key in BASELINES},
                      comparison_rows=[{"model": method, "metrics": pooled[method],
                                        "baseline_id": None if method == "lightgbm" else method,
                                        "cohort_id": "EXP001_2024Q1_2025Q2"} for method in methods])
        save_json(OUT / "record.json", record)
        print(json.dumps({"status": "complete", "seconds": record["elapsed_seconds"], "pooled": pooled}, indent=2), flush=True)
    except BaseException as error:
        record.update(status="failed", failed_at_utc=now(), error=f"{type(error).__name__}: {error}")
        save_json(OUT / "record.json", record)
        raise


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for name, expected in manifest["hashes_sha256"].items():
        assert digest(ROOT / name) == expected, f"Immutable input/code changed: {name}"
    saved = read_predictions(OUT / "predictions.csv.gz")
    verification = verify_predictions(saved, load_prices())
    metrics = json.loads((OUT / "metrics.json").read_text())
    for method in ("lightgbm", *BASELINES):
        current = scores(saved, method)
        for key, value in current.items():
            expected = metrics["pooled"][method][key]
            if isinstance(value, (int, float)):
                assert np.isclose(value, expected, atol=1e-10), (method, key)
            else:
                assert value == expected
    print(json.dumps(dict(verification, saved_metrics_recomputed=True), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--guards", action="store_true", help="Check source, all splits, and future-label independence without experiment fits")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.guards:
        assert digest(TABLE) == EXPECTED_HASH
        prices = load_prices()
        checks = {"source_validation": validate_prices(prices), "leakage_self_test": self_test(),
                  "permutation_check": permutation_check(prices)}
        manifest = make_manifest(prices, checks)
        print(json.dumps({"checks": checks, "fold_rows": [f["parts"] for f in manifest["folds"]],
                          "outer_rows": manifest["total_outer_rows"]}, indent=2))
    elif args.verify:
        verify()
    else:
        run()
