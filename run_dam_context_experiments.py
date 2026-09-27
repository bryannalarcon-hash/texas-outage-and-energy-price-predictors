#!/usr/bin/env python3
"""Run the fixed EXP002/003/004 DAM-context batch approved in preflight 002."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

import run_price_experiment as base


ROOT = base.ROOT
OUT = ROOT / "data/price/experiments"
BATCH = OUT / "BATCH002"
PREFLIGHT = ROOT / "data/price/research/master_ds_preflight_002.md"
COHORT = "EXP001_2024Q1_2025Q2"
ZONES = {z: i for i, z in enumerate(("LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA",
                                    "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST"))}
LOCAL = ["dam_day_min", "dam_day_max", "dam_day_mean", "dam_day_std", "dam_day_negative_share",
         "dam_hours_to_day_peak", "dam_above_day_min", "dam_below_day_max", "dam_prev_hour",
         "dam_next_hour", "dam_ramp_from_prev", "dam_ramp_to_next"]
PEER = [f"dam_peer_{z}" for z in ZONES] + ["dam_minus_peer_median", "dam_peer_std", "dam_peer_range"]
BLOCKS = {"EXP002": LOCAL, "EXP003": PEER, "EXP004": LOCAL + PEER}
HYPOTHESES = {
    "EXP002": "Target-zone whole-day DAM shape improves residual forecasts beyond the target hour alone",
    "EXP003": "Simultaneous peer DAM prices and spreads improve residual forecasts beyond local target-hour DAM",
    "EXP004": "Local temporal DAM context and peer DAM context provide complementary information",
}
FORMULAS = dict(zip(LOCAL, [
    "min(p[d,z,:])", "max(p[d,z,:])", "mean(p[d,z,:]) over actual H_d hours",
    "std(p[d,z,:], ddof=0)", "count(p[d,z,:] < 0) / H_d",
    "earliest UTC-ordered argmax(p[d,z,:]) - j", "p - dam_day_min", "dam_day_max - p",
    "p[d,z,j-1], NaN at j=0", "p[d,z,j+1], NaN at j=H_d-1",
    "p - dam_prev_hour", "dam_next_hour - p",
])) | {f"dam_peer_{z}": f"{z} DAM at identical UTC hour; NaN only when target zone is {z}" for z in ZONES} | {
    "dam_minus_peer_median": "p - median(seven finite other-zone DAM prices)",
    "dam_peer_std": "std(seven finite other-zone DAM prices, ddof=0)",
    "dam_peer_range": "max(seven peers) - min(seven peers)",
}
CODE = [Path(__file__), ROOT / "run_price_experiment.py", ROOT / "train_price_models.py",
        ROOT / "build_price_model_data.py"]
METRICS = ("p50_mae_usd_mwh", "mean_mse_usd_mwh_squared", "mean_pinball_usd_mwh")


def rel(path):
    return str(Path(path).relative_to(ROOT))


def key_hash(frame):
    return hashlib.sha256(frame[base.KEYS].to_csv(index=False).encode()).hexdigest()


def frame_hash(frame):
    metadata = [(name, str(dtype), list(frame[name].cat.categories) if isinstance(dtype, pd.CategoricalDtype)
                 else None) for name, dtype in frame.dtypes.items()]
    h = hashlib.sha256(json.dumps(metadata).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def process():
    return {"pid": os.getpid(), "command": [sys.executable, *sys.argv], "log": rel(BATCH / "run.log")}


def context_features(frame):
    """Fixed DAM-only transform; preserve the caller's exact row order and index."""
    dam = frame[[*base.KEYS, "delivery_date", "dam_spp_usd_mwh"]].copy()
    assert not dam[base.KEYS].duplicated().any(), "Duplicate zone-hour-quarter"
    assert not dam.isna().any().any() and np.isfinite(dam.dam_spp_usd_mwh).all(), "Nonfinite source DAM"
    assert set(dam.settlement_point.unique()) == set(ZONES), "Missing peer zone"
    local = dam.interval_start_utc.dt.tz_convert("America/Chicago")
    assert (local.dt.tz_localize(None).dt.normalize() == dam.delivery_date).all()
    assert local.dt.minute.isin([0, 15, 30, 45]).all() and (local.dt.second == 0).all()
    dam["hour_utc"] = dam.interval_start_utc.dt.floor("h")
    hourly_keys = ["delivery_date", "settlement_point", "hour_utc"]
    grouped = dam.groupby(hourly_keys, sort=True).dam_spp_usd_mwh
    audit = grouped.agg(["size", "nunique", "first"])
    assert (audit["size"] == 4).all(), "Incomplete hourly quarters"
    assert (audit["nunique"] == 1).all(), "Inconsistent hourly DAM"
    hours = audit["first"].rename("p").reset_index()
    assert (hours.groupby("hour_utc").size() == 8).all(), "Missing hourly peer"
    days = hours.groupby(["delivery_date", "settlement_point"], sort=False)
    counts = days.size()
    expected = counts.index.get_level_values(0).map(lambda d: base.expected_intervals(d.date()) // 4)
    assert np.array_equal(counts.to_numpy(), expected.to_numpy()), "Incomplete operating-day curve"
    hours["j"] = days.cumcount()
    hours["h"] = days.p.transform("size")
    for suffix, operation in (("min", "min"), ("max", "max"), ("mean", "mean")):
        hours[f"dam_day_{suffix}"] = days.p.transform(operation)
    hours["dam_day_std"] = days.p.transform("std", ddof=0)
    hours["negative"] = (hours.p < 0).astype(float)
    hours["dam_day_negative_share"] = days.negative.transform("mean")
    hours["peak_slot"] = hours.j.where(hours.p == hours.dam_day_max)
    hours["dam_hours_to_day_peak"] = days.peak_slot.transform("min") - hours.j
    hours["dam_above_day_min"] = hours.p - hours.dam_day_min
    hours["dam_below_day_max"] = hours.dam_day_max - hours.p
    hours["dam_prev_hour"] = days.p.shift(1)
    hours["dam_next_hour"] = days.p.shift(-1)
    hours["dam_ramp_from_prev"] = hours.p - hours.dam_prev_hour
    hours["dam_ramp_to_next"] = hours.dam_next_hour - hours.p
    panel = hours.pivot(index="hour_utc", columns="settlement_point", values="p").reindex(columns=ZONES)
    peers = panel.reindex(hours.hour_utc).to_numpy(copy=True)
    peers[np.arange(len(hours)), hours.settlement_point.map(ZONES).to_numpy()] = np.nan
    assert (np.isfinite(peers).sum(axis=1) == 7).all()
    hours[PEER[:8]] = peers
    hours["dam_minus_peer_median"] = hours.p.to_numpy() - np.nanmedian(peers, axis=1)
    hours["dam_peer_std"] = np.nanstd(peers, axis=1, ddof=0)
    hours["dam_peer_range"] = np.nanmax(peers, axis=1) - np.nanmin(peers, axis=1)
    for column in LOCAL + PEER:
        allowed = (hours.j == 0).to_numpy() if column in ("dam_prev_hour", "dam_ramp_from_prev") else (
            (hours.j == hours.h - 1).to_numpy() if column in ("dam_next_hour", "dam_ramp_to_next") else (
                (hours.settlement_point == column.removeprefix("dam_peer_")).to_numpy()
                if column in PEER[:8] else np.zeros(len(hours), dtype=bool)))
        assert np.array_equal(hours[column].isna().to_numpy(), allowed), f"Unexpected NaN: {column}"
        assert not np.isinf(hours[column]).any(), column
    joined = dam.merge(hours[hourly_keys + LOCAL + PEER], on=hourly_keys, how="left", validate="many_to_one", sort=False)
    assert len(joined) == len(frame) and joined[base.KEYS].equals(frame[base.KEYS].reset_index(drop=True))
    result = joined[LOCAL + PEER]
    result.index = frame.index
    return result


def design(frame, context, arm):
    x = base.design(frame, ZONES)
    x[BLOCKS[arm]] = context.loc[frame.index, BLOCKS[arm]]
    assert x.columns.tolist() == base.EXPECTED_FEATURES + BLOCKS[arm]
    return x


def fit_models(frame, context, start, end, arm, directory, run_prefix, contract, test_rounds=None):
    """The EXP001 two-stage loop, with explicit context columns and pre-fit receipts."""
    fit, stop, refit, _ = base.split(frame, start, end)
    x_fit, x_stop, x_refit = (design(part, context, arm) for part in (fit, stop, refit))
    y_fit, y_stop, y_refit = (base.residual(part) for part in (fit, stop, refit))
    models, selected, timings = {}, {}, {}
    inputs = {label: {"rows": len(part), "row_key_sha256": key_hash(part), "features_sha256": frame_hash(x),
                      "labels_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
              for label, part, x, y in (("inner_fit", fit, x_fit, y_fit), ("early_stop", stop, x_stop, y_stop),
                                        ("final_refit", refit, x_refit, y_refit))}
    for name, q in base.OUTPUTS.items():
        begun = time.monotonic()
        params = dict(base.PARAMS, **({"objective": "regression"} if q is None else {"objective": "quantile", "alpha": q}))
        if test_rounds is not None:
            params["n_estimators"] = test_rounds
        for stage in ("inner", "refit"):
            if stage == "refit":
                params = dict(params, n_estimators=selected[name])
            manifest_path = directory / name / f"{stage}_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            base.save_json(manifest_path, {"run_id": f"{run_prefix}/{name}/{stage}", "created_at_utc": base.now(),
                                          "approval": rel(PREFLIGHT), "preflight_sha256": base.digest(PREFLIGHT),
                                          "contract": contract, "configuration": params, "inputs": inputs,
                                          "features": x_fit.columns.tolist(), "process": process(),
                                          "guard_round_cap": test_rounds}, exclusive=True)
            model = LGBMRegressor(**params)
            if stage == "inner":
                model.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)], callbacks=[early_stopping(35, verbose=False)])
                selected[name] = int(model.best_iteration_)
                assert 1 <= selected[name] <= params["n_estimators"]
            else:
                model.fit(x_refit, y_refit)
                models[name] = model
            model_path = manifest_path.with_name(f"{stage}_model.txt")
            model.booster_.save_model(str(model_path))
            base.save_json(manifest_path.with_name(f"{stage}_result.json"), {
                "run_id": f"{run_prefix}/{name}/{stage}", "completed_at_utc": base.now(),
                "manifest_sha256": base.digest(manifest_path), "model_sha256": base.digest(model_path),
                "selected_iterations": selected[name]}, exclusive=True)
        timings[name] = round(time.monotonic() - begun, 4)
    return models, selected, timings


def prediction(models, x, offset):
    raw = {name: np.asarray(model.predict(x)) + offset for name, model in models.items()}
    values = np.column_stack([raw[name] for name in ("p10", "p50", "p90")])
    crossings = int(np.any(np.diff(values, axis=1) < 0, axis=1).sum())
    ordered = np.sort(values, axis=1)
    pred = dict(mean=raw["mean"], p10=ordered[:, 0], p50=ordered[:, 1], p90=ordered[:, 2])
    assert np.isfinite(np.column_stack(list(raw.values()))).all()
    return pred, raw, crossings


def daily_panel(frame):
    hours = frame[["delivery_date", "settlement_point", "interval_start_utc", "dam_spp_usd_mwh"]].copy()
    hours["hour_utc"] = hours.interval_start_utc.dt.floor("h")
    hours = hours.drop_duplicates(["settlement_point", "hour_utc"])
    return hours.groupby("delivery_date", sort=True).dam_spp_usd_mwh.median()


def donor_setup(frame, cutpoints=None):
    blocks, _ = base.day_blocks(frame)
    medians = daily_panel(frame)
    days = medians.index
    bins = np.searchsorted(cutpoints, medians.to_numpy(), side="right") if cutpoints is not None else None
    strata, keys = {}, []
    for i, (day, block) in enumerate(zip(days, blocks)):
        key = (day.month, len(block) // 32, *((int(bins[i]),) if bins is not None else ()))
        strata.setdefault(key, []).append(i)
        keys.append(key)
    return blocks, strata, keys, medians


def perturb(x, arm, frame, repeat, fold_index, cutpoints=None):
    blocks, strata, keys, medians = donor_setup(frame, cutpoints)
    rng = np.random.default_rng(np.random.SeedSequence([base.SEED, fold_index, repeat]))
    donors = np.arange(len(x))
    donor_days = np.arange(len(blocks))
    perturbable = np.zeros(len(x), dtype=bool)
    for members in strata.values():
        for target, source in zip(members, rng.permutation(members)):
            donors[blocks[target]] = blocks[source]
            donor_days[target] = source
            perturbable[blocks[target]] = len(members) > 1
    group = ["dam_asinh", *BLOCKS[arm]]
    xp = x.copy()
    xp[group] = x[group].iloc[donors].to_numpy()
    a, b = x[group].to_numpy(), xp[group].to_numpy()
    changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    assert np.array_equal(np.isnan(a), np.isnan(b)), "Structural NaN mask changed"
    for col in x.columns.difference(group):
        assert x[col].equals(xp[col]), f"Non-group input changed: {col}"
    mapping = [{"recipient": str(day.date()), "donor": str(medians.index[donor_days[i]].date()),
                "stratum": list(keys[i]), "stratum_days": len(strata[keys[i]]),
                "recipient_median": float(medians.iloc[i]), "donor_median": float(medians.iloc[donor_days[i]]),
                "donor_minus_recipient_median": float(medians.iloc[donor_days[i]] - medians.iloc[i]),
                "rows": len(blocks[i]), "self_donor": bool(donor_days[i] == i)}
               for i, day in enumerate(medians.index)]
    audit = {"donor_mapping": mapping, "non_self_donor_day_share": float(np.mean(donor_days != np.arange(len(blocks)))),
             "non_self_donor_row_share": float(np.mean(donors != np.arange(len(x)))),
             "perturbable_rows": int(perturbable.sum()), "unperturbable_rows": int((~perturbable).sum()),
             "perturbable_row_fraction": float(perturbable.mean()),
             "fraction_rows_changed": float(changed.any(axis=1).mean()), "fraction_cells_changed": float(changed.mean()),
             "fraction_columns_changed": float(changed.any(axis=0).mean()),
             "changed_fraction_by_column": {col: float(changed[:, i].mean()) for i, col in enumerate(group)},
             "unchanged_nan_cells": int((np.isnan(a) & np.isnan(b)).sum()),
             "stratum_sizes": [{"stratum": list(key), "days": len(value)} for key, value in strata.items()],
             "singleton_days": [m["recipient"] for m in mapping if m["stratum_days"] == 1],
             "actual_self_donor_rows": sum(m["rows"] for m in mapping if m["self_donor"])}
    return xp, audit, perturbable, donors


def fixture(days, price=None):
    pieces = []
    for day_index, day in enumerate(days):
        start = pd.Timestamp(day).tz_localize("America/Chicago")
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("America/Chicago")
        times = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
        times = times.repeat(8)
        local = times.tz_convert("America/Chicago")
        slots = np.repeat(np.arange(len(times) // 8) // 4, 8)
        zones = np.tile(np.arange(8), len(times) // 8)
        values = (30 + day_index * .3 + zones * 2 + 5 * np.sin(slots / 3)) if price is None else price(slots, zones, day_index)
        part = pd.DataFrame({"settlement_point": np.asarray(list(ZONES))[zones], "interval_start_utc": times,
                             "interval_end_utc": times + pd.Timedelta(minutes=15),
                             "delivery_date": local.tz_localize(None).normalize(), "hour_ending": local.hour + 1,
                             "quarter": local.minute // 15 + 1, "dam_spp_usd_mwh": values})
        repeated = pd.DataFrame({"zone": zones, "local": local.tz_localize(None)}).duplicated()
        part["repeated_hour_flag"] = np.where(repeated, "Y", "N")
        part["model_issue_utc"] = part.delivery_date.dt.tz_localize("UTC") - pd.Timedelta(hours=2)
        part["input_cutoff_utc"] = part.model_issue_utc
        part["rtm_spp_usd_mwh"] = part.dam_spp_usd_mwh + 3 * np.sin(slots / 5) + (day_index % 9) - 4
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def feature_guards():
    days = ["2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12", "2024-11-03"]
    toy = fixture(days, lambda j, z, d: j * 10. + z + d * 1000)
    x = context_features(toy)
    for day, h in (("2024-03-08", 24), ("2024-03-10", 23), ("2024-11-03", 25)):
        ids = toy.index[(toy.delivery_date == day) & (toy.settlement_point == "LZ_AEN") & (toy.quarter == 1)]
        values, part = toy.loc[ids, "dam_spp_usd_mwh"].to_numpy(), x.loc[ids]
        assert len(ids) == h
        np.testing.assert_allclose(part.dam_day_mean, values.mean())
        np.testing.assert_allclose(part.dam_day_std, values.std(ddof=0))
        np.testing.assert_allclose(part.dam_prev_hour.iloc[1:], values[:-1])
        np.testing.assert_allclose(part.dam_next_hour.iloc[:-1], values[1:])
        assert part.dam_prev_hour.isna().tolist() == [True] + [False] * (h - 1)
        assert part.dam_next_hour.isna().tolist() == [False] * (h - 1) + [True]
        np.testing.assert_allclose(part.dam_hours_to_day_peak, np.arange(h - 1, -1, -1))
    fall = toy[(toy.delivery_date == "2024-11-03") & (toy.settlement_point == "LZ_AEN") & (toy.quarter == 1)]
    assert fall[fall.hour_ending == 2].dam_spp_usd_mwh.nunique() == 2
    for constant in (0., -10.):
        equal = fixture(["2024-01-01"], lambda j, z, d: np.full(len(j), constant))
        xc = context_features(equal)
        assert (xc.dam_day_negative_share == int(constant < 0)).all()
        assert (xc[["dam_day_std", "dam_above_day_min", "dam_below_day_max", "dam_minus_peer_median",
                    "dam_peer_std", "dam_peer_range"]] == 0).all().all()
        assert (xc[["dam_ramp_from_prev", "dam_ramp_to_next"]].fillna(0) == 0).all().all()
        np.testing.assert_equal(xc.dam_hours_to_day_peak, -np.repeat(np.arange(24), 32))
    tied = fixture(["2024-01-01"], lambda j, z, d: np.choose(np.minimum(j, 3), [-10., 0., 10., 10.]))
    xt = context_features(tied)
    expected = np.r_[-10., 0., np.full(22, 10.)]
    np.testing.assert_allclose(xt.dam_day_std, expected.std(ddof=0))
    np.testing.assert_allclose(xt.dam_day_negative_share, 1 / 24)
    np.testing.assert_equal(xt.dam_hours_to_day_peak, 2 - np.repeat(np.arange(24), 32))
    np.testing.assert_allclose(xt.dam_above_day_min, tied.dam_spp_usd_mwh + 10)
    np.testing.assert_allclose(xt.dam_below_day_max, 10 - tied.dam_spp_usd_mwh)
    for i, zone in enumerate(ZONES):
        row = x.iloc[i]
        peers = np.delete(np.arange(8, dtype=float), i)
        assert np.isnan(row[f"dam_peer_{zone}"])
        np.testing.assert_equal(row[PEER[:8]].dropna().to_numpy(), peers)
        np.testing.assert_allclose([row.dam_minus_peer_median, row.dam_peer_std, row.dam_peer_range],
                                   [i - np.median(peers), peers.std(ddof=0), np.ptp(peers)])
    shuffled = toy.sample(frac=1, random_state=base.SEED)
    pd.testing.assert_frame_equal(context_features(shuffled).sort_index(), x)
    changed = toy.copy()
    changed["rtm_spp_usd_mwh"] += 1_000_000
    pd.testing.assert_frame_equal(context_features(changed), x)
    changed.loc[changed.delivery_date == "2024-03-09", "dam_spp_usd_mwh"] += 9999
    pd.testing.assert_frame_equal(context_features(changed).loc[toy.delivery_date == "2024-03-08"],
                                  x.loc[toy.delivery_date == "2024-03-08"])
    malformed = [pd.concat([toy, toy.iloc[[0]]], ignore_index=True), toy.drop(index=0),
                 toy[toy.settlement_point != "LZ_AEN"]]
    for value in (np.nan, np.inf, 987654.):
        bad = toy.copy()
        bad.loc[0, "dam_spp_usd_mwh"] = value
        malformed.append(bad)
    for bad in malformed:
        try:
            context_features(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Malformed DAM input accepted")
    for arm in BLOCKS:
        encoded = design(toy, x, arm)
        for cutpoints in (None, np.array([1500., 3500.])):
            xp, audit, _, donors = perturb(encoded, arm, toy, 0, 0, cutpoints)
            group = ["dam_asinh", *BLOCKS[arm]]
            np.testing.assert_allclose(xp[group], encoded[group].iloc[donors], equal_nan=True)
            assert toy.iloc[donors].settlement_point.reset_index(drop=True).equals(toy.settlement_point)
            assert toy.iloc[donors].quarter.reset_index(drop=True).equals(toy.quarter)
            assert toy.iloc[donors].repeated_hour_flag.reset_index(drop=True).equals(toy.repeated_hour_flag)
            assert {"2024-03-10", "2024-11-03"}.issubset(audit["singleton_days"])
            for mapping in audit["donor_mapping"]:
                recipient, donor = pd.Timestamp(mapping["recipient"]), pd.Timestamp(mapping["donor"])
                assert recipient.month == donor.month
                assert base.expected_intervals(recipient.date()) == base.expected_intervals(donor.date())
                if cutpoints is not None:
                    assert np.searchsorted(cutpoints, mapping["recipient_median"], side="right") == np.searchsorted(
                        cutpoints, mapping["donor_median"], side="right")
            assert audit["unchanged_nan_cells"] == int(encoded[group].isna().sum().sum())
            np.testing.assert_equal(toy.dam_spp_usd_mwh.to_numpy(), toy.dam_spp_usd_mwh.to_numpy().copy())
    return {"status": "passed", "actual_hour_counts": [23, 24, 25], "dst_chronology_and_distinct_fall_hours": True,
            "boundary_nan_masks": True, "constant_zero_negative_and_tied_peak": True, "seven_peer_exact_values": True,
            "strict_negative_population_std_extreme_signs": True, "row_reordering_invariant": True,
            "label_mutation_invariant": True, "next_operating_day_isolation": True,
            "bad_source_cases_rejected": len(malformed), "coherent_gpi_cpi_maps": True,
            "singleton_dst_unmoved": True, "nan_to_nan_unchanged": True}


def input_contract(prices):
    frozen = json.loads((base.OUT / "manifest.json").read_text())
    assert base.digest(base.TABLE) == base.EXPECTED_HASH
    assert base.digest(base.OUT / "manifest.json") == "6dfebca8891a898a048200dd167e12ed8468ed3d5b8e48ed78c4a9eaa97a41ca"
    for name in ("run_price_experiment.py", "train_price_models.py", "data/price/source_audit.json"):
        assert base.digest(ROOT / name) == frozen["hashes_sha256"][name], f"EXP001 dependency changed: {name}"
    assert frozen["features"] == base.EXPECTED_FEATURES and frozen["configuration"] == base.PARAMS
    assert frozen["zone_codes"] == ZONES and frozen["objectives"] == base.OUTPUTS
    for fold, spec in zip(base.FOLDS, frozen["folds"]):
        name, start, end, *counts = fold
        assert [name, start, end] == [spec["fold"], spec["score_start_inclusive"], spec["score_end_exclusive"]]
        actual = [len(part) for part in base.split(prices, start, end)]
        assert actual == counts == [spec["parts"][p]["rows"] for p in ("inner_fit", "early_stop", "final_refit", "outer_score")]
    baseline = base.read_predictions(base.OUT / "predictions.csv.gz")
    base.verify_predictions(baseline, prices)
    for fold, spec in zip(base.FOLDS, frozen["folds"]):
        assert len(baseline[baseline.fold == fold[0]]) == spec["parts"]["outer_score"]["rows"]
    paths = [*CODE, base.TABLE, PREFLIGHT, ROOT / "data/price/source_audit.json",
             ROOT / "data/ercot/source_manifest.json", base.OUT / "manifest.json", base.OUT / "predictions.csv.gz"]
    hashes = {rel(path): base.digest(path) for path in paths}
    return frozen, baseline, {"hashes_sha256": hashes, "row_key_sha256": key_hash(baseline), "cohort_id": COHORT,
                              "baseline_code_snapshot_hashes": {rel(p): base.digest(p) for p in (base.OUT / "code_snapshot").glob("*.py")},
                              "builder_provenance_note": "Current builder differs from EXP001 snapshot only in 2026 audit filename isolation; see EXP001_independent_review.md. Shared model helpers match exactly."}


def check_contract(contract):
    for path, expected in contract["hashes_sha256"].items():
        assert base.digest(ROOT / path) == expected, f"Immutable contract changed: {path}"


def guards(prices, contract):
    result = {"source_validation": base.validate_prices(prices), "feature_and_donor_guards": feature_guards()}
    context = context_features(prices)
    assert context.index.equals(prices.index)
    result["source_feature_audit"] = {"rows": len(context), "zone_hours": len(prices) // 4,
                                      "no_rows_dropped": True, "producer_missing_values": 0,
                                      "structural_nan_counts": context.isna().sum().to_dict(),
                                      "context_matrix_sha256": frame_hash(context)}
    stamp = base.now().replace(":", "").replace("+", "_")
    toy = fixture(pd.date_range("2023-11-24", "2024-01-03", freq="D"))
    toy_context = context_features(toy)
    result["fit_guards"] = {}
    for arm in BLOCKS:
        path = OUT / arm / "guard" / stamp
        path.mkdir(parents=True, exist_ok=False)
        (path / "code_snapshot").mkdir()
        for code in CODE:
            shutil.copy2(code, path / "code_snapshot" / code.name)
        joblib.dump({"rows": toy, "features": toy_context}, path / "fixture.joblib", compress=3)
        guard_contract = dict(contract, fixture_sha256=base.digest(path / "fixture.joblib"))
        original = fit_models(toy, toy_context, "2024-01-01", "2024-04-01", arm, path / "original",
                              f"{arm}/guard/{stamp}/original", guard_contract, test_rounds=40)
        changed = toy.copy()
        changed.loc[changed.delivery_date >= "2024-01-01", "rtm_spp_usd_mwh"] += 1_000_000
        altered = fit_models(changed, toy_context, "2024-01-01", "2024-04-01", arm, path / "outer_labels_mutated",
                             f"{arm}/guard/{stamp}/outer_labels_mutated", guard_contract, test_rounds=40)
        assert original[1] == altered[1], "Outer labels changed iteration choice"
        assert all(original[0][k].booster_.model_to_string() == altered[0][k].booster_.model_to_string()
                   for k in base.OUTPUTS), "Outer labels changed fitted model"
        xd = design(toy, toy_context, arm)
        xp, _, _, _ = perturb(xd, arm, toy, 0, 0)
        offset = toy.dam_spp_usd_mwh.to_numpy(copy=True)
        _, raw, _ = prediction(original[0], xp, offset)
        for key, model in original[0].items():
            np.testing.assert_allclose(raw[key] - model.predict(xp), offset, rtol=0, atol=1e-10)
        np.testing.assert_equal(offset, toy.dam_spp_usd_mwh.to_numpy())
        result["fit_guards"][arm] = {"status": "passed", "all_four_model_bodies_and_iterations_identical": True,
                                     "permutation_original_offset_unchanged": True, "round_cap": 40,
                                     "selected_guard_iterations": original[1], "artifact_path": rel(path)}
    bad = toy.copy()
    bad.loc[bad.delivery_date == "2023-12-30", "interval_end_utc"] = pd.Timestamp("2024-01-02", tz="UTC")
    try:
        base.split(bad, "2024-01-01", "2024-04-01")
    except AssertionError:
        result["injected_embargo_violation"] = "rejected"
    else:
        raise AssertionError("Injected embargo breach accepted")
    check_contract(contract)
    result["status"] = "passed"
    context_path = BATCH / "guards" / f"{stamp}_context.joblib"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(context, context_path, compress=3)
    receipt = dict(result, contract=contract, context_path=rel(context_path), context_file_sha256=base.digest(context_path))
    receipt_path = BATCH / "guards" / f"{stamp}.json"
    base.save_json(receipt_path, receipt, exclusive=True)
    base.save_json(BATCH / "guards" / "latest.json", {"receipt": rel(receipt_path), "sha256": base.digest(receipt_path)})
    return context, result


def make_manifest(arm, frozen, contract, checks):
    features = base.EXPECTED_FEATURES + BLOCKS[arm]
    return {"experiment_id": arm, "parent": "EXP001", "created_at_utc": base.now(), "hypothesis": HYPOTHESES[arm],
            "preflight_path": rel(PREFLIGHT), "approval": "Prespecified master data scientist batch approval, preflight 002",
            **contract, "features": features, "feature_columns": features,
            "ordered_feature_blocks": {"B": base.EXPECTED_FEATURES,
                                       **({"L": LOCAL} if arm != "EXP003" else {}),
                                       **({"C": PEER} if arm != "EXP002" else {})},
            "feature_formulas": {name: FORMULAS[name] for name in BLOCKS[arm]},
            "feature_units": {name: "share" if name == "dam_day_negative_share" else "elapsed hours" if name == "dam_hours_to_day_peak"
                              else "USD/MWh" for name in BLOCKS[arm]},
            "baseline_feature_definition": "Exact EXP001 matrix including float32 asinh(DAM/50), hour midpoints, and category ordering",
            "categorical": base.CATS, "zone_codes": ZONES,
            "category_vocabulary": {"zone_code": list(ZONES.values()), "weekday": list(range(7)), "quarter": [1, 2, 3, 4], "repeated": [0, 1]},
            "row_keys": base.KEYS, "folds": frozen["folds"], "total_outer_rows": frozen["total_outer_rows"],
            "configuration": base.PARAMS, "objectives": base.OUTPUTS, "early_stopping_rounds": 35,
            "training_sample": "all eligible rows", "sample_weight": None, "hyperparameter_search": False,
            "target_clipping": False, "target": "rtm_spp_usd_mwh - dam_spp_usd_mwh", "seed": base.SEED,
            "issue_rule": frozen["issue_rule"], "data_vintage_caveats": frozen["data_vintage_caveats"] + [
                "All target-day DAM hours assumed available by cutoff; historical DAM revisions unverified",
                "No 2026 rows accessed, scored, or used for feature selection in this batch; earlier aggregate exposure remains disclosed"],
            "preprocessing": "Fixed DAM-only context; no clipping, normalization, imputation, extra transforms, or new categories",
            "package_versions": {name: importlib.metadata.version(name) for name in frozen["package_versions"]},
            "process": process(), "checks_before_fit": checks,
            "diagnostic_technique_ids": [f"{arm}-GPI-v1"] + (["EXP004-CPI-v1"] if arm == "EXP004" else [])}


def full_scores(frame, method):
    if not len(frame):
        return {"n": 0, "p50_mae_usd_mwh": None, "mean_mse_usd_mwh_squared": None, "mean_rmse_usd_mwh": None,
                "mean_pinball": None, "mean_bias_usd_mwh": None, "pinball_p10": None, "pinball_p50": None,
                "pinball_p90": None, "p10_p90_coverage": None, "p10_p90_mean_width_usd_mwh": None,
                "below_p10": None, "above_p90": None, "high_spike_n": 0, "negative_price_n": 0,
                "high_spike_p90_recall_at_200": None, "high_spike_p90_precision_at_200": None,
                "high_spike_false_alarms_n": 0, "high_spike_false_alarm_rate": None,
                "event_dates": [], "realized_negative_day_count": 0, "negative_row_quantiles": None}
    result = base.scores(frame, method)
    p = base.get_pred(frame, method)
    y = frame.rtm_spp_usd_mwh.to_numpy()
    neg, high, alarm = y < 0, y >= 200, p["p90"] >= 200
    result.update({"high_spike_true_alarms_n": int((high & alarm).sum()),
                   "high_spike_missed_n": int((high & ~alarm).sum()),
                   "high_spike_mean_rmse_usd_mwh": float(np.sqrt(np.mean((p["mean"][high] - y[high]) ** 2))) if high.any() else None,
                   "negative_price_mean_rmse_usd_mwh": float(np.sqrt(np.mean((p["mean"][neg] - y[neg]) ** 2))) if neg.any() else None,
                   "realized_negative_day_count": int(frame.loc[neg, "delivery_date"].nunique()),
                   "negative_price_delivery_dates": sorted(frame.loc[neg, "delivery_date"].dt.strftime("%Y-%m-%d").unique().tolist()),
                   "event_dates": sorted(frame.delivery_date.dt.strftime("%Y-%m-%d").unique().tolist()),
                   "negative_row_quantiles": {key: {"prediction_mean_usd_mwh": float(p[key][neg].mean()),
                                                    "prediction_median_usd_mwh": float(np.median(p[key][neg])),
                                                    "fraction_prediction_below_zero": float((p[key][neg] < 0).mean())}
                                              for key in ("p10", "p50", "p90")} if neg.any() else None})
    return result


def slice_masks(frame):
    y = frame.rtm_spp_usd_mwh.to_numpy()
    return {"all": np.ones(len(frame), dtype=bool), "high_price": y >= 200,
            "negative_price": y < 0, "ordinary_price": (y >= 0) & (y < 200)}


def arm_metrics(saved, fits):
    methods = ("lightgbm", "EXP001", *base.BASELINES)
    groups = {"fold": saved.fold, "zone": saved.settlement_point,
              "month": saved.delivery_date.dt.strftime("%Y-%m"),
              "lead_hour": ((saved.interval_start_utc - saved.model_issue_utc).dt.total_seconds() // 3600).astype(int),
              "day_hours": saved.operating_day_hours}
    slices = {kind: {str(value): {method: full_scores(saved.loc[ids], method) for method in methods}
                      for value, ids in values.groupby(values).groups.items()} for kind, values in groups.items()}
    slices["price_regime"] = {name: {method: full_scores(saved.loc[mask], method) for method in methods}
                              for name, mask in slice_masks(saved).items() if name != "all"}
    raw = saved[[f"lightgbm_raw_{q}" for q in ("p10", "p50", "p90")]].to_numpy()
    return {"pooled": {method: full_scores(saved, method) for method in methods}, "slices": slices,
            "model_training": fits, "raw_quantiles": {"crossing_rows": int(np.any(np.diff(raw, axis=1) < 0, axis=1).sum()),
                                                       "crossing_fraction": float(np.any(np.diff(raw, axis=1) < 0, axis=1).mean()),
                                                       "scores_before_sort": full_scores(saved, "lightgbm_raw")},
            "raw_dam_distribution_note": "Point price repeated for same-row diagnostics; raw DAM has no predictive distribution",
            "interpretation": "Median and distribution measures are separate from mean MSE/RMSE and tail alarms; P50 pinball equals half P50 MAE. No calibrated negative or spike probability and no revenue claim."}


def loss_values(y, pred):
    return {"p50_mae_usd_mwh": np.abs(y - pred["p50"]), "mean_mse_usd_mwh_squared": (y - pred["mean"]) ** 2,
            "mean_pinball_usd_mwh": np.mean([np.maximum(q * (y - pred[name]), (q - 1) * (y - pred[name]))
                                              for name, q in base.OUTPUTS.items() if q is not None], axis=0)}


def importance(arm, manifest, metrics):
    out = OUT / arm
    records = []
    for kind in (["GPI", "CPI"] if arm == "EXP004" else ["GPI"]):
        technique = f"{arm}-{kind}-v1"
        directory = out / technique
        directory.mkdir(parents=True, exist_ok=False)
        inputs = {rel(path): base.digest(path) for fold, *_ in base.FOLDS for path in (
            out / "folds" / fold / "models.joblib", out / "folds" / fold / "input_rows.joblib",
            out / "folds" / fold / "outer_features.joblib", out / "folds" / fold / "predictions.csv.gz")}
        base.save_json(directory / "manifest.json", {
            "technique_id": technique, "created_at_utc": base.now(), "preflight_sha256": base.digest(PREFLIGHT),
            "experiment_manifest_sha256": base.digest(out / "manifest.json"), "hashes_sha256": inputs,
            "feature_builder_sha256": base.digest(Path(__file__)), "group": ["dam_asinh", *BLOCKS[arm]],
            "repeats": 10, "seed_rule": "SeedSequence([43, zero_based_fold_index, zero_based_repeat])",
            "strata": ["calendar_month", "operating_day_hours"] + (["training_daily_level_bin"] if kind == "CPI" else []),
            "original_additive_dam_offset_fixed": True, "process": process(), "output_path": rel(directory),
            "interpretation": "Coarse conditional grouped sensitivity given calendar month, day shape, and a daily level bin; approximate restricted donors, not exact conditional importance" if kind == "CPI"
                              else "Frozen residual-correction reliance on the coherent DAM source group"}, exclusive=True)
        for fold_index, (fold, *_rest) in enumerate(base.FOLDS):
            fold_dir = out / "folds" / fold
            (directory / fold).mkdir()
            rows = joblib.load(fold_dir / "input_rows.joblib")
            x = joblib.load(fold_dir / "outer_features.joblib")
            bundle = joblib.load(fold_dir / "models.joblib")
            pd.testing.assert_frame_equal(x, design(rows, context_features(rows), arm))
            frozen = base.read_predictions(fold_dir / "predictions.csv.gz")
            original = base.losses(frozen, "lightgbm")
            cutpoints = np.asarray(bundle["cpi_cutpoints"]) if kind == "CPI" else None
            for repeat in range(10):
                check_contract(manifest)
                xp, audit, perturbable, _ = perturb(x, arm, rows, repeat, fold_index, cutpoints)
                run_id = f"{technique}/{fold}/repeat{repeat:02}"
                path = directory / fold / f"repeat{repeat:02}.json"
                base.save_json(path.with_name(f"repeat{repeat:02}_manifest.json"), {
                    "run_id": run_id, "technique_id": technique, "created_at_utc": base.now(),
                    "technique_manifest_sha256": base.digest(directory / "manifest.json"),
                    "model_sha256": base.digest(fold_dir / "models.joblib"), "input_sha256": base.digest(fold_dir / "input_rows.joblib"),
                    "features_sha256": base.digest(fold_dir / "outer_features.joblib"), "seed": [base.SEED, fold_index, repeat],
                    "cutpoints": cutpoints.tolist() if cutpoints is not None else None,
                    "bin_training_rule": bundle["cpi_bin_training"] if kind == "CPI" else None,
                    "donors_and_changed_fractions": audit, "output_path": rel(path), "process": process()}, exclusive=True)
                pred, _, crossings = prediction(bundle["models"], xp, rows.dam_spp_usd_mwh.to_numpy())
                trial = loss_values(frozen.rtm_spp_usd_mwh.to_numpy(), pred)
                changes = {key: float(trial[key].mean() - original[key].mean()) for key in METRICS}
                changes["mean_rmse_usd_mwh"] = float(np.sqrt(trial[METRICS[1]].mean()) - np.sqrt(original[METRICS[1]].mean()))
                subset = {key: {"original": float(original[key][perturbable].mean()),
                                "perturbed": float(trial[key][perturbable].mean()),
                                "increase": float((trial[key][perturbable] - original[key][perturbable]).mean())}
                          for key in METRICS} if perturbable.any() else None
                record = {"run_id": run_id, "technique_id": technique, "fold": fold, "repeat": repeat,
                          "rows": len(rows), "loss_increases": changes, "raw_quantile_crossings": crossings,
                          "perturbable_only": {"rows": int(perturbable.sum()), "identical_unperturbed_subset": True, "metrics": subset},
                          "audit": audit}
                base.save_json(path, record, exclusive=True)
                records.append(record)
            print(f"{base.now()} {arm} {fold} {kind} completed 10 inference repeats", flush=True)
    summary = {}
    for kind in (["GPI", "CPI"] if arm == "EXP004" else ["GPI"]):
        technique = f"{arm}-{kind}-v1"
        selected = [r for r in records if r["technique_id"] == technique]
        pooled = []
        for repeat in range(10):
            items = [r for r in selected if r["repeat"] == repeat]
            assert len(items) == 6
            values = {key: sum(r["loss_increases"][key] * r["rows"] for r in items) / sum(r["rows"] for r in items)
                      for key in METRICS}
            mse = metrics["pooled"]["lightgbm"]["mean_mse_usd_mwh_squared"]
            values["mean_rmse_usd_mwh"] = float(np.sqrt(mse + values[METRICS[1]]) - np.sqrt(mse))
            values["fraction_rows_changed"] = sum(r["audit"]["fraction_rows_changed"] * r["rows"] for r in items) / sum(r["rows"] for r in items)
            pooled.append(values)
        summarize = lambda items: {key: {"mean": float(np.mean([r[key] for r in items])),
                                        "monte_carlo_std": float(np.std([r[key] for r in items], ddof=1)),
                                        "minimum": float(min(r[key] for r in items)), "maximum": float(max(r[key] for r in items))}
                                  for key in items[0]}
        summary[technique] = {"pooled_repeats": pooled, "pooled": summarize(pooled),
                              "by_fold": {fold: summarize([r["loss_increases"] for r in selected if r["fold"] == fold])
                                          for fold, *_ in base.FOLDS},
                              "audit_by_fold_repeat": [{"fold": r["fold"], "repeat": r["repeat"],
                                                        **{k: v for k, v in r["audit"].items() if k != "donor_mapping"}} for r in selected]}
    output = {"estimand": "Residual-correction reliance; original additive DAM offset fixed, dam_asinh moves jointly with all context",
              "interpretation": "Repeat standard deviations and ranges describe Monte Carlo variation, not a population confidence interval. CPI is approximate coarse conditional grouped sensitivity; refit contrasts measure incremental blocks.",
              "techniques": summary}
    base.save_json(out / "importance.json", output, exclusive=True)
    return output


def verify_arm(arm, prices, baseline):
    out = OUT / arm
    manifest = json.loads((out / "manifest.json").read_text())
    check_contract(manifest)
    saved = base.read_predictions(out / "predictions.csv.gz")
    result = base.verify_predictions(saved, prices)
    assert key_hash(saved) == manifest["row_key_sha256"] == key_hash(baseline)
    for col in ["fold", *base.ROW_COLS]:
        if col in ("dam_spp_usd_mwh", "rtm_spp_usd_mwh"):
            np.testing.assert_allclose(saved[col], baseline[col], rtol=0, atol=1e-9)
        else:
            assert saved[col].equals(baseline[col]), col
    for method in base.BASELINES:
        for key in base.OUTPUTS:
            np.testing.assert_allclose(saved[f"{method}_{key}"], baseline[f"{method}_{key}"], rtol=0, atol=1e-9)
    for key in base.OUTPUTS:
        np.testing.assert_allclose(saved[f"EXP001_{key}"], baseline[f"lightgbm_{key}"], rtol=0, atol=1e-9)
    metrics = json.loads((out / "metrics.json").read_text())
    actual = arm_metrics(saved, metrics["model_training"])
    assert actual == metrics, "Saved metrics differ from reloaded prediction scores"
    artifacts = []
    for fold, *_ in base.FOLDS:
        directory = out / "folds" / fold
        fit_record = json.loads((directory / "fit.json").read_text())
        for name, expected in fit_record["artifact_hashes_sha256"].items():
            assert base.digest(ROOT / name) == expected, name
        bundle = joblib.load(directory / "models.joblib")
        assert bundle["manifest_sha256"] == base.digest(out / "manifest.json")
        assert bundle["source_sha256"] == base.EXPECTED_HASH
        assert bundle["features"] == manifest["features"]
        rows, x = joblib.load(directory / "input_rows.joblib"), joblib.load(directory / "outer_features.joblib")
        pd.testing.assert_frame_equal(x, design(rows, context_features(rows), arm))
        p, raw, crosses = prediction(bundle["models"], x, rows.dam_spp_usd_mwh.to_numpy())
        forecast = base.read_predictions(directory / "predictions.csv.gz")
        for key in base.OUTPUTS:
            np.testing.assert_allclose(p[key], forecast[f"lightgbm_{key}"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(raw[key], forecast[f"lightgbm_raw_{key}"], rtol=0, atol=1e-9)
        assert crosses == fit_record["raw_quantile_crossings"]
        for key in base.OUTPUTS:
            for stage in ("inner", "refit"):
                receipt_path = directory / "fits" / key / f"{stage}_result.json"
                receipt = json.loads(receipt_path.read_text())
                assert base.digest(receipt_path.with_name(f"{stage}_manifest.json")) == receipt["manifest_sha256"]
                assert base.digest(receipt_path.with_name(f"{stage}_model.txt")) == receipt["model_sha256"]
        artifacts.append({"fold": fold, "models_reloaded": True, "raw_and_sorted_predictions_replayed": True,
                          "feature_matrix_rebuilt_identically": True, "artifact_hashes_verified": True})
    for path in (out / "code_snapshot").glob("*.py"):
        assert base.digest(path) == manifest["hashes_sha256"][path.name]
    result.update({"cohort_id": COHORT, "row_key_sha256": key_hash(saved), "benchmarks_and_EXP001_verified_same_rows": True,
                   "saved_metrics_recomputed": True, "source_feature_code_and_models_verified": True, "folds": artifacts,
                   "prediction_sha256": base.digest(out / "predictions.csv.gz")})
    return result


def run_arm(arm, prices, context, baseline, manifest, record):
    begun = time.monotonic()
    out = OUT / arm
    frames, fits = [], []
    try:
        for fold, start, end, *_ in base.FOLDS:
            check_contract(manifest)
            directory = out / "folds" / fold
            directory.mkdir(parents=True, exist_ok=False)
            print(f"{base.now()} fitting {arm} {fold}", flush=True)
            _, _, refit, outer = base.split(prices, start, end)
            x = design(outer, context, arm).reset_index(drop=True)
            rows = outer.reset_index(drop=True)
            joblib.dump(rows, directory / "input_rows.joblib", compress=3)
            joblib.dump(x, directory / "outer_features.joblib", compress=3)
            models, iterations, timing = fit_models(prices, context, start, end, arm, directory / "fits",
                                                    f"{arm}/{fold}", {"arm_manifest_sha256": base.digest(out / "manifest.json"),
                                                                       **{k: manifest[k] for k in ("hashes_sha256", "row_key_sha256", "cohort_id")}})
            medians = daily_panel(refit)
            cutpoints = np.unique(np.quantile(medians, [1 / 3, 2 / 3], method="linear"))
            bundle = {"models": models, "features": manifest["features"], "zone_codes": ZONES, "fold": fold,
                      "manifest_sha256": base.digest(out / "manifest.json"), "source_sha256": base.EXPECTED_HASH,
                      "feature_builder_sha256": base.digest(Path(__file__)), "selected_iterations": iterations,
                      "cpi_cutpoints": cutpoints.tolist(), "cpi_bin_training": {"days": len(medians),
                        "delivery_start": str(medians.index.min().date()), "delivery_end": str(medians.index.max().date()),
                        "refit_row_key_sha256": key_hash(refit), "daily_medians_sha256": frame_hash(medians.to_frame()),
                        "quantiles": [1 / 3, 2 / 3], "quantile_method": "linear", "searchsorted_side": "right",
                        "geography_note": "Unweighted settlement-price descriptor, not a physical or load-weighted system price"}}
            joblib.dump(bundle, directory / "models.joblib", compress=3)
            bundle = joblib.load(directory / "models.joblib")
            pred, raw, crossings = prediction(bundle["models"], joblib.load(directory / "outer_features.joblib"), rows.dam_spp_usd_mwh.to_numpy())
            reference = baseline[baseline.fold == fold].reset_index(drop=True)
            assert reference[base.KEYS].equals(rows[base.KEYS])
            assert ((reference.delivery_date >= start) & (reference.delivery_date < end)).all()
            np.testing.assert_allclose(reference[["rtm_spp_usd_mwh", "dam_spp_usd_mwh"]],
                                       rows[["rtm_spp_usd_mwh", "dam_spp_usd_mwh"]], rtol=0, atol=1e-9)
            saved = rows[base.ROW_COLS].copy()
            saved["fold"] = fold
            saved["operating_day_hours"] = rows.groupby(["delivery_date", "settlement_point"]).interval_start_utc.transform("size") // 4
            saved["chronological_hour_index"] = rows.groupby(["delivery_date", "settlement_point"]).cumcount() // 4
            forecasts = {"lightgbm": pred, "lightgbm_raw": raw, "EXP001": base.get_pred(reference, "lightgbm"),
                         **{method: base.get_pred(reference, method) for method in base.BASELINES}}
            for method, outputs in forecasts.items():
                for key, values in outputs.items():
                    saved[f"{method}_{key}"] = values
            saved.to_csv(directory / "predictions.csv.gz", index=False, compression="gzip")
            saved = base.read_predictions(directory / "predictions.csv.gz")
            artifact_hashes = {rel(directory / name): base.digest(directory / name) for name in
                               ("models.joblib", "input_rows.joblib", "outer_features.joblib", "predictions.csv.gz")}
            fit_record = {"fold": fold, "selected_iterations": iterations, "objective_fit_seconds": timing,
                          "raw_quantile_crossings": crossings, "raw_quantile_crossing_fraction": crossings / len(saved),
                          "artifact_hashes_sha256": artifact_hashes, "scores": full_scores(saved, "lightgbm")}
            base.save_json(directory / "fit.json", fit_record, exclusive=True)
            frames.append(saved)
            fits.append(fit_record)
            record["completed_folds"].append(fold)
            base.save_json(out / "record.json", record)
            print(f"{base.now()} {arm} {fold}: iterations {iterations}; P50 MAE {fit_record['scores']['p50_mae_usd_mwh']:.6f}; mean RMSE {fit_record['scores']['mean_rmse_usd_mwh']:.6f}", flush=True)
        saved = pd.concat(frames, ignore_index=True)
        saved.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
        saved = base.read_predictions(out / "predictions.csv.gz")
        assert key_hash(saved) == manifest["row_key_sha256"]
        metrics = arm_metrics(saved, fits)
        base.save_json(out / "metrics.json", metrics, exclusive=True)
        chart = []
        for kind, values in {"pooled": {"all": metrics["pooled"]}, **metrics["slices"]}.items():
            for value, methods in values.items():
                for method, scores in methods.items():
                    chart.append({"experiment_id": arm, "slice_type": kind, "slice_value": value, "method": method,
                                  **{key: v for key, v in scores.items() if not isinstance(v, (list, dict))}})
        pd.DataFrame(chart).to_csv(out / "chart_metrics.csv", index=False)
        base.save_json(out / "verification.json", verify_arm(arm, prices, baseline), exclusive=True)
        record.update(status="diagnostics_running", pooled_metrics=metrics["pooled"]["lightgbm"],
                      baselines={method: metrics["pooled"][method] for method in base.BASELINES},
                      comparison_rows=[{"model": method, "metrics": values, "baseline_id": None if method == "lightgbm" else method,
                                        "cohort_id": COHORT, "row_key_sha256": manifest["row_key_sha256"]}
                                       for method, values in metrics["pooled"].items()])
        base.save_json(out / "record.json", record)
        importance(arm, manifest, metrics)
        record.update(status="awaiting_paired_batch", elapsed_seconds=round(time.monotonic() - begun, 3))
        base.save_json(out / "record.json", record)
    except BaseException as error:
        record.update(status="failed", failed_at_utc=base.now(), error=f"{type(error).__name__}: {error}")
        base.save_json(out / "record.json", record)
        raise


def paired_batch():
    frames = {arm: base.read_predictions(OUT / arm / "predictions.csv.gz") for arm in ("EXP001", *BLOCKS)}
    frame = frames["EXP001"]
    ordered_hash = key_hash(frame)
    for arm, saved in frames.items():
        assert key_hash(saved) == ordered_hash and saved.fold.equals(frame.fold)
        np.testing.assert_allclose(saved[["rtm_spp_usd_mwh", "dam_spp_usd_mwh"]],
                                   frame[["rtm_spp_usd_mwh", "dam_spp_usd_mwh"]], rtol=0, atol=1e-9)
    loss = {arm: base.losses(saved, "lightgbm") for arm, saved in frames.items()}
    loss.update({method: base.losses(frame, method) for method in base.BASELINES})
    contrasts = {f"{arm}_minus_{reference}": {arm: 1, reference: -1}
                 for arm in BLOCKS for reference in ("EXP001", *base.BASELINES)}
    contrasts.update({"EXP004_minus_EXP003": {"EXP004": 1, "EXP003": -1},
                      "EXP004_minus_EXP002": {"EXP004": 1, "EXP002": -1},
                      "interaction": {"EXP004": 1, "EXP002": -1, "EXP003": -1, "EXP001": 1}})
    plans = {}
    metadata = {}
    for scheme, labels in (("whole_delivery_week", frame.delivery_date.dt.to_period("W-SUN")),
                            ("moving_seven_day", frame.delivery_date)):
        codes, dates = pd.factorize(labels, sort=True)
        rng = np.random.default_rng(base.SEED)
        if scheme == "whole_delivery_week":
            picked = rng.integers(0, len(dates), size=(3000, len(dates)))
            unique_days = frame[["delivery_date"]].drop_duplicates()
            week_sizes = unique_days.groupby(unique_days.delivery_date.dt.to_period("W-SUN")).size()
            metadata[scheme] = {"delivery_days": len(unique_days), "resampling_groups": len(dates), "draws": 3000,
                                "full_seven_day_groups": int((week_sizes == 7).sum()),
                                "partial_boundary_groups": {str(day): int(n) for day, n in week_sizes.items() if n != 7},
                                "groups_per_replicate": len(dates)}
        else:
            assert (pd.Series(dates).diff().dropna() == pd.Timedelta(days=1)).all()
            starts = rng.integers(0, len(dates) - 6, size=(3000, (len(dates) + 6) // 7))
            picked = (starts[:, :, None] + np.arange(7)).reshape(3000, -1)[:, :len(dates)]
            metadata[scheme] = {"delivery_days": len(dates), "block_length_days": 7, "possible_block_starts": len(dates) - 6,
                                "blocks_per_replicate": (len(dates) + 6) // 7, "draws": 3000,
                                "sampled_days_after_truncation": len(dates)}
        path = BATCH / f"{scheme}_resamples.npz"
        np.savez_compressed(path, group_codes=codes, picked=picked)
        plans[scheme] = (codes, dates, picked)
        metadata[scheme]["artifact_path"] = rel(path)
        metadata[scheme]["sha256"] = base.digest(path)
    manifest = {"technique_id": "BATCH002-PAIRED-v1", "created_at_utc": base.now(), "preflight_sha256": base.digest(PREFLIGHT),
                "cohort_id": COHORT, "row_key_sha256": ordered_hash, "seed": base.SEED, "draws": 3000,
                "prediction_hashes_sha256": {rel(OUT / arm / "predictions.csv.gz"): base.digest(OUT / arm / "predictions.csv.gz") for arm in frames},
                "contrasts": contrasts, "shared_resampling_plans": metadata, "process": process(),
                "interpretation": "Candidate-minus-reference paired development-sample loss differences; negative is lower loss. Percentile intervals are not multiplicity-corrected confirmation. Interaction is descriptive learner performance, not causal."}
    base.save_json(BATCH / "paired_manifest.json", manifest, exclusive=True)
    results = {name: {"coefficients": weights, "slices": {}} for name, weights in contrasts.items()}
    for slice_name, mask in slice_masks(frame).items():
        for name in results:
            results[name]["slices"][slice_name] = {"n": int(mask.sum()), "event_dates": sorted(frame.loc[mask, "delivery_date"].dt.strftime("%Y-%m-%d").unique().tolist()),
                                                   "fold_changes": {}}
        for fold, *_ in base.FOLDS:
            eligible = mask & (frame.fold == fold).to_numpy()
            for name, weights in contrasts.items():
                values = {key: float(sum(weight * loss[arm][key][eligible].mean() for arm, weight in weights.items()))
                          if eligible.any() else None for key in METRICS}
                values["mean_rmse_usd_mwh"] = float(sum(weight * np.sqrt(loss[arm][METRICS[1]][eligible].mean())
                                                        for arm, weight in weights.items())) if eligible.any() else None
                results[name]["slices"][slice_name]["fold_changes"][fold] = {"n": int(eligible.sum()), **values}
        for scheme, (codes, dates, picked) in plans.items():
            counts = np.bincount(codes, weights=mask.astype(int), minlength=len(dates))
            denominators = counts[picked].sum(axis=1)
            draws, group_means, observed = {}, {}, {}
            for arm, losses in loss.items():
                draws[arm], group_means[arm], observed[arm] = {}, {}, {}
                for key in METRICS:
                    sums = np.bincount(codes, weights=losses[key] * mask, minlength=len(dates))
                    sample_sums = sums[picked].sum(axis=1)
                    draws[arm][key] = np.divide(sample_sums, denominators, out=np.full(3000, np.nan), where=denominators > 0)
                    group_means[arm][key] = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
                    observed[arm][key] = float(losses[key][mask].mean()) if mask.any() else None
                draws[arm]["mean_rmse_usd_mwh"] = np.sqrt(draws[arm][METRICS[1]])
                group_means[arm]["mean_rmse_usd_mwh"] = np.sqrt(group_means[arm][METRICS[1]])
                observed[arm]["mean_rmse_usd_mwh"] = float(np.sqrt(observed[arm][METRICS[1]])) if mask.any() else None
            for name, weights in contrasts.items():
                entries = {}
                for key in (*METRICS, "mean_rmse_usd_mwh"):
                    values = sum(weight * draws[arm][key] for arm, weight in weights.items())
                    valid = np.isfinite(values)
                    entries[key] = {"observed_delta": float(sum(weight * observed[arm][key] for arm, weight in weights.items())) if mask.any() else None,
                                    "ci95": np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else None,
                                    "defined_draws": int(valid.sum()), "undefined_no_eligible_event_draws": int((~valid).sum())}
                    if scheme == "whole_delivery_week":
                        delta = sum(weight * group_means[arm][key] for arm, weight in weights.items())
                        entries[key].update(improved_weeks=int((delta < 0).sum()), weeks_with_eligible_rows=int(np.isfinite(delta).sum()),
                                            weeks_total=len(dates))
                results[name]["slices"][slice_name][scheme] = {"metadata": metadata[scheme], "metrics": entries}
    output = {"technique_id": "BATCH002-PAIRED-v1", "manifest_sha256": base.digest(BATCH / "paired_manifest.json"),
              "contrasts": results, "raw_dam_note": "Raw DAM quantile fields are point-price diagnostics, not a distribution"}
    base.save_json(BATCH / "paired_comparisons.json", output, exclusive=True)
    return output


def research_decision(paired):
    metrics = {arm: json.loads((OUT / arm / "metrics.json").read_text())["pooled"]["lightgbm"] for arm in ("EXP001", *BLOCKS)}
    raw = json.loads((OUT / "EXP002" / "metrics.json").read_text())["pooled"]["raw_dam"]
    columns = {"EXP001": 7, **{arm: 7 + len(block) for arm, block in BLOCKS.items()}}
    primary = ("p50_mae_usd_mwh", "mean_pinball")
    decisions = {}
    for arm in BLOCKS:
        improve = all(metrics[arm][key] < metrics["EXP001"][key] - 1e-9 for key in primary)
        dominates = lambda other: all(metrics[other][k] <= metrics[arm][k] + 1e-9 for k in primary) and (
            any(metrics[other][k] < metrics[arm][k] - 1e-9 for k in primary) or (columns[other], other) < (columns[arm], arm))
        smaller_dominators = [other for other in metrics if other != arm and columns[other] < columns[arm] and dominates(other)]
        all_dominators = [other for other in metrics if other != arm and dominates(other)]
        stability = {}
        comparison = paired["contrasts"][f"{arm}_minus_EXP001"]["slices"]["all"]
        for key in (METRICS[0], METRICS[2]):
            fold_changes = {fold: values[key] for fold, values in comparison["fold_changes"].items()}
            interval = comparison["whole_delivery_week"]["metrics"][key]["ci95"]
            stability[key] = {"fold_changes": fold_changes, "improved_folds": sum(v < 0 for v in fold_changes.values()),
                              "same_gain_sign_at_least_four_of_six": sum(v < 0 for v in fold_changes.values()) >= 4,
                              "paired_week_ci95": interval, "paired_week_interval_includes_zero": interval[0] <= 0 <= interval[1],
                              "moving_seven_day_ci95": comparison["moving_seven_day"]["metrics"][key]["ci95"]}
        decisions[arm] = {"columns": columns[arm], "improves_both_primary_losses_over_EXP001": improve,
                          "dominated_by_smaller_arms": smaller_dominators, "dominated_on_primary_losses": all_dominators,
                          "distribution_research_candidate": improve and not smaller_dominators,
                          "nondominated_tradeoff_candidate": not improve and not all_dominators,
                          "stability": stability,
                          "mean_improves_over_EXP001_and_raw_DAM": metrics[arm][METRICS[1]] < min(metrics["EXP001"][METRICS[1]], raw[METRICS[1]]) - 1e-9,
                          "mean_mse_delta_vs_EXP001": metrics[arm][METRICS[1]] - metrics["EXP001"][METRICS[1]],
                          "mean_mse_delta_vs_raw_DAM": metrics[arm][METRICS[1]] - raw[METRICS[1]],
                          "spike_p50_mae_delta_vs_EXP001": metrics[arm]["high_spike_p50_mae_usd_mwh"] - metrics["EXP001"]["high_spike_p50_mae_usd_mwh"]}
    result = {"status": "research_only", "arms": decisions, "ties": "1e-9 in reported losses; fewer added columns, then lower ID",
              "next_consultation": "Continue source discovery. Separately preflight inner-selected mean-residual shrinkage and 90/180-day recency weighting on a frozen feature set.",
              "limits": ["No deployment or revenue claim", "No splicing quantiles across arms", "Reused development folds; intervals not multiplicity corrected",
                         "2025H2 already exposed; 2026 aggregate previously exposed, no 2026 access in this batch", "No evidence here about post-December-2025 transfer"]}
    base.save_json(BATCH / "research_decision.json", result, exclusive=True)
    return result


def run(guards_only=False):
    begun = time.monotonic()
    BATCH.mkdir(parents=True, exist_ok=True)
    prices = base.load_prices()
    frozen, baseline, contract = input_contract(prices)
    print(f"{base.now()} BATCH002 process {os.getpid()} validating approved inputs", flush=True)
    latest = BATCH / "guards/latest.json"
    receipt = None
    if latest.exists():
        pointer = json.loads(latest.read_text())
        assert base.digest(ROOT / pointer["receipt"]) == pointer["sha256"]
        candidate = json.loads((ROOT / pointer["receipt"]).read_text())
        if candidate["contract"] == contract:
            receipt = candidate
    if receipt is None:
        try:
            context, checks = guards(prices, contract)
        except BaseException as error:
            failure = BATCH / f"guard_failure_{time.time_ns()}.json"
            base.save_json(failure, {"status": "failed", "stage": "pre_fit_guards", "at_utc": base.now(),
                                    "contract": contract, "error": f"{type(error).__name__}: {error}"}, exclusive=True)
            raise
    else:
        assert base.digest(ROOT / receipt["context_path"]) == receipt["context_file_sha256"]
        context = joblib.load(ROOT / receipt["context_path"])
        checks = {k: v for k, v in receipt.items() if k not in ("contract", "context_path", "context_file_sha256")}
    print(f"{base.now()} all BATCH002 guards passed", flush=True)
    if guards_only:
        print(json.dumps(checks, indent=2), flush=True)
        return
    batch_record = {"experiment_id": "BATCH002", "status": "running", "started_at_utc": base.now(),
                    "process": process(), "arms": list(BLOCKS), "cohort_id": COHORT, "completed_arms": []}
    base.save_json(BATCH / "manifest.json", {"batch_id": "BATCH002", "created_at_utc": base.now(), **contract,
                   "preflight_path": rel(PREFLIGHT), "planned_real_fits": 144, "arms": list(BLOCKS), "checks_before_fit": checks,
                   "process": process()}, exclusive=True)
    base.save_json(BATCH / "record.json", batch_record)
    manifests, records = {}, {}
    for arm in BLOCKS:
        out = OUT / arm
        manifests[arm] = make_manifest(arm, frozen, contract, checks)
        base.save_json(out / "manifest.json", manifests[arm], exclusive=True)
        base.save_json(out / "approval.json", {"experiment_id": arm, "preflight_path": rel(PREFLIGHT),
                       "preflight_sha256": base.digest(PREFLIGHT), "approved_fits": 48, "approved_before_first_real_fit": True,
                       "guard_receipt": json.loads(latest.read_text()), "created_at_utc": base.now()}, exclusive=True)
        snapshot = out / "code_snapshot"
        snapshot.mkdir(parents=True, exist_ok=False)
        for path in CODE:
            shutil.copy2(path, snapshot / path.name)
        record = {"experiment_id": arm, "status": "queued", "hypothesis": HYPOTHESES[arm], "parent": "EXP001",
                  "model": "lightgbm", "model_family": "LightGBM residual mean and quantiles", "feature_columns": manifests[arm]["features"],
                  "feature_groups": list(manifests[arm]["ordered_feature_blocks"]), "cohort_id": COHORT,
                  "cohort": {"id": COHORT, "rows": 420064, "row_key_sha256": contract["row_key_sha256"]},
                  "techniques": ["prespecified nested expanding-window fits", "same-row frozen EXP001 benchmarks", "four-cell group ablations",
                                 "paired whole-week and moving-seven-day bootstrap", f"{arm}-GPI-v1"] + (["EXP004-CPI-v1"] if arm == "EXP004" else []),
                  "evaluation_role": "retrospective exposed development-validation", "preflight_path": rel(PREFLIGHT),
                  "started_at_utc": None, "completed_at_utc": None, "completed_folds": [], "process": process(),
                  "data_vintage_caveats": manifests[arm]["data_vintage_caveats"], "comparison_rows": [],
                  "artifact_links": {"manifest": "manifest.json", "predictions": "predictions.csv.gz", "metrics": "metrics.json",
                                     "importance": "importance.json", "verification": "verification.json", "models": "folds/",
                                     "chart_metrics": "chart_metrics.csv", "code_snapshot": "code_snapshot/",
                                     "paired_comparisons": "../BATCH002/paired_comparisons.json", "research_decision": "../BATCH002/research_decision.json"}}
        records[arm] = record
        base.save_json(out / "record.json", record)
    try:
        for arm in BLOCKS:
            records[arm].update(status="running", started_at_utc=base.now())
            base.save_json(OUT / arm / "record.json", records[arm])
            run_arm(arm, prices, context, baseline, manifests[arm], records[arm])
            batch_record["completed_arms"].append(arm)
            base.save_json(BATCH / "record.json", batch_record)
        print(f"{base.now()} computing shared 3000-draw paired comparisons", flush=True)
        paired = paired_batch()
        decision = research_decision(paired)
        check_contract(contract)
        for arm in BLOCKS:
            records[arm].update(status="complete", completed_at_utc=base.now(), research_decision=decision["arms"][arm])
            records[arm]["process"]["terminal_exit_code"] = 0
            base.save_json(OUT / arm / "record.json", records[arm])
        batch_record.update(status="complete", completed_at_utc=base.now(), elapsed_seconds=round(time.monotonic() - begun, 3))
        batch_record["process"]["terminal_exit_code"] = 0
        base.save_json(BATCH / "record.json", batch_record)
        print(json.dumps({"status": "complete", "seconds": batch_record["elapsed_seconds"],
                          "pooled": {arm: records[arm]["pooled_metrics"] for arm in BLOCKS}, "decision": decision}, indent=2), flush=True)
    except BaseException as error:
        batch_record.update(status="failed", failed_at_utc=base.now(), error=f"{type(error).__name__}: {error}")
        base.save_json(BATCH / "record.json", batch_record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Generated feature and donor guards; no model fitting")
    parser.add_argument("--guards", action="store_true", help="All source and bounded synthetic fit guards; no real-data fitting")
    parser.add_argument("--verify", action="store_true", help="Replay saved models and metrics without fitting")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(feature_guards(), indent=2))
    elif args.verify:
        prices = base.load_prices()
        _, baseline, _ = input_contract(prices)
        print(json.dumps({arm: verify_arm(arm, prices, baseline) for arm in BLOCKS}, indent=2))
    else:
        run(guards_only=args.guards)
