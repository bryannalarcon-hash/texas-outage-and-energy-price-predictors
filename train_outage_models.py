#!/usr/bin/env python3
"""Train a reproducible *recorded county-scenario* onset/duration pilot.

This is not a home-outage model. Missing PNNL scenarios are noisy negatives.
Run after download_eaglei_tx.py and build_outlook_features.py:
.venv/bin/python train_outage_models.py
"""

import csv
import datetime as dt
import gzip
import json
import math
import sys
import zipfile
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data/outage/raw/Outage_Dataset_R1.zip"
GEO = ROOT / "data/outage/raw/texas_counties_fips.geojson"
OUTLOOK = ROOT / "data/outage/forecast/outlook_county_day_2018_2023.csv.gz"
OUTPUT = ROOT / "data/outage/models"
UTC = dt.timezone.utc
YEARS = range(2018, 2024)
HOURS = np.arange(24)
THRESHOLDS = np.array([1., 4., 12., 24.], dtype=np.float32)
AGES = range(169)  # hourly updates through one week; rarer tails are outside the evaluated range
SEED = 1729
ONSET_COLUMNS = ["latitude", "longitude", "doy_sin", "doy_cos", "spc_risk", "spc_available",
                 "wpc_risk", "wpc_available", "hour_sin", "hour_cos", "lead_hours"]
DURATION_COLUMNS = ["latitude", "longitude", "doy_sin", "doy_cos", "spc_risk", "spc_available",
                    "wpc_risk", "wpc_available", "hour_sin", "hour_cos", "log_age", "log_threshold"]


def year_limit(year):
    # ORNL source files through 2021 end on Dec 31 at 00:00 UTC.
    return dt.datetime(year, 12, 31, tzinfo=UTC) if year <= 2021 else dt.datetime(year + 1, 1, 1, tzinfo=UTC)


def load_events():
    records = []
    with zipfile.ZipFile(ZIP) as archive:
        for year in YEARS:
            with archive.open(f"Outage_Dataset/eaglei_outages_{year}_merged.csv") as file:
                for row in csv.DictReader(line.decode() for line in file):
                    if row["state"] != "Texas":
                        continue
                    start = dt.datetime.fromisoformat(row["start_time"]).replace(tzinfo=UTC)
                    duration = float(row["duration"])
                    if duration <= 0 or not math.isfinite(duration):
                        continue
                    records.append({"fips": row["fips"].zfill(5), "year": year, "start": start,
                                    "end": start + dt.timedelta(hours=duration), "duration": duration})
    return pd.DataFrame(records).sort_values(["fips", "start"]).reset_index(drop=True)


def mark_observed_ends(events):
    """Keep only durations whose next raw 15-minute count visibly falls below 200."""
    wanted = {year: {(row.fips, row.end.strftime("%Y-%m-%d %H:%M:%S"))
                     for row in events[events.year == year].itertuples()} for year in YEARS}
    observed = {}
    for year in YEARS:
        with (ROOT / f"data/outage/raw/eaglei_tx_{year}.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                key = (row["fips_code"].zfill(5), row["run_start_time"])
                if key not in wanted[year]:
                    continue
                value = row["customers_out"]
                observed[(year, *key)] = ("missing_count" if not value else
                                           "observed_below_200" if float(value) < 200 else
                                           "observed_200_plus")
    events = events.copy()
    events["end_status"] = [observed.get((row.year, row.fips, row.end.strftime("%Y-%m-%d %H:%M:%S")),
                                         "no_raw_row") for row in events.itertuples()]
    return events


def load_counties():
    geo = json.loads(GEO.read_text())
    result = {}
    for feature in geo["features"]:
        fips = f"48{int(feature['properties']['Fips']):03d}"
        point = shape(feature["geometry"]).centroid
        result[fips] = (float(point.y), float(point.x))
    assert len(result) == 254
    return result


def load_outlooks():
    result = {}
    with gzip.open(OUTLOOK, "rt", newline="") as file:
        reader = csv.DictReader(file)
        required = {"issue_utc", "county_fips", "spc_risk", "spc_forecast_available",
                    "wpc_risk", "wpc_forecast_available"}
        assert required <= set(reader.fieldnames), reader.fieldnames
        for row in reader:
            issue = dt.datetime.fromisoformat(row["issue_utc"].replace("Z", "+00:00"))
            assert issue.tzinfo is not None and issue.hour == 12
            key = (issue.date(), row["county_fips"].zfill(5))
            assert key not in result
            result[key] = (float(row["spc_risk"]) if row["spc_risk"] else -1.,
                           float(row["spc_forecast_available"]),
                           float(row["wpc_risk"]) if row["wpc_risk"] else -1.,
                           float(row["wpc_forecast_available"]))
    return result


def date_features(time):
    angle = 2 * math.pi * (time.timetuple().tm_yday - 1) / 365.25
    return math.sin(angle), math.cos(angle)


def hour_features(hour):
    angle = 2 * math.pi * (hour % 24) / 24
    return math.sin(angle), math.cos(angle)


def active_intervals(events):
    intervals = []
    for start, end in events:
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    return [x[0] for x in intervals], [x[1] for x in intervals]


def first_onset_hour(issue, starts, merged_starts, merged_ends):
    active_index = bisect_right(merged_starts, issue) - 1
    if active_index >= 0 and issue < merged_ends[active_index]:
        return None  # not at risk at issue time
    index = bisect_right(starts, issue)  # events already underway at issue are not forecasts
    if index == len(starts) or starts[index] >= issue + dt.timedelta(hours=24):
        return -1
    return int((starts[index] - issue).total_seconds() // 3600)


def eligible_counties(events):
    # Eligibility uses training years only; looking at 2022/2023 would select on held-out labels.
    groups = events[events.year <= 2021].groupby(["year", "fips"]).size()
    return sorted(fips for fips in events.fips.unique()
                  if all(groups.get((year, fips), 0) >= 5 for year in range(2018, 2022)))


def build_issues(events, counties, outlooks, eligible):
    by_county = {}
    for fips, group in events.groupby("fips"):
        pairs = list(zip(group.start.tolist(), group.end.tolist()))
        starts = [x[0] for x in pairs]
        merged_starts, merged_ends = active_intervals(pairs)
        by_county[fips] = (starts, merged_starts, merged_ends)
    rows = []
    for year in YEARS:
        issue = dt.datetime(year, 1, 1, 12, tzinfo=UTC)
        while issue + dt.timedelta(hours=24) <= year_limit(year):
            doy_sin, doy_cos = date_features(issue)
            for fips in eligible:
                starts, merged_starts, merged_ends = by_county[fips]
                first_hour = first_onset_hour(issue, starts, merged_starts, merged_ends)
                if first_hour is None:
                    continue
                lat, lon = counties[fips]
                spc, spc_available, wpc, wpc_available = outlooks.get((issue.date(), fips), (-1., 0., -1., 0.))
                rows.append((issue, year, fips, issue.month, first_hour, lat, lon, doy_sin, doy_cos,
                             spc, spc_available, wpc, wpc_available))
            issue += dt.timedelta(days=1)
    columns = ["issue", "year", "fips", "month", "first_hour", "latitude", "longitude", "doy_sin",
               "doy_cos", "spc_risk", "spc_available", "wpc_risk", "wpc_available"]
    return pd.DataFrame.from_records(rows, columns=columns)


def onset_matrix(issues, training=False, negative_keep=0.12):
    if training:
        rng = np.random.default_rng(SEED)
        selected = (issues.first_hour.to_numpy() >= 0) | (rng.random(len(issues)) < negative_keep)
        issues = issues.iloc[np.flatnonzero(selected)].reset_index(drop=True)
    base = issues[["latitude", "longitude", "doy_sin", "doy_cos", "spc_risk", "spc_available",
                   "wpc_risk", "wpc_available"]].to_numpy(dtype=np.float32)
    count = len(issues)
    lead = np.tile(HOURS, count)
    hour = (12 + lead) % 24
    first = np.repeat(issues.first_hour.to_numpy(dtype=np.int16), 24)
    angle = hour * (2 * np.pi / 24)
    x = np.column_stack((np.repeat(base, 24, axis=0), np.sin(angle), np.cos(angle), lead)).astype(np.float32)
    valid = (first < 0) | (lead <= first)
    y = (lead == first).astype(np.uint8)
    weight = np.repeat(np.where(issues.first_hour.to_numpy() < 0, 1 / negative_keep if training else 1., 1.), 24)
    if training:
        return x[valid], y[valid], weight[valid]
    return x, y, valid


def county_prior(train, scored):
    global_rate = (train.first_hour >= 0).sum() / np.where(train.first_hour >= 0, train.first_hour + 1, 24).sum()
    month = train.groupby("month").apply(
        lambda g: (g.first_hour >= 0).sum() / np.where(g.first_hour >= 0, g.first_hour + 1, 24).sum(),
        include_groups=False).to_dict()
    county_month = defaultdict(lambda: [0, 0])
    for row in train.itertuples():
        item = county_month[(row.fips, row.month)]
        item[0] += row.first_hour >= 0
        item[1] += row.first_hour + 1 if row.first_hour >= 0 else 24
    hazards = []
    for row in scored.itertuples():
        positive, hours = county_month[(row.fips, row.month)]
        hazards.append((positive + 500 * month.get(row.month, global_rate)) / (hours + 500))
    return np.repeat(np.asarray(hazards, dtype=np.float32)[:, None], 24, axis=1)


def score_binary(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.uint8)
    result = {"n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
              "average_precision": float(average_precision_score(y, p)),
              "brier": float(brier_score_loss(y, p)),
              "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    if 0 < y.sum() < len(y):
        result["roc_auc"] = float(roc_auc_score(y, p))
    order = np.argsort(p)[::-1][:max(1, len(y) // 10)]
    result["recall_at_top_10pct"] = float(y[order].sum() / max(1, y.sum()))
    bins = np.array_split(np.argsort(p), 10)
    result["ece_10_equal_count"] = float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins))
    return result


def onset_probabilities(hazard):
    hazard = np.clip(np.asarray(hazard), 1e-7, 1 - 1e-7)
    no_prior = np.concatenate((np.ones((len(hazard), 1)), np.cumprod(1 - hazard, axis=1)[:, :-1]), axis=1)
    first = hazard * no_prior
    return first, first.sum(axis=1)


def candidates():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, solver="lbfgs")),
        "forest": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=80,
                                         max_features=0.8, n_jobs=4, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15,
                                                                  min_samples_leaf=200, learning_rate=0.06,
                                                                  l2_regularization=1., random_state=SEED),
    }


def fit_candidate(model, x, y, weight=None):
    if hasattr(model, "steps"):
        model.fit(x, y, **({"logisticregression__sample_weight": weight} if weight is not None else {}))
    else:
        model.fit(x, y, **({"sample_weight": weight} if weight is not None else {}))
    return model


def predict_candidate(model, x):
    return np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)


def onset_experiment(issues):
    train, val, test = (issues[issues.year.isin(years)].reset_index(drop=True)
                        for years in ([2018, 2019, 2020, 2021], [2022], [2023]))
    x_train, y_train, weights = onset_matrix(train, training=True)
    x_val, y_val, valid_val = onset_matrix(val)
    x_test, y_test, valid_test = onset_matrix(test)
    assert x_train.shape[1] == len(ONSET_COLUMNS)
    assert train.issue.max().year < val.issue.min().year < test.issue.min().year
    val_any = (val.first_hour.to_numpy() >= 0).astype(np.uint8)
    test_any = (test.first_hour.to_numpy() >= 0).astype(np.uint8)
    prior_val = county_prior(train, val)
    prior_test = county_prior(train, test)
    results = {"baseline": {"validation": score_binary(val_any, onset_probabilities(prior_val)[1]),
                             "test": score_binary(test_any, onset_probabilities(prior_test)[1])}}
    fitted = {}
    raw_val = {}
    for name, model in candidates().items():
        print(f"Fitting onset {name}", flush=True)
        fitted[name] = fit_candidate(model, x_train, y_train, weights)
        p_val = predict_candidate(model, x_val).reshape(-1, 24)
        raw_val[name] = p_val
        p_test = predict_candidate(model, x_test).reshape(-1, 24)
        results[name] = {
            "validation": score_binary(val_any, onset_probabilities(p_val)[1]),
            "test_raw": score_binary(test_any, onset_probabilities(p_test)[1]),
            "hourly_test_raw": score_binary(y_test[valid_test], p_test.ravel()[valid_test]),
        }
    chosen = min(fitted, key=lambda name: results[name]["validation"]["brier"])
    calibration = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibration.fit(raw_val[chosen].ravel()[valid_val], y_val[valid_val])
    selected_test = calibration.predict(predict_candidate(fitted[chosen], x_test)).reshape(-1, 24)
    first, any_next = onset_probabilities(selected_test)
    assert np.allclose(first.sum(axis=1), any_next)
    results[chosen]["test_calibrated"] = score_binary(test_any, any_next)
    results[chosen]["hourly_test_calibrated"] = score_binary(y_test[valid_test], selected_test.ravel()[valid_test])
    artifact = {"model": fitted[chosen], "calibration": calibration, "features": ONSET_COLUMNS,
                "selected_by": "2022 validation 24-hour Brier", "county_grain_only": True}
    joblib.dump(artifact, OUTPUT / "onset_county.joblib", compress=3)
    sample = {"county_fips": test.fips.iloc[0], "issued_at": test.issue.iloc[0].isoformat(),
              "p_first_start_by_hour": [round(float(v), 6) for v in first[0]],
              "p_any_next_24h": round(float(any_next[0]), 6)}
    (OUTPUT / "onset_example.json").write_text(json.dumps(sample, indent=2) + "\n")
    summary = {"train_issues": len(train), "validation_issues": len(val), "test_issues": len(test),
               "train_sampled_hour_rows": len(y_train), "selected": chosen, "models": results}
    return summary


def nonoverlap_events(events):
    bad = set()
    for _, group in events.groupby("fips"):
        previous = None
        furthest_end = None
        for row in group.itertuples():
            if furthest_end is not None and row.start < furthest_end:
                bad.add(row.Index)
                bad.add(previous)
            if furthest_end is None or row.end > furthest_end:
                previous, furthest_end = row.Index, row.end
    return events.loc[~events.index.isin(bad)], len(bad)


def build_duration_rows(events, counties, outlooks, eligible):
    events = events[events.fips.isin(eligible)].copy()
    end_status_counts = events.end_status.value_counts().to_dict()
    events, excluded_overlap = nonoverlap_events(events)
    events = events[events.end_status == "observed_below_200"]
    excluded_cutoff = int((events.apply(lambda row: row["end"] > year_limit(row["year"]), axis=1)).sum())
    events = events[events.apply(lambda row: row["end"] <= year_limit(row["year"]), axis=1)]
    records = []
    for row in events.itertuples():
        lat, lon = counties[row.fips]
        for age in AGES:
            if age >= row.duration:
                continue
            landmark = row.start + dt.timedelta(hours=age)
            prior_date = (landmark - dt.timedelta(hours=12)).date()
            spc, spc_available, wpc, wpc_available = outlooks.get((prior_date, row.fips), (-1., 0., -1., 0.))
            doy_sin, doy_cos = date_features(landmark)
            hour_sin, hour_cos = hour_features(landmark.hour + landmark.minute / 60)
            for threshold in THRESHOLDS:
                records.append((row.Index, row.year, row.fips, age, float(threshold),
                                int(row.duration - age > threshold), lat, lon, doy_sin, doy_cos,
                                spc, spc_available, wpc, wpc_available, hour_sin, hour_cos,
                                math.log1p(age), math.log1p(float(threshold))))
    columns = ["event_id", "year", "fips", "age", "threshold", "target", *DURATION_COLUMNS]
    result = pd.DataFrame.from_records(records, columns=columns)
    return result, {"usable_events": len(events), "end_status_counts_before_other_exclusions": end_status_counts,
                    "excluded_past_file_cutoff": excluded_cutoff,
                    "excluded_overlapping_events": excluded_overlap}


def km_predict(train_durations, age, threshold):
    sorted_durations = np.sort(np.asarray(train_durations, dtype=np.float64))
    n_age = len(sorted_durations) - np.searchsorted(sorted_durations, age, side="right")
    n_later = len(sorted_durations) - np.searchsorted(sorted_durations, age + threshold, side="right")
    return np.clip(n_later / np.maximum(n_age, 1), 1e-7, 1 - 1e-7)


def duration_scores(rows, probabilities):
    probabilities = np.asarray(probabilities)
    output = {}
    for threshold in THRESHOLDS:
        mask = rows.threshold.to_numpy() == threshold
        y = rows.target.to_numpy()[mask]
        p = np.clip(probabilities[mask], 1e-7, 1 - 1e-7)
        bins = np.array_split(np.argsort(p), 10)
        output[str(int(threshold))] = {"n": int(mask.sum()), "positive": int(y.sum()),
                                       "prevalence": float(y.mean()),
                                       "brier": float(np.mean((y - p) ** 2)),
                                       "log_loss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
                                       "ece_10_equal_count": float(sum(len(b) / len(y) * abs(
                                           y[b].mean() - p[b].mean())
                                           for b in bins if len(b)))}
    output["mean_brier"] = float(np.mean([output[str(int(h))]["brier"] for h in THRESHOLDS]))
    return output


def monotone_duration(p):
    # Records are emitted in four sorted threshold rows per event-landmark.
    return np.minimum.accumulate(np.asarray(p).reshape(-1, 4), axis=1).ravel()


def duration_experiment(duration_rows, events):
    train, val, test = (duration_rows[duration_rows.year.isin(years)].reset_index(drop=True)
                        for years in ([2018, 2019, 2020, 2021], [2022], [2023]))
    x_train = train[DURATION_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train.target.to_numpy(dtype=np.uint8)
    x_val = val[DURATION_COLUMNS].to_numpy(dtype=np.float32)
    x_test = test[DURATION_COLUMNS].to_numpy(dtype=np.float32)
    train_durations = events.loc[events.index.isin(train.event_id.unique()), "duration"].to_numpy()
    baseline_val = km_predict(train_durations, val.age.to_numpy(), val.threshold.to_numpy())
    baseline_test = km_predict(train_durations, test.age.to_numpy(), test.threshold.to_numpy())
    results = {"baseline_km": {"validation": duration_scores(val, baseline_val),
                               "test": duration_scores(test, baseline_test)}}
    fitted = {}
    for name, model in candidates().items():
        print(f"Fitting duration {name}", flush=True)
        fitted[name] = fit_candidate(model, x_train, y_train)
        p_val = monotone_duration(predict_candidate(model, x_val))
        p_test = monotone_duration(predict_candidate(model, x_test))
        results[name] = {"validation": duration_scores(val, p_val), "test_raw": duration_scores(test, p_test)}
    chosen = min(fitted, key=lambda name: results[name]["validation"]["mean_brier"])
    calibration = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    p_val = monotone_duration(predict_candidate(fitted[chosen], x_val))
    calibration.fit(p_val, val.target.to_numpy(dtype=np.uint8))
    p_test = monotone_duration(calibration.predict(monotone_duration(predict_candidate(fitted[chosen], x_test))))
    results[chosen]["test_calibrated"] = duration_scores(test, p_test)
    bands = {"at_start": test.age == 0, "elapsed_1_to_4h": test.age.between(1, 4),
             "elapsed_5_to_24h": test.age.between(5, 24), "elapsed_25_to_168h": test.age.between(25, 168)}
    results[chosen]["test_calibrated_by_elapsed_age"] = {
        name: duration_scores(test.loc[mask].reset_index(drop=True), p_test[mask.to_numpy()])
        for name, mask in bands.items()}
    results["baseline_km"]["test_by_elapsed_age"] = {
        name: duration_scores(test.loc[mask].reset_index(drop=True), baseline_test[mask.to_numpy()])
        for name, mask in bands.items()}
    joblib.dump({"model": fitted[chosen], "calibration": calibration, "features": DURATION_COLUMNS,
                 "selected_by": "2022 validation mean threshold Brier", "county_grain_only": True},
                OUTPUT / "duration_county.joblib", compress=3)
    sample_rows = test.iloc[:4]
    assert len(sample_rows) == 4 and len(sample_rows.event_id.unique()) == 1 and len(sample_rows.age.unique()) == 1
    started = events.loc[sample_rows.event_id.iloc[0], "start"]
    sample = {"county_fips": sample_rows.fips.iloc[0], "issued_at": (started + dt.timedelta(hours=float(sample_rows.age.iloc[0]))).isoformat(),
              "outage_started_at": started.isoformat(),
              "elapsed_hours": float(sample_rows.age.iloc[0]),
              **{f"p_remaining_gt_{int(h)}h": round(float(p_test[i]), 6) for i, h in enumerate(THRESHOLDS)}}
    (OUTPUT / "duration_example.json").write_text(json.dumps(sample, indent=2) + "\n")
    return {"train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "train_events": train.event_id.nunique(), "validation_events": val.event_id.nunique(),
            "test_events": test.event_id.nunique(), "selected": chosen, "models": results}


def self_test():
    t = dt.datetime(2023, 5, 1, 12, tzinfo=UTC)
    starts = [t + dt.timedelta(minutes=15), t + dt.timedelta(hours=3)]
    merged_starts, merged_ends = active_intervals([(starts[0], t + dt.timedelta(hours=2)),
                                                    (starts[1], t + dt.timedelta(hours=4))])
    assert first_onset_hour(t, starts, merged_starts, merged_ends) == 0
    assert first_onset_hour(t + dt.timedelta(hours=1), starts, merged_starts, merged_ends) is None
    assert first_onset_hour(starts[0], starts, merged_starts, merged_ends) is None
    assert first_onset_hour(t - dt.timedelta(hours=24), starts, merged_starts, merged_ends) == -1
    hazard = np.array([[0.1, 0.2] + [0.] * 22])
    first, any_next = onset_probabilities(hazard)
    assert np.allclose(first[0, :2], [0.1, 0.18]) and abs(any_next[0] - 0.28) < 1e-5
    assert np.allclose(monotone_duration([.9, .8, .85, .4]), [.9, .8, .8, .4])
    score = duration_scores(pd.DataFrame({"threshold": np.tile(THRESHOLDS, 2),
                                           "target": [0] * 4 + [1] * 4}), [0.5] * 8)
    assert score["1"]["prevalence"] == 0.5 and score["1"]["brier"] == 0.25


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = mark_observed_ends(load_events())
    events.groupby(["year", "fips", "end_status"]).duration.agg(episodes="size", median_duration_hours="median").to_csv(
        OUTPUT / "duration_end_audit.csv")
    counties = load_counties()
    outlooks = load_outlooks()
    eligible = eligible_counties(events)
    issues = build_issues(events, counties, outlooks, eligible)
    assert len(eligible) >= 100 and len(issues) > 100_000
    issues.to_csv(OUTPUT / "county_day_issues.csv.gz", index=False, compression="gzip")
    onset = onset_experiment(issues)
    duration_rows, exclusions = build_duration_rows(events, counties, outlooks, eligible)
    duration = duration_experiment(duration_rows, events)
    report = {"dataset": "PNNL merged Texas county scenarios, not home outages",
              "outcome_limit": "No PNNL merged episode recorded; absence may mean no qualifying event, below the observed coverage floor, or missing observation",
              "split": {"train": "2018-2021", "validation": "2022", "test": "2023", "issue_utc": "12:00 daily",
                        "year_boundary": "exclude 24h forecast windows crossing observed file cutoff"},
              "boundary_episodes_not_forecast": {str(year): int(((events.fips.isin(eligible)) &
                  (events.year == year) & (events.start.dt.hour == 12) & (events.start.dt.minute == 0) &
                  (events.start.dt.second == 0) & (events.start.dt.microsecond == 0)).sum())
                  for year in YEARS},
              "stage2_label": "persistence of recorded county count at or above the observed 200-customer floor; require a raw end count below 200",
              "stage2_evaluated_elapsed_hours": "0 through 168, hourly",
              "stage2_metric_unit": "one still-active episode-hour per row and remaining-time threshold; at-start slice scores each episode once",
              "forecast_archive_limit": "IEM reconstructed NOAA outlook issue times; exact historical distribution latency is unavailable",
              "eligible_counties_train_only": len(eligible), "eligible_fips": eligible,
              "source_zip": "https://data.openei.org/files/6458/Outage_Dataset_R1.zip",
              "source_outlooks": "data/outage/forecast/outlook_sources.json",
              "source_alignment": "data/outage/models/label_alignment.json",
              "models": {"onset": onset, "duration": duration}, "duration_exclusions": exclusions,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected onset={onset['selected']}, duration={duration['selected']}; metrics: {OUTPUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    self_test()
    if sys.argv[1:] == ["--self-test"]:
        print("self-test passed")
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: .venv/bin/python train_outage_models.py [--self-test]")
