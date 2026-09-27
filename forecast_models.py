#!/usr/bin/env python3
"""Pure inference from public, non-pickle frozen model assets. No training or I/O.

Input acquisition and mutable run publication belong to the service. The only
file reads here are versioned model assets; original research archives are not
required. Run ``python -m unittest test_forecast_models`` for historical replay.
"""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / 'model_assets'
UTC, CENTRAL = timezone.utc, ZoneInfo('America/Chicago')


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise ValueError('An explicit UTC timestamp is required')
    if parsed.microsecond:
        raise ValueError('Timestamps must use whole seconds')
    return parsed.astimezone(UTC)


def stamp(value):
    return utc(value).strftime('%Y-%m-%dT%H:%M:%SZ')


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ASSETS / 'manifest.json').read_text())


@lru_cache(maxsize=12)
def asset(name):
    data = (ASSETS / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest()['assets_sha256'][name]:
        raise ValueError(f'Model asset checksum mismatch: {name}')
    return json.loads(data) if name.endswith('.json') else lgb.Booster(model_str=data.decode())


def load_counties():
    """Original county centroids and train-only eligibility, keyed by FIPS."""
    return asset('counties.json')


def blend_weight(issued_at_utc, *, blend_state=None, history_rows=()):
    """Causal least-squares weight, with at least 48 elapsed hours label delay.

History rows use original exp002_mean (or exp002_mean_usd_mwh), DAM, actual RTM,
interval_end_utc, model_issue_utc, settlement_point; optionally actual_available_at_utc.
For late or revised labels, omit blend_state and pass the complete saved ledger:
each call recomputes from the immutable historical seed. Reusing returned state
is only valid for complete, chronologically admitted batches; its single maximum
timestamp deliberately excludes older outcomes from being counted twice.
    """
    issue = utc(issued_at_utc)
    eligible_end = issue - timedelta(hours=48)
    if blend_state is None:
        snapshots = [s for s in asset('blend_history.json') if utc(s['cutoff_utc']) <= issue]
        if not snapshots:
            raise ValueError('No causally eligible blend calibration exists before 2025-06-30T22:00:00Z')
        blend_state = snapshots[-1]
    state = dict(blend_state)
    maximum = utc(state['max_used_interval_end_utc'])
    if utc(state['cutoff_utc']) > issue or maximum > eligible_end:
        raise ValueError('Blend state includes future or less-than-48-hour-old outcomes')
    numerator, denominator, count = float(state['numerator']), float(state['denominator']), int(state['rows'])
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator < 0 or count < 0:
        raise ValueError('Invalid blend sufficient statistics')
    seen, admitted, last = set(), 0, maximum
    for row in history_rows:
        end = utc(row['interval_end_utc'])
        if end <= maximum or end > eligible_end:
            continue
        available = utc(row.get('actual_available_at_utc', stamp(end + timedelta(hours=48))))
        if available > issue:
            continue
        key = (row['settlement_point'], end)
        if key in seen:
            raise ValueError('Duplicate blend outcome')
        if key[0] not in manifest()['price']['zones']:
            raise ValueError('Unknown blend settlement point')
        if utc(row['model_issue_utc']) >= end - timedelta(minutes=15):
            raise ValueError('Blend forecast must have been issued before its interval')
        seen.add(key)
        dam, actual = float(row['dam_spp_usd_mwh']), float(row['rtm_spp_usd_mwh'])
        predicted = float(row.get('exp002_mean_usd_mwh', row.get('exp002_mean')))
        if not all(math.isfinite(v) for v in (dam, actual, predicted)):
            raise ValueError('Nonfinite blend price')
        residual = predicted - dam
        numerator += residual * (actual - dam)
        denominator += residual * residual
        admitted += 1
        last = max(last, end)
    weight = float(np.clip(numerator / denominator, 0, 1)) if denominator else 0.
    return dict(numerator=numerator, denominator=denominator, rows=count + admitted,
                cutoff_utc=stamp(issue), max_used_interval_end_utc=stamp(last),
                label_available_through_utc=stamp(eligible_end), exp002_weight=weight,
                new_outcome_rows=admitted, minimum_label_delay_hours=48,
                adaptation_status='updated_from_delayed_actuals' if admitted else 'frozen_history')


def _price_design(frame):
    """Exact original EXP002 transform, including float32 baseline features."""
    zone_codes = frame.settlement_point.map(manifest()['price']['zones'])
    x = pd.DataFrame({
        'dam_asinh': np.arcsinh(frame.dam_spp_usd_mwh.to_numpy(dtype=np.float32) / 50),
        'local_hour': frame.hour_ending.to_numpy(dtype=np.float32) - 1 + (frame.quarter.to_numpy(dtype=np.float32) - .5) / 4,
        'day_of_year': frame.delivery_date.dt.dayofyear.to_numpy(dtype=np.float32),
        'zone_code': zone_codes.to_numpy(dtype=np.int8),
        'weekday': frame.delivery_date.dt.dayofweek.to_numpy(dtype=np.int8),
        'quarter': frame.quarter.to_numpy(dtype=np.int8),
        'repeated': frame.repeated_hour_flag.eq('Y').to_numpy(dtype=np.int8),
    })
    for column, categories in [('zone_code', list(manifest()['price']['zones'].values())),
                               ('weekday', list(range(7))), ('quarter', [1, 2, 3, 4]), ('repeated', [0, 1])]:
        x[column] = pd.Categorical(x[column], categories=categories)
    hourly = frame.assign(hour_utc=frame.interval_start_utc.dt.floor('h')).drop_duplicates(['settlement_point', 'hour_utc'])
    hourly = hourly.sort_values(['delivery_date', 'settlement_point', 'hour_utc']).copy()
    groups = hourly.groupby(['delivery_date', 'settlement_point'], sort=False)
    hourly['j'] = groups.cumcount()
    p = 'dam_spp_usd_mwh'
    for operation in ('min', 'max', 'mean'):
        hourly[f'dam_day_{operation}'] = groups[p].transform(operation)
    hourly['dam_day_std'] = groups[p].transform('std', ddof=0)
    hourly['negative'] = hourly[p].lt(0).astype(float)
    hourly['dam_day_negative_share'] = groups.negative.transform('mean')
    hourly['peak_slot'] = hourly.j.where(hourly[p] == hourly.dam_day_max)
    hourly['dam_hours_to_day_peak'] = groups.peak_slot.transform('min') - hourly.j
    hourly['dam_above_day_min'] = hourly[p] - hourly.dam_day_min
    hourly['dam_below_day_max'] = hourly.dam_day_max - hourly[p]
    hourly['dam_prev_hour'], hourly['dam_next_hour'] = groups[p].shift(1), groups[p].shift(-1)
    hourly['dam_ramp_from_prev'] = hourly[p] - hourly.dam_prev_hour
    hourly['dam_ramp_to_next'] = hourly.dam_next_hour - hourly[p]
    context = manifest()['price']['features'][7:]
    joined = frame.assign(hour_utc=frame.interval_start_utc.dt.floor('h')).merge(
        hourly[['settlement_point', 'hour_utc', *context]], on=['settlement_point', 'hour_utc'], validate='many_to_one', sort=False)
    x[context] = joined[context]
    return x[manifest()['price']['features']]


def price_forecast(hourly_dam_rows, issued_at_utc, *, target_date=None, input_cutoff_utc=None,
                   blend_state=None, history_rows=()):
    """Forecast a full day or its future quarters using the complete DAM curve.

    A same-day issue never backdates its run or exports intervals already begun.
    The full 23/24/25-hour input remains necessary for the frozen model features.
    """
    issue, cutoff = utc(issued_at_utc), utc(input_cutoff_utc or issued_at_utc)
    if cutoff > issue:
        raise ValueError('Price input cutoff must not follow issue time')
    rows, dates, seen = [], set(), set()
    for row in hourly_dam_rows:
        start = utc(row['interval_start_utc'])
        zone, price = row['settlement_point'], float(row['dam_spp_usd_mwh'])
        if zone not in manifest()['price']['zones'] or not math.isfinite(price):
            raise ValueError('Unknown settlement point or nonfinite DAM price')
        if start.minute or start.second:
            raise ValueError('DAM intervals must begin on UTC hour boundaries')
        if (zone, start) in seen:
            raise ValueError('Duplicate hourly DAM row')
        seen.add((zone, start))
        dates.add(str(start.astimezone(CENTRAL).date()))
        for quarter in range(4):
            begin = start + timedelta(minutes=15 * quarter)
            local = begin.astimezone(CENTRAL)
            rows.append(dict(settlement_point=zone, interval_start_utc=begin,
                interval_end_utc=begin + timedelta(minutes=15), delivery_date=str(local.date()),
                hour_ending=local.hour + 1, quarter=quarter + 1, repeated_hour_flag='Y' if local.fold else 'N',
                dam_spp_usd_mwh=price))
    if len(dates) != 1 or (target_date is not None and dates != {target_date}):
        raise ValueError('DAM inputs must contain exactly one requested Central operating day')
    day = dates.pop()
    midnight = datetime.fromisoformat(day).replace(tzinfo=CENTRAL)
    finish = midnight + timedelta(days=1)
    first_start = max(midnight.astimezone(UTC), pd.Timestamp(issue).ceil('15min').to_pydatetime())
    if first_start >= finish.astimezone(UTC):
        raise ValueError('No future delivery quarters remain at the actual issue time')
    horizon = 'remaining_day' if issue >= midnight.astimezone(UTC) else 'full_day'
    expected = pd.date_range(midnight.astimezone(UTC), finish.astimezone(UTC), freq='15min', inclusive='left')
    frame = pd.DataFrame(rows).sort_values(['interval_start_utc', 'settlement_point']).reset_index(drop=True)
    for _, group in frame.groupby('settlement_point'):
        if list(group.interval_start_utc) != list(expected):
            raise ValueError('DAM source is not ready: each supplied zone needs its complete 23/24/25-hour curve')
    frame['delivery_date'] = pd.to_datetime(frame.delivery_date)
    x = _price_design(frame)
    dam = frame.dam_spp_usd_mwh.to_numpy()
    values = {key: asset(f'exp002_{key}.txt').predict(x, num_threads=2) + dam for key in ('mean', 'p10', 'p50', 'p90')}
    ordered = np.sort(np.column_stack([values[key] for key in ('p10', 'p50', 'p90')]), axis=1)
    state = blend_weight(issued_at_utc, blend_state=blend_state, history_rows=history_rows)
    blended = dam + state['exp002_weight'] * (values['mean'] - dam)
    records = []
    for i, row in enumerate(frame.itertuples()):
        if row.interval_start_utc < first_start:
            continue
        records.append(dict(settlement_point=row.settlement_point,
            interval_start_utc=stamp(row.interval_start_utc), interval_end_utc=stamp(row.interval_end_utc),
            delivery_date=day, hour_ending=int(row.hour_ending), quarter=int(row.quarter), repeated_hour_flag=row.repeated_hour_flag,
            availability='available', dam_spp_usd_mwh=float(dam[i]), rtm_mean_usd_mwh=float(blended[i]),
            exp002_mean_usd_mwh=float(values['mean'][i]), rtm_p10_usd_mwh=float(ordered[i, 0]),
            rtm_p50_usd_mwh=float(ordered[i, 1]), rtm_p90_usd_mwh=float(ordered[i, 2])))
    return dict(status='available', model_version='EXP002-DAM-blend-frozen-Q2-2025',
        issued_at_utc=stamp(issue), input_cutoff_utc=stamp(cutoff), max_age_hours=48, target_date=day,
        horizon=horizon, records=records, provenance=dict(blend=state, quantiles=manifest()['price']['quantiles'],
            mean_model_training=manifest()['price']['training_period'],
            mean_model_frozen=True, model_manifest_sha256=hashlib.sha256((ASSETS / 'manifest.json').read_bytes()).hexdigest()))


def _county_hazards(x):
    """Evaluate exported numeric HGB nodes, then the fitted isotonic calibration."""
    model = asset('onset_county.json')
    raw = np.full(len(x), model['baseline'], dtype=float)
    for tree in model['trees']:
        node = np.zeros(len(x), dtype=int)
        leaf = np.asarray(tree['is_leaf'], dtype=bool)
        feature, threshold = np.asarray(tree['feature_idx']), np.asarray(tree['num_threshold'])
        left, right = np.asarray(tree['left']), np.asarray(tree['right'])
        missing_left = np.asarray(tree['missing_go_to_left'], dtype=bool)
        while not np.all(leaf[node]):
            active = np.flatnonzero(~leaf[node])
            current = node[active]
            values = x[active, feature[current]]
            choose_left = np.where(np.isnan(values), missing_left[current], values <= threshold[current])
            node[active] = np.where(choose_left, left[current], right[current])
        raw += np.asarray(tree['value'])[node]
    probabilities = np.clip(1 / (1 + np.exp(-raw)), 1e-7, 1 - 1e-7)
    calibration = model['calibration']
    return np.clip(np.interp(probabilities, calibration['x'], calibration['y']), 1e-7, 1 - 1e-7)


def outage_forecast(county_rows, issued_at_utc, *, forecast_origin_utc, input_cutoff_utc=None):
    """Preserve the trained 12Z origin; never recenter stale outlooks as current.

At-risk assumptions emit scenario coverage. Verified at-risk observations emit
observed coverage. Unknown or active county state never gets onset probabilities.
    """
    county_rows = list(county_rows)
    issue, origin = utc(issued_at_utc), utc(forecast_origin_utc)
    cutoff = utc(input_cutoff_utc or forecast_origin_utc)
    if origin.hour != 12 or origin.minute or origin.second or not cutoff <= origin <= issue:
        raise ValueError('County model requires cutoff <= original 12Z forecast origin <= issue')
    intervals = [dict(interval_start_utc=stamp(origin + timedelta(hours=h)),
                      interval_end_utc=stamp(origin + timedelta(hours=h + 1))) for h in range(24)]
    records, design, scored, seen = [], [], [], set()
    angle = 2 * math.pi * (origin.timetuple().tm_yday - 1) / 365.25
    counties = load_counties()
    for row in county_rows:
        fips = str(row['county_fips'])
        if fips not in counties or fips in seen:
            raise ValueError('Unknown or duplicate Texas county FIPS')
        seen.add(fips)
        record = dict(county_fips=fips, coverage='unknown', p_first_start_by_hour=None,
                      p_any_next_24h=None, active_outage=None)
        records.append(record)
        observed, assumed = row.get('at_risk_observed') is True, row.get('at_risk_assumed') is True
        if not counties[fips]['eligible'] or row.get('active_outage') or not (observed or assumed):
            continue
        risks = []
        for prefix, maximum in [('spc', 6), ('wpc', 4)]:
            available, risk = row[f'{prefix}_available'], float(row[f'{prefix}_risk'])
            if available not in (0, 1) or not math.isfinite(risk) or risk != int(risk):
                raise ValueError('Invalid weather availability or categorical rank')
            if (available == 0 and risk != -1) or (available == 1 and not 0 <= risk <= maximum):
                raise ValueError('Missing weather must be risk=-1/available=0; observed ranks must be valid')
            risks.extend([risk, float(available)])
        base = np.array([counties[fips]['latitude'], counties[fips]['longitude'], math.sin(angle), math.cos(angle), *risks], dtype=np.float32)
        hours = np.arange(24)
        hour_angle = (12 + hours) % 24 * (2 * np.pi / 24)
        design.append(np.column_stack((np.repeat(base[None, :], 24, axis=0), np.sin(hour_angle), np.cos(hour_angle), hours)).astype(np.float32))
        record.update(coverage='observed' if observed else 'scenario', at_risk_assumed=not observed)
        scored.append(record)
    if design:
        hazards = _county_hazards(np.vstack(design)).reshape(-1, 24)
        survival = np.column_stack((np.ones(len(hazards)), np.cumprod(1 - hazards, axis=1)[:, :-1]))
        first = hazards * survival
        for record, probabilities in zip(scored, first):
            record.update(p_first_start_by_hour=probabilities.tolist(), p_any_next_24h=float(probabilities.sum()))
    return dict(status='available', model_version='county-onset-HGB-isotonic-v1', issued_at_utc=stamp(issue),
        forecast_origin_utc=stamp(origin), input_cutoff_utc=stamp(cutoff), max_age_hours=24,
        grain='county_scenario', probability_kind='first_onset', intervals=intervals, records=records,
        scenario_definition='First recorded PNNL qualifying county outage scenario within the original 12Z–12Z window. Scenario coverage assumes no episode is active; this is not a household outage forecast or live outage observation.',
        provenance=dict(training_years='2018-2021', calibration_year=2022, forecast_origin_hour_utc=12,
            duration_available=False, active_episode_feed_available=any(r.get('at_risk_observed') is True for r in county_rows),
            model_manifest_sha256=hashlib.sha256((ASSETS / 'manifest.json').read_bytes()).hexdigest()))
