#!/usr/bin/env python3
"""Export existing fitted models and a historical replay; never fits a model.

Only maintainers with the original research artifacts need this script. Public
clones use the committed safe text/JSON assets directly via forecast_models.py.
"""
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_dam_context_experiments as price_source
import train_outage_models as outage_source

OUT = Path(__file__).resolve().parent
PRICE = ROOT / 'data/price/experiments/EXP002/folds/2025Q2/models.joblib'
ONSET = ROOT / 'data/outage/models/onset_county.joblib'
DURATION = ROOT / 'data/outage/models/duration_county.joblib'
BLEND = ROOT / 'data/price/experiments/BATTERY_BLEND'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')


def main():
    price, outage = joblib.load(PRICE), joblib.load(ONSET)
    names = []
    for name, model in price['models'].items():
        filename = f'exp002_{name}.txt'
        model.booster_.save_model(str(OUT / filename))
        names.append(filename)
    model, calibration = outage['model'], outage['calibration']
    trees = []
    for stage in model._predictors:
        assert len(stage) == 1
        nodes = stage[0].nodes
        assert not nodes['is_categorical'].any()
        fields = ['value', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'is_leaf']
        trees.append({name: nodes[name].tolist() for name in fields})
    save('onset_county.json', dict(
        family='HistGradientBoostingClassifier', baseline=float(model._baseline_prediction[0, 0]),
        features=outage['features'], trees=trees,
        calibration=dict(x=calibration.X_thresholds_.tolist(), y=calibration.y_thresholds_.tolist()),
        note='Tree values already include shrinkage. Sigmoid then clipped isotonic interpolation.'))
    names.append('onset_county.json')
    shutil.copyfile(DURATION, OUT / 'duration_county.joblib')
    names.append('duration_county.joblib')
    metrics = json.loads((ROOT / 'data/outage/models/metrics.json').read_text())
    counties = outage_source.load_counties()
    geo = json.loads(outage_source.GEO.read_text())
    names_by_fips = {f"48{int(f['properties']['Fips']):03d}": f['properties']['Name'] for f in geo['features']}
    save('counties.json', {fips: dict(latitude=lat, longitude=lon, name=names_by_fips[fips],
                                    eligible=fips in metrics['eligible_fips'])
                           for fips, (lat, lon) in counties.items()})
    names.append('counties.json')
    weight_rows = pd.read_csv(BLEND / 'weights.csv', float_precision='round_trip').to_dict('records')
    save('blend_history.json', weight_rows)
    names.append('blend_history.json')

    # This fixture is an existing historical forecast, never relabeled as live.
    raw = pd.read_csv(BLEND / 'prices.csv.gz', parse_dates=['delivery_date', 'interval_start_utc', 'interval_end_utc'])
    part = raw[raw.delivery_date == '2025-07-01'].copy().reset_index(drop=True)
    x = price_source.design(part, price_source.context_features(part), 'EXP002')
    p, _, _ = price_source.prediction(price['models'], x, part.dam_spp_usd_mwh.to_numpy())
    hourly = part[part.interval_start_utc.dt.minute == 0]
    hourly_rows = [dict(settlement_point=r.settlement_point, interval_start_utc=r.interval_start_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        dam_spp_usd_mwh=float(r.dam_spp_usd_mwh)) for r in hourly.itertuples()]
    expected_price = []
    for i, row in enumerate(part.itertuples()):
        expected_price.append(dict(settlement_point=row.settlement_point,
            interval_start_utc=row.interval_start_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            exp002_mean_usd_mwh=float(p['mean'][i]), rtm_mean_usd_mwh=float(row.adaptive_mean),
            **{f'rtm_{key}_usd_mwh': float(p[key][i]) for key in ('p10', 'p50', 'p90')}))
    weather = pd.read_csv(BLEND / 'outage_inputs/county_issue_inputs.csv.gz', dtype={'county_fips': str}, float_precision='round_trip')
    previous_weather = weather[pd.to_datetime(weather.issue_utc, utc=True) == pd.Timestamp('2025-06-30T12:00:00Z')]
    previous_weather_rows = [dict(county_fips=r.county_fips, spc_risk=r.spc_risk, spc_available=int(r.spc_forecast_available),
                         wpc_risk=r.wpc_risk, wpc_available=int(r.wpc_forecast_available), at_risk_assumed=True)
                    for r in previous_weather.itertuples()]
    weather = weather[pd.to_datetime(weather.issue_utc, utc=True) == pd.Timestamp('2025-07-01T12:00:00Z')]
    weather_rows = [dict(county_fips=r.county_fips, spc_risk=r.spc_risk, spc_available=int(r.spc_forecast_available),
                         wpc_risk=r.wpc_risk, wpc_available=int(r.wpc_forecast_available), at_risk_assumed=True)
                    for r in weather.itertuples()]
    forecast = pd.read_csv(BLEND / 'outage_inputs/forecasts.csv.gz', dtype={'county_fips': str}, float_precision='round_trip')
    previous_forecast = forecast[pd.to_datetime(forecast.issue_utc, utc=True) == pd.Timestamp('2025-06-30T12:00:00Z')]
    previous_expected_outage = {fips: group.sort_values('lead_hour').first_onset_probability.tolist()
                       for fips, group in previous_forecast.groupby('county_fips')}
    assert len(previous_weather_rows) == len(previous_expected_outage) == 4
    forecast = forecast[pd.to_datetime(forecast.issue_utc, utc=True) == pd.Timestamp('2025-07-01T12:00:00Z')]
    expected_outage = {fips: group.sort_values('lead_hour').first_onset_probability.tolist()
                       for fips, group in forecast.groupby('county_fips')}
    save('replay.json', dict(target_date='2025-07-01', price_issued_at_utc='2025-06-30T22:00:00Z',
        outage_issued_at_utc='2025-07-01T12:00:00Z', hourly_dam_rows=hourly_rows,
        county_rows=weather_rows, expected_price=expected_price, expected_outage=expected_outage,
        previous_outage_issued_at_utc='2025-06-30T12:00:00Z', previous_county_rows=previous_weather_rows,
        previous_expected_outage=previous_expected_outage,
        meaning='Historical fixture from the frozen battery experiment; timestamps must remain historical.'))
    names.append('replay.json')
    manifest = dict(format_version=1, model_fits=0,
        sources={str(p.relative_to(ROOT)): digest(p) for p in (PRICE, ONSET, DURATION, Path(price_source.__file__),
                    Path(outage_source.__file__), BLEND / 'weights.csv', BLEND / 'prices.csv.gz',
                    BLEND / 'outage_inputs/forecasts.csv.gz', BLEND / 'outage_inputs/county_issue_inputs.csv.gz')},
        assets_sha256={name: digest(OUT / name) for name in names},
        price=dict(family='LightGBM residual regression and quantile regression', features=price['features'],
            zones=price['zone_codes'], selected_iterations=price['selected_iterations'],
            training_period='2023-01-01 through 2025-03-30', training_rows=629728,
            mean='DAM + clipped least-squares weight * EXP002 residual',
            quantiles='Unblended EXP002 P10/P50/P90, sorted to prevent crossing; not an adaptive predictive interval.'),
        outage=dict(family='HistGradientBoostingClassifier + isotonic calibration', features=outage['features'],
            training_years='2018-2021', selection_calibration_year=2022, forecast_origin_hour_utc=12,
            eligible_counties=len(metrics['eligible_fips']), target='First recorded PNNL qualifying county scenario onset in 24 hours',
            duration_family='RandomForestClassifier + isotonic calibration', duration_asset='duration_county.joblib',
            limitations=['County events are not household outages.', 'No active episode is assumed unless observed status is supplied.',
                        'Missing recorded scenarios may reflect missing coverage.', 'Duration is unavailable without active episode observations.']),
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'lightgbm', 'scikit-learn')})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'assets': len(names), 'bytes': sum((OUT / name).stat().st_size for name in names),
                      'historical_price_rows': len(expected_price), 'historical_counties': len(expected_outage)}))


if __name__ == '__main__':
    main()
