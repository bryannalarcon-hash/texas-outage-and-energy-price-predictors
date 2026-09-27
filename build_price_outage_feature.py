#!/usr/bin/env python3
"""Build a 2023 county-onset overlay for four broad load-zone price series.

Run: .venv/bin/python build_price_outage_feature.py [--self-test]
The source polygons are a third-party geographic approximation; their county
matches are not ERCOT settlement assignments. No 2023 outcomes are read.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.prepared import prep
from shapely.validation import make_valid

from train_outage_models import (ONSET_COLUMNS, date_features, load_counties,
                                 onset_matrix, onset_probabilities, predict_candidate)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/price"
OUTLOOK = ROOT / "data/outage/forecast/outlook_county_day_2018_2023.csv.gz"
OUTLOOK_MANIFEST = ROOT / "data/outage/forecast/outlook_sources.json"
MODEL = ROOT / "data/outage/models/onset_county.joblib"
METRICS = ROOT / "data/outage/models/metrics.json"
EXAMPLE = ROOT / "data/outage/models/onset_example.json"
COUNTY_GEOMETRY = ROOT / "data/outage/raw/texas_counties_fips.geojson"
ZONE_SOURCE = OUT / "ercot_load_zones_third_party.geojson"
ZONE_MAP = OUT / "outage_county_lz_map.csv"
FEATURE = OUT / "outage_risk_2023.csv.gz"
AUDIT = OUT / "outage_risk_2023_audit.json"
ZONE_URL = ("https://services3.arcgis.com/fwwoCWVtaahwlvxO/ArcGIS/rest/services/"
            "ERCOT_Load_Zones/FeatureServer/7/query?where=1%3D1&outFields=NAME"
            "&returnGeometry=true&outSR=4326&geometryPrecision=5&f=geojson")
ZONES = {name: f"LZ_{name.upper()}" for name in ("Houston", "North", "South", "West")}
UTC = dt.timezone.utc
FIELDS = ("issue_utc", "valid_start_utc", "valid_end_utc", "lead_hour",
          "settlement_point", "county_count", "mean_county_first_onset_probability")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def source_polygons():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ZONE_SOURCE.exists():
        request = urllib.request.Request(ZONE_URL, headers={"User-Agent": "bpc-price-outage/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        document = json.loads(content)
        if document.get("type") != "FeatureCollection" or len(document.get("features", [])) != 4:
            raise ValueError("Unexpected ArcGIS load-zone response")
        temporary = ZONE_SOURCE.with_suffix(".geojson.part")
        temporary.write_bytes(content)
        os.replace(temporary, ZONE_SOURCE)
    document = json.loads(ZONE_SOURCE.read_bytes())
    if document.get("type") != "FeatureCollection":
        raise ValueError("Cached load-zone source is not GeoJSON")
    features = document.get("features", [])
    names = [feature.get("properties", {}).get("NAME") for feature in features]
    if len(names) != 4 or set(names) != set(ZONES):
        raise ValueError(f"Unexpected load-zone names: {names}")
    polygons = {}
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            raise ValueError(f"Empty polygon: {feature['properties']['NAME']}")
        polygons[feature["properties"]["NAME"]] = prep(make_valid(geometry))
    return polygons


def county_zone_map(eligible, counties, polygons):
    assignments = {}
    counts = Counter()
    rows = []
    for fips in sorted(eligible):
        lat, lon = counties[fips]
        point = Point(lon, lat)
        matches = [ZONES[name] for name, polygon in polygons.items() if polygon.contains(point)]
        status = "mapped" if len(matches) == 1 else "outside_zone_polygons" if not matches else "ambiguous"
        zone = matches[0] if status == "mapped" else ""
        if zone:
            assignments[fips] = zone
        counts[status] += 1
        rows.append((fips, lat, lon, zone, status))
    if len(assignments) != sum(1 for row in rows if row[4] == "mapped"):
        raise AssertionError("County mapping lost an assignment")
    return assignments, counts, rows


def read_outlooks(mapped_fips):
    by_day = defaultdict(dict)
    unavailable = Counter()
    unavailable_days = defaultdict(set)
    with gzip.open(OUTLOOK, "rt", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            day = row["date_utc"]
            fips = row["county_fips"]
            if not day.startswith("2023-") or fips not in mapped_fips:
                continue
            issue = stamp(row["issue_utc"])
            if issue != dt.datetime.combine(dt.date.fromisoformat(day), dt.time(12), UTC):
                raise ValueError(f"Bad outlook issue time: {day} {fips}")
            values = []
            for prefix, maximum in (("spc", 6), ("wpc", 4)):
                available = row[f"{prefix}_forecast_available"]
                risk = row[f"{prefix}_risk"]
                product = row[f"{prefix}_product_issued_utc"]
                if available == "1":
                    if (not risk or not 0 <= int(risk) <= maximum or not product
                            or not stamp(product) < issue
                            or not stamp(row[f"{prefix}_valid_start_utc"]) <= issue
                            or not stamp(row[f"{prefix}_valid_end_utc"]) >= issue + dt.timedelta(hours=24)):
                        raise ValueError(f"Later, invalid, or incomplete {prefix} outlook: {day} {fips}")
                    values.extend((float(risk), 1.))
                elif available == "0" and not risk and not product:
                    values.extend((-1., 0.))
                    unavailable[prefix] += 1
                    unavailable_days[prefix].add(day)
                else:
                    raise ValueError(f"Invalid {prefix} availability: {day} {fips}")
            if fips in by_day[day]:
                raise ValueError(f"Duplicate outlook row: {day} {fips}")
            by_day[day][fips] = tuple(values)
    expected_days = [(dt.date(2023, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(365)]
    for day in expected_days:
        if set(by_day[day]) != mapped_fips:
            raise ValueError(f"Missing mapped-county outlook rows for {day}")
    if len(by_day) != 365:
        raise ValueError(f"Unexpected 2023 outlook dates: {len(by_day)}")
    return by_day, unavailable, {kind: sorted(days) for kind, days in unavailable_days.items()}


def forecast_day(issue, outlooks, counties, assignments, artifact):
    doy_sin, doy_cos = date_features(issue)
    fips_order = sorted(assignments)
    records = []
    for fips in fips_order:
        lat, lon = counties[fips]
        spc, spc_available, wpc, wpc_available = outlooks[fips]
        # Dummy -1 is required by onset_matrix's API; it affects labels, never X.
        records.append((lat, lon, doy_sin, doy_cos, spc, spc_available,
                        wpc, wpc_available, -1))
    columns = ["latitude", "longitude", "doy_sin", "doy_cos", "spc_risk",
               "spc_available", "wpc_risk", "wpc_available", "first_hour"]
    x, _, _ = onset_matrix(pd.DataFrame.from_records(records, columns=columns))
    hazards = artifact["calibration"].predict(
        predict_candidate(artifact["model"], x)).reshape(len(fips_order), 24)
    first, _ = onset_probabilities(hazards)
    return fips_order, first


def self_test():
    first, total = onset_probabilities([[0.1, 0.2] + [0.] * 22])
    assert np.allclose(first[0, :2], [.1, .18]) and abs(total[0] - .28) < 1e-5
    artifact = joblib.load(MODEL)
    assert artifact["features"] == ONSET_COLUMNS and artifact["county_grain_only"]
    with gzip.open(OUTLOOK, "rt", newline="") as file:
        row = next(row for row in csv.DictReader(file)
                   if row["date_utc"] == "2023-01-01" and row["county_fips"] == "48001")
    issue = stamp(row["issue_utc"])
    outlook = (float(row["spc_risk"]), float(row["spc_forecast_available"]),
               float(row["wpc_risk"]), float(row["wpc_forecast_available"]))
    _, probabilities = forecast_day(issue, {"48001": outlook}, load_counties(),
                                    {"48001": "LZ_NORTH"}, artifact)
    expected = json.loads(EXAMPLE.read_text())["p_first_start_by_hour"]
    assert np.allclose(probabilities[0], expected, atol=5e-7)


def build():
    artifact = joblib.load(MODEL)
    if artifact["features"] != ONSET_COLUMNS or not artifact["county_grain_only"]:
        raise ValueError("Frozen onset model has an unexpected feature contract")
    metrics = json.loads(METRICS.read_text())
    eligible = set(metrics["eligible_fips"])
    counties = load_counties()
    if len(eligible) != 186 or not eligible <= set(counties):
        raise ValueError("Unexpected eligible county set")
    assignments, map_counts, map_rows = county_zone_map(eligible, counties, source_polygons())
    per_zone = Counter(assignments.values())
    if set(per_zone) != set(ZONES.values()):
        raise ValueError(f"One or more broad zones have no mapped counties: {per_zone}")
    if assignments.get("48029") != "LZ_SOUTH" or assignments.get("48453") != "LZ_SOUTH":
        raise ValueError("Bexar/Travis broad-region overlap examples changed; review the source map")
    outlooks, unavailable, unavailable_days = read_outlooks(set(assignments))
    map_temp = ZONE_MAP.with_suffix(".csv.part")
    with map_temp.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("county_fips", "centroid_lat", "centroid_lon", "associated_broad_price_series", "match_status"))
        writer.writerows(map_rows)
    os.replace(map_temp, ZONE_MAP)
    feature_temp = FEATURE.with_suffix(".gz.part")
    with gzip.open(feature_temp, "wt", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for day in sorted(outlooks):
            issue = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(12), UTC)
            order, first = forecast_day(issue, outlooks[day], counties, assignments, artifact)
            indices = {zone: [i for i, fips in enumerate(order) if assignments[fips] == zone]
                       for zone in sorted(per_zone)}
            for hour in range(24):
                start = issue + dt.timedelta(hours=hour)
                for zone, county_indices in indices.items():
                    writer.writerow({"issue_utc": iso(issue), "valid_start_utc": iso(start),
                                     "valid_end_utc": iso(start + dt.timedelta(hours=1)),
                                     "lead_hour": hour, "settlement_point": zone,
                                     "county_count": len(county_indices),
                                     "mean_county_first_onset_probability":
                                         float(first[county_indices, hour].mean())})
    os.replace(feature_temp, FEATURE)
    audit = {
        "feature_semantics": "Unweighted mean of county probability that its first recorded qualifying episode in the 24-hour window starts in this hour; a four-region weather/outage overlay associated with broad LZ price series, not county settlement membership, household outage probability, or expected MW lost",
        "issue_rule": "12:00 UTC daily in 2023; each issued forecast is valid for the following 24 UTC hours only",
        "inference_only": "No future PNNL starts, observed outage outcomes, or labeled county_day_issues.csv.gz used",
        "model": {"path": str(MODEL.relative_to(ROOT)), "sha256": digest(MODEL),
                  "fit_years": "2018-2021", "selection_and_calibration_year": 2022,
                  "held_out_test_year": 2023, "last_evaluated_issue_utc": "2023-12-30T12:00:00Z",
                  "county_grain_only": True,
                  "eligible_fips_source": str(METRICS.relative_to(ROOT))},
        "weather": {"path": str(OUTLOOK.relative_to(ROOT)), "sha256": digest(OUTLOOK),
                    "source_manifest": str(OUTLOOK_MANIFEST.relative_to(ROOT)),
                    "source_manifest_sha256": digest(OUTLOOK_MANIFEST),
                    "producer": "NOAA SPC/WPC", "distributor": "Iowa Environmental Mesonet reconstructed outlook archive",
                    "as_issued_check": "product issue strictly before daily 12:00 UTC cutoff; validity covers all 24 hours",
                    "unavailable_mapped_county_days": dict(unavailable),
                    "unavailable_dates": unavailable_days,
                    "limitation": "Exact historical product distribution latency is not archived"},
        "geography": {"source_url": ZONE_URL, "cache": str(ZONE_SOURCE.relative_to(ROOT)),
                      "sha256": digest(ZONE_SOURCE), "method": "county centroid strictly inside exactly one polygon",
                      "status": "Third-party ArcGIS four-region overlay, approximate and not verified as ERCOT's historical 2023 settlement assignments; counties can span regions and utility-specific settlement areas lie within broad polygons",
                      "price_series_join_note": "CSV settlement_point names the broad price series receiving this regional feature; it does not assert that every contributing county settles at that point",
                      "utility_specific_overlap_examples": [
                          {"county_fips": "48029", "county": "Bexar", "associated_broad_price_series": "LZ_SOUTH", "note": "CPS area overlaps broad South geography; CPS has a separate ERCOT settlement price series"},
                          {"county_fips": "48453", "county": "Travis", "associated_broad_price_series": "LZ_SOUTH", "note": "Austin Energy/LCRA areas overlap broad South geography; AEN and LCRA have separate ERCOT settlement price series"}],
                      "comparison_sources": {
                          "ercot_load_zone_map": "https://www.ercot.com/news/mediakit/maps/index",
                          "ercot_settlement_price_series": "https://www.ercot.com/content/cdr/html/real_time_spp.html",
                          "ercot_imm_broad_zone_rollup": "https://www.ercot.com/files/docs/2018/04/04/5_Independent_Market_Monitor__IMM__Report.pdf"},
                      "county_geometry": str(COUNTY_GEOMETRY.relative_to(ROOT)),
                      "county_geometry_sha256": digest(COUNTY_GEOMETRY),
                      "eligible_texas_counties": len(eligible), "match_counts": dict(map_counts),
                      "mapped_counties_by_zone": dict(sorted(per_zone.items())),
                      "excluded_fips": [row[0] for row in map_rows if row[4] != "mapped"],
                      "exclusion_note": "Centroids outside all four polygons are omitted to avoid assigning non-ERCOT locations. This does not prove an excluded county is wholly outside ERCOT.",
                      "utility_specific_zones": "no utility-specific feature produced; broad South overlay can include counties overlapping those areas",
                      "map_csv": str(ZONE_MAP.relative_to(ROOT)), "map_sha256": digest(ZONE_MAP)},
        "output": {"path": str(FEATURE.relative_to(ROOT)), "sha256": digest(FEATURE),
                   "columns": list(FIELDS), "issues": 365, "hours_per_issue": 24,
                   "broad_zones": sorted(per_zone), "rows": 365 * 24 * len(per_zone),
                   "first_issue_utc": "2023-01-01T12:00:00Z",
                   "last_issue_utc": "2023-12-31T12:00:00Z",
                   "last_valid_end_utc": "2024-01-01T12:00:00Z",
                   "next_day_price_join_limit": "Only target intervals within [issue_utc, issue_utc + 24h) and after that price run's cutoff may use this issue; no later issue backfill. A next-day ERCOT operating day is only partly covered."},
        "limitations": ["The model was trained on at-risk county-days; no point-in-time active-episode mask is available here, so predictions are issued for all eligible mapped counties.",
                        "County mean gives every mapped county equal weight, not load or customer weight.",
                        "The 2023-12-31 issue extends beyond the pilot's observed-label boundary and was not part of its held-out test.",
                        "Source polygons were retrieved as currently served; historical 2023 map vintage is unverified."]}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {FEATURE}: {audit['output']['rows']:,} zone-hour rows; {dict(sorted(per_zone.items()))} counties")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
    else:
        build()
