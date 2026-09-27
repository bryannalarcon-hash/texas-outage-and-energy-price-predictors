#!/usr/bin/env python3
"""Build 12:00 UTC Texas county-centroid outlook features for 2018–2023.

Run: .venv/bin/python build_outlook_features.py
Check: .venv/bin/python build_outlook_features.py --check
Needs pyshp and shapely in the project-local virtual environment.

NOAA SPC/WPC issue the forecasts; Iowa Environmental Mesonet parses their
issued text products into the bulk shapefile archive used here. Archive URLs
and SHA-256 hashes are written beside the output. An empty rank means that no
eligible, full-24-hour archived product was found; rank zero means an eligible
product was found but its outlook did not cover the county centroid.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import shapefile
from shapely.geometry import Point, shape
from shapely.prepared import prep
from shapely.validation import make_valid


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/outage/forecast"
COUNTIES = ROOT / "data/outage/raw/texas_counties_fips.geojson"
RANKS = {
    "C": {"TSTM": 1, "MRGL": 2, "SLGT": 3, "ENH": 4, "MDT": 5, "HIGH": 6},
    "E": {"MRGL": 1, "SLGT": 2, "MDT": 3, "HIGH": 4},
}
FIELDS = [
    "date_utc", "issue_utc", "county_fips", "county_name", "centroid_lon", "centroid_lat",
    "spc_risk", "spc_forecast_available", "spc_category", "spc_product_issued_utc", "spc_valid_start_utc", "spc_valid_end_utc",
    "wpc_risk", "wpc_forecast_available", "wpc_category", "wpc_product_issued_utc", "wpc_valid_start_utc", "wpc_valid_end_utc",
]


def stamp(value):
    return datetime.strptime(value, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def archive_url(kind, year):
    return (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/outlooks.py"
        f"?d=1&type={kind}&sts={year}-01-01T00:00Z&ets={year + 1}-01-01T00:00Z"
    )


def download(kind, year):
    dest = OUT / "raw" / f"{kind}_day1_{year}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = archive_url(kind, year)
    if not dest.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "bpc-outage-research/1.0"})
                with urllib.request.urlopen(request, timeout=180) as response, dest.with_suffix(".part").open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                dest.with_suffix(".part").replace(dest)
                break
            except Exception:
                dest.with_suffix(".part").unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
    with zipfile.ZipFile(dest) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt archive: {dest}")
        if not {".shp", ".shx", ".dbf", ".prj"}.issubset({Path(name).suffix for name in archive.namelist()}):
            raise ValueError(f"Incomplete shapefile: {dest}")
    return dest, {"kind": kind, "year": year, "url": url, "file": str(dest.relative_to(ROOT)),
                  "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "bytes": dest.stat().st_size}


def counties():
    features = json.loads(COUNTIES.read_text())["features"]
    result = []
    for feature in features:
        properties = feature["properties"]
        centroid = shape(feature["geometry"]).centroid
        result.append((f"48{int(properties['Fips']):03d}", properties["Name"],
                       round(centroid.x, 6), round(centroid.y, 6), centroid))
    result.sort()
    if len(result) != 254 or len({county[0] for county in result}) != 254:
        raise ValueError("Expected 254 unique Texas county FIPS codes")
    return result


def reader(path):
    with zipfile.ZipFile(path) as archive:
        members = {Path(name).suffix: io.BytesIO(archive.read(name)) for name in archive.namelist()}
    return shapefile.Reader(shp=members[".shp"], shx=members[".shx"], dbf=members[".dbf"])


def choose_products(records, year):
    """One latest archived product issued before each 12Z cutoff and valid for all 24h."""
    chosen = {}
    day = date(year, 1, 1)
    while day.year == year:
        cutoff = day.strftime("%Y%m%d") + "1200"
        end = (day + timedelta(days=1)).strftime("%Y%m%d") + "1200"
        eligible = ((r["PRODISS"], r["ISSUE"], r["EXPIRE"]) for r in records
                    if r["PRODISS"] < cutoff and r["ISSUE"] <= cutoff and r["EXPIRE"] >= end)
        chosen[day.isoformat()] = max(eligible, default=None)
        day += timedelta(days=1)
    return chosen


def risk_by_county(path, kind, year, county_rows):
    source = reader(path)
    records = [record.as_dict() for record in source.iterRecords()]
    if any(record["TYPE"] != kind or record["DAY"] != 1 for record in records):
        raise ValueError(f"Unexpected outlook type/day in {path}")
    chosen = choose_products(records, year)
    results = {day: [0] * len(county_rows) for day, product in chosen.items() if product}
    for index, record in enumerate(records):
        day = record["ISSUE"][:8]
        key = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        if key not in results or (record["PRODISS"], record["ISSUE"], record["EXPIRE"]) != chosen[key]:
            continue
        if record["CATEGORY"] != "CATEGORICAL":
            continue
        threshold = record["THRESHOLD"].strip()
        if threshold not in RANKS[kind]:
            raise ValueError(f"Unexpected {kind} category {threshold!r} in {path}")
        polygon = shape(source.shape(index).__geo_interface__)
        if polygon.is_empty:
            raise ValueError(f"Empty geometry for {kind} {threshold} on {key}")
        prepared = prep(polygon if polygon.is_valid else make_valid(polygon))
        rank = RANKS[kind][threshold]
        for county_index, county in enumerate(county_rows):
            if rank > results[key][county_index] and prepared.covers(county[4]):
                results[key][county_index] = rank
    return chosen, results


def self_check():
    cutoff = "202301011200"
    rows = [
        {"PRODISS": "202301010600", "ISSUE": cutoff, "EXPIRE": "202301021200"},
        {"PRODISS": "202301011300", "ISSUE": cutoff, "EXPIRE": "202301021200"},
        {"PRODISS": "202301011159", "ISSUE": "202301011300", "EXPIRE": "202301021200"},
        {"PRODISS": "202301010700", "ISSUE": cutoff, "EXPIRE": "202301020600"},
    ]
    assert choose_products(rows, 2023)["2023-01-01"] == ("202301010600", cutoff, "202301021200")
    assert choose_products(rows[1:], 2023)["2023-01-01"] is None
    assert prep(Point(-100, 30).buffer(1)).covers(Point(-100, 30))


def check_output(path):
    previous = ("", "")
    count = 0
    with gzip.open(path, "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != FIELDS:
            raise ValueError("Unexpected output schema")
        for row in rows:
            day = row["date_utc"]
            issue = day + "T12:00:00Z"
            end = (date.fromisoformat(day) + timedelta(days=1)).isoformat() + "T12:00:00Z"
            key = (row["issue_utc"], row["county_fips"])
            if key <= previous or row["issue_utc"] != issue:
                raise ValueError(f"Duplicate, unsorted, or mistimed row: {key}")
            previous = key
            for kind in ("spc", "wpc"):
                rank = row[f"{kind}_risk"]
                available = row[f"{kind}_forecast_available"]
                if available == "0":
                    if rank or row[f"{kind}_category"] or row[f"{kind}_product_issued_utc"]:
                        raise ValueError(f"Missing product encoded as risk: {key}")
                elif available == "1":
                    if (not row[f"{kind}_product_issued_utc"] < issue
                            or not row[f"{kind}_valid_start_utc"] <= issue
                            or not row[f"{kind}_valid_end_utc"] >= end):
                        raise ValueError(f"Forecast issue or validity leakage: {key}")
                    if dict(NONE=0, **RANKS["C" if kind == "spc" else "E"])[row[f"{kind}_category"]] != int(rank):
                        raise ValueError(f"Risk/category mismatch: {key}")
                else:
                    raise ValueError(f"Invalid availability flag: {key}")
            count += 1
    if count != 254 * ((date(2024, 1, 1) - date(2018, 1, 1)).days):
        raise ValueError(f"Wrong county-day count: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run the issue-time and geometry check only")
    args = parser.parse_args()
    self_check()
    if args.check:
        output = OUT / "outlook_county_day_2018_2023.csv.gz"
        count = check_output(output) if output.exists() else 0
        print(f"Issue-time, full-horizon, geometry, and output checks passed ({count:,} rows)")
        return

    county_rows = counties()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "outlook_county_day_2018_2023.csv.gz"
    manifest = []
    with gzip.open(output, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for year in range(2018, 2024):
            sources = {}
            for kind in ("C", "E"):
                path, meta = download(kind, year)
                manifest.append(meta)
                sources[kind] = risk_by_county(path, kind, year, county_rows)
            day = date(year, 1, 1)
            missing = {kind: 0 for kind in sources}
            while day.year == year:
                day_text = day.isoformat()
                issued = day_text + "T12:00:00Z"
                product_info = {}
                for kind in ("C", "E"):
                    product = sources[kind][0][day_text]
                    product_info[kind] = tuple(stamp(value) for value in product) if product else None
                for county_index, (fips, name, lon, lat, _) in enumerate(county_rows):
                    row = [day_text, issued, fips, name, lon, lat]
                    for kind in ("C", "E"):
                        _, risks = sources[kind]
                        if product_info[kind] is None:
                            row.extend(["", 0, "", "", "", ""])
                            if county_index == 0:
                                missing[kind] += 1
                        else:
                            rank = risks[day_text][county_index]
                            category = next((category for category, value in RANKS[kind].items() if value == rank), "NONE")
                            row.extend([rank, 1, category, *product_info[kind]])
                    writer.writerow(row)
                day += timedelta(days=1)
            print(f"{year}: {254 * (day - date(year, 1, 1)).days:,} county-days; missing full-horizon days SPC={missing['C']}, WPC={missing['E']}")
    (OUT / "outlook_sources.json").write_text(json.dumps({
        "output": output.name,
        "forecast_cutoff_utc": "12:00",
        "horizon_hours": 24,
        "feature_schema": FIELDS,
        "rank_map": {"spc": {"NONE": 0, **RANKS["C"]}, "wpc_ero": {"NONE": 0, **RANKS["E"]}},
        "producer": {"C": "NOAA/NWS Storm Prediction Center", "E": "NOAA/NWS Weather Prediction Center"},
        "processor": "Iowa Environmental Mesonet; PTS text-product parsing, not the original SPC/WPC GIS polygon",
        "processor_documentation": "https://mesonet.agron.iastate.edu/request/gis/outlooks.phtml",
        "county_geometry": str(COUNTIES.relative_to(ROOT)),
        "county_geometry_source": "https://services.twdb.texas.gov/arcgis/rest/services/PWS/Texas_Counties_FIPS/FeatureServer/0",
        "archives": manifest,
        "limitations": ["County centroid only; no county-wide exposure fraction.",
                        "IEM retrospectively reparses NOAA text products; exact historical feed latency is not archived.",
                        "SPC covers convective storm risk and WPC ERO covers excessive-rainfall risk, not all outage causes.",
                        "An absent full-horizon product is missing, never a zero-risk observation."]
    }, indent=2) + "\n")
    check_output(output)
    print(f"Wrote {output} and outlook_sources.json")


if __name__ == "__main__":
    main()
