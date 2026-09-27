#!/usr/bin/env python3
"""Build audited, retrospective ERCOT load-zone DAM/RTM training rows."""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

from analyze_dam_rt import RAW, ZONES, dam_prices, price_cents, workbook_rows


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "price"
MANIFEST = ROOT / "data" / "ercot" / "source_manifest.json"
CENTRAL = ZoneInfo("America/Chicago")
UTC = dt.timezone.utc
FIELDS = ("model_issue_utc", "input_cutoff_utc", "delivery_date", "settlement_point",
          "hour_ending", "quarter", "repeated_hour_flag", "interval_start_utc",
          "interval_end_utc", "dam_spp_usd_mwh", "rtm_spp_usd_mwh")


def utc_interval_start(day, hour_ending, quarter, repeated):
    if not 1 <= hour_ending <= 24 or not 1 <= quarter <= 4 or repeated not in ("N", "Y"):
        raise ValueError((day, hour_ending, quarter, repeated))
    naive = dt.datetime.combine(day, dt.time()) + dt.timedelta(
        hours=hour_ending - 1, minutes=15 * (quarter - 1))
    local = naive.replace(tzinfo=CENTRAL, fold=(repeated == "Y"))
    start = local.astimezone(UTC)
    if start.astimezone(CENTRAL).replace(tzinfo=None) != naive:
        raise ValueError(f"Nonexistent local interval: {naive}")
    if repeated == "Y" and naive.replace(tzinfo=CENTRAL, fold=0).utcoffset() == local.utcoffset():
        raise ValueError(f"Spurious repeated-hour flag: {naive}")
    return start


def expected_intervals(day):
    start = dt.datetime.combine(day, dt.time(), CENTRAL).astimezone(UTC)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), CENTRAL).astimezone(UTC)
    return int((end - start).total_seconds() // 900)


def iso(time):
    return time.isoformat().replace("+00:00", "Z")


def verified_zip(kind, year, manifest):
    info = manifest[f"{kind}_{year}"]
    path = RAW / f"{kind}_{year}_{info['doc_id']}.zip"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != info["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path, info


def build(years):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    target = OUT / f"price_rows_{min(years)}_{max(years)}.csv.gz"
    audit = {"status": "retrospective", "reason": "Annual archive prices do not prove original as-issued DAM vintages",
             "producer": "ERCOT", "distributor": "ERCOT public MIS annual archive",
             "price_unit": "USD/MWh", "model_issue_rule": "22:00 UTC on the previous calendar day",
             "input_cutoff_rule": "same as model issue; assumed DAM available, not verified per run",
             "original_dam_publication_verified": False, "revisions": "not recoverable from annual archives",
             "years": {}, "source_manifest": str(MANIFEST.relative_to(ROOT))}
    with tempfile.NamedTemporaryFile(dir=OUT, suffix=".csv.gz", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with gzip.open(temp, "wt", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for year in years:
                dam_path, dam_source = verified_zip("dam", year, manifest)
                rtm_path, rtm_source = verified_zip("rtm", year, manifest)
                dam, dates = dam_prices(dam_path)
                seen = set()
                day_zone = Counter()
                date_cache = {}
                type_count = Counter()
                unmatched = 0
                for row in workbook_rows(rtm_path):
                    point = row["Settlement Point Name"]
                    if point not in ZONES:
                        continue
                    type_count[row["Settlement Point Type"]] += 1
                    if row["Settlement Point Type"] != "LZ":
                        continue
                    date_str = row["Delivery Date"]
                    if date_str not in date_cache:
                        date_cache[date_str] = dt.datetime.strptime(date_str, "%m/%d/%Y").date()
                    day = date_cache[date_str]
                    he = int(row["Delivery Hour"])
                    q = int(row["Delivery Interval"])
                    repeated = row["Repeated Hour Flag"]
                    key = (date_str, he, repeated, point)
                    if key not in dam:
                        unmatched += 1
                        continue
                    start = utc_interval_start(day, he, q, repeated)
                    unique = (point, start)
                    if unique in seen:
                        raise ValueError(f"Duplicate UTC RTM interval: {unique}")
                    seen.add(unique)
                    day_zone[(day, point)] += 1
                    issue = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(22), UTC)
                    if not issue < start:
                        raise ValueError(f"Issue time after target: {issue} {start}")
                    writer.writerow({"model_issue_utc": iso(issue), "input_cutoff_utc": iso(issue),
                                     "delivery_date": day.isoformat(), "settlement_point": point,
                                     "hour_ending": he, "quarter": q, "repeated_hour_flag": repeated,
                                     "interval_start_utc": iso(start),
                                     "interval_end_utc": iso(start + dt.timedelta(minutes=15)),
                                     "dam_spp_usd_mwh": dam[key] / 100,
                                     "rtm_spp_usd_mwh": price_cents(row["Settlement Point Price"]) / 100})
                bad_days = [(str(day), zone, count, expected_intervals(day)) for (day, zone), count in day_zone.items()
                            if count != expected_intervals(day)]
                if unmatched or bad_days or len(day_zone) != len(dates) * len(ZONES) or len(seen) != 4 * len(dam):
                    raise ValueError(f"Coverage failed for {year}: unmatched={unmatched}, bad_days={bad_days[:5]}")
                audit["years"][str(year)] = {
                    "days": len(dates), "zones": len(ZONES), "matched_rows": len(seen),
                    "dam_hours": len(dam), "unmatched_rtm": unmatched,
                    "rtm_point_type_counts_for_zone_names": dict(type_count),
                    "dst_days": {str(day): expected_intervals(day) for day in sorted(set(d for d, _ in day_zone))
                                 if expected_intervals(day) != 96},
                    "dam_source": {"file": str(dam_path.relative_to(ROOT)), **dam_source},
                    "rtm_source": {"file": str(rtm_path.relative_to(ROOT)), **rtm_source}}
                print(f"{year}: {len(seen):,} matched load-zone quarters", flush=True)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    audit_name = ("source_audit.json" if years == [2023, 2024, 2025]
                  else f"source_audit_{min(years)}_{max(years)}.json")
    (OUT / audit_name).write_text(json.dumps(audit, indent=2) + "\n")
    print(target)


def self_test():
    spring = dt.date(2025, 3, 9)
    fall = dt.date(2025, 11, 2)
    assert expected_intervals(spring) == 92 and expected_intervals(fall) == 100
    assert (utc_interval_start(fall, 2, 1, "Y") - utc_interval_start(fall, 2, 1, "N")
            == dt.timedelta(hours=1))
    try:
        utc_interval_start(spring, 3, 1, "N")
    except ValueError:
        pass
    else:
        raise AssertionError("Nonexistent spring hour accepted")
    assert price_cents("34.380000000000003") == 3438


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
    else:
        build(sorted(set(args.years)))
