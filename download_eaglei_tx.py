#!/usr/bin/env python3
"""Download ORNL EAGLE-I CSVs and retain Texas reports with source checksums."""

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "outage" / "raw"
ARTICLE = "https://api.figshare.com/v2/articles/24237376"


def texas_rows(source, target):
    reader = csv.DictReader(source)
    assert reader.fieldnames in (["fips_code", "county", "state", "customers_out", "run_start_time"],
                                 ["fips_code", "county", "state", "sum", "run_start_time"])
    count_field = "customers_out" if "customers_out" in reader.fieldnames else "sum"
    writer = csv.DictWriter(target, fieldnames=["fips_code", "county", "state", "customers_out", "run_start_time"])
    writer.writeheader()
    rows = zeros = missing = 0
    for row in reader:
        if row["state"] == "Texas":
            count = row[count_field]
            writer.writerow({"fips_code": row["fips_code"], "county": row["county"], "state": row["state"],
                             "customers_out": count, "run_start_time": row["run_start_time"]})
            rows += 1
            zeros += count == "0"
            missing += count == ""
    return rows, zeros, missing


def download_file(item, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".source.download")
    if not temporary.exists():
        try:
            with urllib.request.urlopen(item["download_url"], timeout=120) as source, temporary.open("wb") as temp:
                while block := source.read(8 * 1024 * 1024):
                    temp.write(block)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    digest = hashlib.md5()
    with temporary.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    if temporary.stat().st_size != item["size"] or digest.hexdigest() != item["computed_md5"]:
        temporary.unlink()
        raise ValueError(f"Source size/checksum mismatch for {item['name']}")
    if item["name"].startswith("eaglei_outages_"):
        with temporary.open(newline="") as source, output.open("w", newline="") as target:
            rows, zeros, missing = texas_rows(source, target)
        temporary.unlink()
        assert rows > 0
        detail = {"texas_rows": rows, "texas_zero_rows": zeros, "texas_missing_count_rows": missing}
    else:
        temporary.replace(output)
        detail = {}
    return {"source_name": item["name"], "source_url": item["download_url"],
            "source_md5": item["computed_md5"], "source_bytes": item["size"],
            "texas_path": str(output.relative_to(ROOT)), **detail}


def self_test():
    from io import StringIO
    source = StringIO("fips_code,county,state,customers_out,run_start_time\n48001,A,Texas,0,2023-01-01 00:00:00\n01001,B,Alabama,5,2023-01-01 00:00:00\n")
    target = StringIO()
    assert texas_rows(source, target) == (1, 1, 0)
    assert "Alabama" not in target.getvalue()
    source = StringIO("fips_code,county,state,sum,run_start_time\n48001,A,Texas,,2023-01-01 00:00:00\n")
    assert texas_rows(source, StringIO()) == (1, 0, 1)


def main(years):
    with urllib.request.urlopen(ARTICLE, timeout=30) as response:
        article = json.load(response)
    files = {item["name"]: item for item in article["files"]}
    manifest_path = DEST / "eaglei_tx_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in ["coverage_history.csv", "DQI.csv", *(f"eaglei_outages_{year}.csv" for year in years)]:
        item = files[name]
        output = DEST / (f"eaglei_tx_{name[-8:-4]}.csv" if name.startswith("eaglei_outages_") else name)
        previous = manifest.get(name)
        if output.exists() and previous and previous["source_md5"] == item["computed_md5"]:
            print(f"Cached {name}: {previous.get('texas_rows', 'metadata')} rows", flush=True)
            continue
        print(f"Downloading {name} ({item['size'] / 1e6:.0f} MB)", flush=True)
        manifest[name] = download_file(item, output)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Saved {output.name}: {manifest[name].get('texas_rows', 'metadata')} rows", flush=True)


if __name__ == "__main__":
    self_test()
    if sys.argv[1:] == ["--self-test"]:
        print("self-test passed")
    else:
        main([int(arg) for arg in sys.argv[1:]] or list(range(2018, 2024)))
