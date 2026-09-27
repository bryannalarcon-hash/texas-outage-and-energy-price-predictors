#!/usr/bin/env python3
"""Download ERCOT's annual load-zone prices and compare DAM with RTM."""

import csv
import datetime as dt
import hashlib
import html.parser
import io
import json
import math
import os
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "ercot" / "raw"
REPORTS = {"dam": 13060, "rtm": 13061}
ZONES = ("LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
BASE = "https://www.ercot.com"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "bpc-price-study/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def documents(report_id):
    url = f"{BASE}/misapp/servlets/IceDocListJsonWS?reportTypeId={report_id}"
    listing = json.loads(get(url))["ListDocsByRptTypeRes"]["DocumentList"]
    return {int(item["Document"]["FriendlyName"][-4:]): item["Document"] for item in listing}


def download(kind, year, document):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{kind}_{year}_{document['DocID']}.zip"
    if not path.exists():
        url = f"{BASE}/misdownload/servlets/mirDownload?doclookupId={document['DocID']}"
        data = get(url)
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise ValueError(f"ERCOT did not return a ZIP for {kind} {year}")
        with tempfile.NamedTemporaryFile(dir=RAW, delete=False) as temp:
            temp.write(data)
            temporary = Path(temp.name)
        os.replace(temporary, path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid cached ZIP: {path}")
    return path


def workbook_rows(outer_path):
    """Read the simple ERCOT XLSX tables using only the Python standard library."""
    with zipfile.ZipFile(outer_path) as outer:
        xlsx = next(name for name in outer.namelist() if name.lower().endswith(".xlsx"))
        with zipfile.ZipFile(io.BytesIO(outer.read(xlsx))) as book:
            strings = []
            with book.open("xl/sharedStrings.xml") as shared:
                for _, element in ET.iterparse(shared, events=("end",)):
                    if element.tag == NS + "si":
                        strings.append("".join(t.text or "" for t in element.iter(NS + "t")))
                        element.clear()
            sheets = sorted(
                (name for name in book.namelist() if name.lower().startswith("xl/worksheets/sheet") and name.endswith(".xml")),
                key=lambda name: int(Path(name).stem.lower().removeprefix("sheet")),
            )
            for sheet in sheets:
                with book.open(sheet) as source:
                    context = ET.iterparse(source, events=("start", "end"))
                    root = None
                    header = None
                    for event, element in context:
                        if root is None:
                            root = element
                        if event != "end" or element.tag != NS + "row":
                            continue
                        cells = []
                        for cell in element.findall(NS + "c"):
                            value = cell.find(NS + "v")
                            text = "" if value is None or value.text is None else value.text
                            if cell.get("t") == "s":
                                text = strings[int(text)]
                            cells.append(text)
                        if header is None:
                            header = cells
                        elif cells:
                            if len(cells) != len(header):
                                if len(cells) == 3 and cells[0].isdigit():
                                    # A stray numeric Excel row follows the 2011 RTM October table.
                                    print(f"Skipping non-price row in {outer_path.name}/{sheet}", file=sys.stderr)
                                    root.clear()
                                    continue
                                raise ValueError(f"Unexpected columns in {sheet}: {cells[:3]}")
                            yield dict(zip(header, cells))
                        root.clear()


def price_cents(value):
    # XLSX numeric cells can contain binary-float tails such as 34.380000000000003.
    return int((Decimal(value) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def dam_prices(path):
    prices = {}
    dates = set()
    for row in workbook_rows(path):
        point = row["Settlement Point"]
        if point not in ZONES:
            continue
        key = (row["Delivery Date"], int(row["Hour Ending"].split(":")[0]), row["Repeated Hour Flag"], point)
        if key in prices:
            raise ValueError(f"Duplicate DAM interval: {key}")
        prices[key] = price_cents(row["Settlement Point Price"])
        dates.add(key[0])
    return prices, dates


def summarize_year(year, dam_path, rtm_path, totals):
    dam, dates = dam_prices(dam_path)
    seen = set()
    unmatched = 0
    for row in workbook_rows(rtm_path):
        point = row["Settlement Point Name"]
        if point not in ZONES or row["Settlement Point Type"] != "LZ":
            continue  # LZEW is a different energy-weighted series with the same zone name.
        interval = int(row["Delivery Interval"])
        if interval not in (1, 2, 3, 4):
            raise ValueError(f"Invalid RTM quarter-hour: {row}")
        key = (row["Delivery Date"], int(row["Delivery Hour"]), row["Repeated Hour Flag"], point)
        unique = key + (interval,)
        if unique in seen:
            raise ValueError(f"Duplicate RTM interval: {unique}")
        seen.add(unique)
        if key not in dam:
            unmatched += 1
            continue
        diff = price_cents(row["Settlement Point Price"]) - dam[key]
        record_diff(totals, year, point, diff)
    return {"year": year, "dam_hours": len(dam), "rtm_quarters": len(seen), "unmatched_rtm": unmatched,
            "dam_without_four_quarters": sum(sum(k + (i,) in seen for i in (1, 2, 3, 4)) != 4 for k in dam),
            "first_date": min(dates, key=lambda value: dt.datetime.strptime(value, "%m/%d/%Y")),
            "last_date": max(dates, key=lambda value: dt.datetime.strptime(value, "%m/%d/%Y"))}


def record_diff(totals, year, point, diff):
    for group in ((year, point), (year, "ALL_ZONES"), ("ALL_YEARS", point), ("ALL_YEARS", "ALL_ZONES")):
        item = totals[group]
        item[0] += 1
        item[1] += diff
        item[2] += abs(diff)
        item[3] += diff * diff
        item[4] += diff > 0


class FirstTable(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.inside = False
        self.finished = False
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table" and not self.finished:
            self.inside = True
        elif self.inside and tag == "tr":
            self.row = []
        elif self.row is not None and tag in ("th", "td"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.inside:
            self.inside = False
            self.finished = True


def daily_table(day, kind, manifest):
    date_code = day.strftime("%Y%m%d")
    url = f"{BASE}/content/cdr/html/{date_code}_{kind}_spp.html"
    path = RAW / f"{kind}_{date_code}.html"
    if not path.exists():
        path.write_bytes(get(url))
    data = path.read_bytes()
    manifest[f"{kind}_{date_code}"] = {"source": url, "sha256": hashlib.sha256(data).hexdigest()}
    parser = FirstTable()
    parser.feed(data.decode("utf-8-sig"))
    if not parser.rows:
        raise ValueError(f"No price table in {url}")
    header, *rows = parser.rows
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"Malformed price table in {url}")
    return [dict(zip(header, row)) for row in rows]


def missing_daily_rtm(day, end, manifest):
    date_code = day.strftime("%Y%m%d")
    cached = list(RAW.glob(f"rtm_{date_code}_{end}_*.zip"))
    if len(cached) > 1:
        raise ValueError(f"Ambiguous cached interval files for {day} {end}")
    if cached:
        path = cached[0]
        doc_id = path.stem.rsplit("_", 1)[-1]
    else:
        listing = json.loads(get(f"{BASE}/misapp/servlets/IceDocListJsonWS?reportTypeId=12301"))["ListDocsByRptTypeRes"]["DocumentList"]
        name = f"SPPHLZNP6905_{date_code}_{end}_csv"
        matches = [item["Document"] for item in listing if item["Document"]["FriendlyName"] == name]
        if len(matches) != 1:
            raise ValueError(f"No unique ERCOT interval CSV for {day} {end}")
        doc_id = matches[0]["DocID"]
        path = RAW / f"rtm_{date_code}_{end}_{doc_id}.zip"
        path.write_bytes(get(f"{BASE}/misdownload/servlets/mirDownload?doclookupId={doc_id}"))
    url = f"{BASE}/misdownload/servlets/mirDownload?doclookupId={doc_id}"
    data = path.read_bytes()
    manifest[f"rtm_{date_code}_{end}"] = {"doc_id": doc_id, "source": url,
                                        "sha256": hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        csv_name = next(name for name in outer.namelist() if name.endswith(".csv"))
        rows = csv.DictReader(io.StringIO(outer.read(csv_name).decode("utf-8-sig")))
        result = {"Oper Day": day.strftime("%m/%d/%Y"), "Interval Ending": end}
        for row in rows:
            if row["SettlementPointType"] == "LZ" and row["SettlementPointName"] in ZONES:
                if row["DeliveryDate"] != result["Oper Day"]:
                    raise ValueError(f"Wrong delivery date in {url}")
                result[row["SettlementPointName"]] = row["SettlementPointPrice"]
    if not all(zone in result for zone in ZONES):
        raise ValueError(f"Missing load-zone prices in {url}")
    return result


def summarize_recent(day, totals, manifest):
    dam_rows = daily_table(day, "dam", manifest)
    rtm_rows = daily_table(day, "real_time", manifest)
    if len(dam_rows) != 24:
        # ponytail: Daily HTML lacks an explicit DST flag; annual archives handle transition days.
        raise ValueError(f"Unexpected daily intervals on {day}: {len(dam_rows)} DAM, {len(rtm_rows)} RTM")
    expected = {f"{minute//60:02d}{minute%60:02d}" for minute in range(15, 1441, 15)}
    actual = {row["Interval Ending"] for row in rtm_rows}
    if actual - expected:
        raise ValueError(f"Unexpected RTM interval labels on {day}: {actual - expected}")
    for end in sorted(expected - actual):
        rtm_rows.append(missing_daily_rtm(day, end, manifest))
    if len(rtm_rows) != 96:
        raise ValueError(f"Unexpected RTM interval count on {day}: {len(rtm_rows)}")
    date_text = day.strftime("%m/%d/%Y")
    dam = {}
    for row in dam_rows:
        if row["Oper Day"] != date_text:
            raise ValueError(f"Wrong DAM delivery date on {day}")
        hour = int(row["Hour Ending"])
        for zone in ZONES:
            dam[hour, zone] = price_cents(row[zone])
    seen = set()
    for row in rtm_rows:
        if row["Oper Day"] != date_text:
            raise ValueError(f"Wrong RTM delivery date on {day}")
        end = row["Interval Ending"]
        minute = int(end[:2]) * 60 + int(end[2:])
        hour = (minute - 1) // 60 + 1
        quarter = ((minute - 1) % 60) // 15 + 1
        if (hour, quarter) in seen:
            raise ValueError(f"Duplicate RTM interval on {day}: {end}")
        seen.add((hour, quarter))
        for zone in ZONES:
            record_diff(totals, day.year, zone, price_cents(row[zone]) - dam[hour, zone])
    if len(seen) != 96:
        raise ValueError(f"Missing RTM intervals on {day}")
    return {"source": "daily_html", "first_date": date_text, "last_date": date_text,
            "dam_hours": 24 * len(ZONES), "rtm_quarters": 96 * len(ZONES),
            "unmatched_rtm": 0, "dam_without_four_quarters": 0}


def write_results(totals, coverage, manifest):
    output = ROOT / "data" / "ercot"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "source_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    with (output / "coverage.json").open("w") as file:
        json.dump(coverage, file, indent=2)
        file.write("\n")
    with (output / "dam_rt_deviation.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("year", "load_zone", "matched_15min_intervals", "mean_signed_rtm_minus_dam_usd_per_mwh",
                         "mean_absolute_deviation_usd_per_mwh", "root_mean_square_deviation_usd_per_mwh", "rtm_above_dam_pct"))
        for (year, zone), (n, signed, absolute, squared, above) in sorted(totals.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            if n:
                writer.writerow((year, zone, n, f"{signed/n/100:.4f}", f"{absolute/n/100:.4f}",
                                 f"{math.sqrt(squared/n)/100:.4f}", f"{above/n*100:.2f}"))


def self_test():
    assert price_cents("-1.05") == -105
    assert price_cents("2.7") == 270
    assert price_cents("34.380000000000003") == 3438
    dam_key = ("11/02/2025", 2, "Y", "LZ_NORTH")
    assert dam_key != ("11/02/2025", 2, "N", "LZ_NORTH")
    assert (price_cents("35.25") - price_cents("30.00")) == 525
    assert (60 - 1) // 60 + 1 == 1  # Interval ending 0100 belongs to hour ending 1.


def main():
    self_test()
    listings = {kind: documents(report_id) for kind, report_id in REPORTS.items()}
    years = sorted(set.intersection(*(set(items) for items in listings.values())))
    if not years:
        raise ValueError("No matching annual DAM/RTM archives")
    totals = defaultdict(lambda: [0, 0, 0, 0, 0])
    coverage, manifest = [], {}
    for year in years:
        paths = {}
        for kind in REPORTS:
            doc = listings[kind][year]
            paths[kind] = download(kind, year, doc)
            manifest[f"{kind}_{year}"] = {"doc_id": doc["DocID"], "published": doc["PublishDate"],
                                          "source": f"{BASE}/misdownload/servlets/mirDownload?doclookupId={doc['DocID']}",
                                          "sha256": hashlib.sha256(paths[kind].read_bytes()).hexdigest()}
        result = summarize_year(year, paths["dam"], paths["rtm"], totals)
        coverage.append(result)
        print(result, flush=True)
    latest = dt.datetime.strptime(coverage[-1]["last_date"], "%m/%d/%Y").date()
    yesterday = dt.datetime.now(ZoneInfo("America/Chicago")).date() - dt.timedelta(days=1)
    day = latest + dt.timedelta(days=1)
    while day <= yesterday:
        result = summarize_recent(day, totals, manifest)
        coverage.append(result)
        print(result, flush=True)
        day += dt.timedelta(days=1)
    write_results(totals, coverage, manifest)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("self-test passed")
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("Usage: python3 analyze_dam_rt.py [--self-test]")
