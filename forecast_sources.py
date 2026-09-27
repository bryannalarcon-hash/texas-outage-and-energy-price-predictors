#!/usr/bin/env python3
"""Bounded, public inputs for the frozen forecast models; no credentials or fitting.

ERCOT: NP4-190-CD / report 12331, as listed by IceDocListJsonWS. Results are
normally due by 13:30 Central, but readiness below requires the actual CSV.
https://www.ercot.com/files/docs/2025/08/22/2026_01-Day-Ahead-Market-Operations.pdf
Weather: NOAA Day 1 products distributed by the same IEM archive used in training.
https://mesonet.agron.iastate.edu/cgi-bin/request/gis/outlooks.py?help

First receipt is persisted in forecast_cache/sources. Publisher issue time is
separate from observed availability; callers schedule ten minutes after the
maximum first receipt. An unavailable source never fabricates a forecast.
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

from shapely.geometry import Point

from analyze_dam_rt import FirstTable, ZONES
from build_outlook_features import risk_by_county, stamp


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "forecast_cache/sources"
UTC = dt.timezone.utc
CENTRAL = ZoneInfo("America/Chicago")
LIMIT = 8_000_000
EXPANDED_LIMIT = 40_000_000
DAM_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=12331"
HOSTS = {"www.ercot.com", "mesonet.agron.iastate.edu"}


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def instant(value=None):
    result = dt.datetime.now(UTC) if value is None else value
    if isinstance(result, str):
        result = dt.datetime.fromisoformat(result.replace("Z", "+00:00"))
    if not isinstance(result, dt.datetime) or result.tzinfo is None:
        raise ValueError("An offset-aware timestamp is required")
    return result.astimezone(UTC)


def target_day(value, now):
    if value is None:
        return now.astimezone(CENTRAL).date() + dt.timedelta(days=1)
    if isinstance(value, str):
        value = dt.date.fromisoformat(value)
    if isinstance(value, dt.datetime) or not isinstance(value, dt.date):
        raise ValueError("Target date must be YYYY-MM-DD")
    return value


def _url_allowed(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc not in HOSTS
            or parsed.username or parsed.password):
        raise ValueError("Only the public ERCOT and IEM HTTPS endpoints are allowed")


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        _url_allowed(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def _get(url, now):
    _url_allowed(url)
    request = urllib.request.Request(url, headers={"User-Agent": "BPC forecast demo/1.0"})
    with urllib.request.build_opener(_PublicRedirect()).open(request, timeout=25) as response:
        _url_allowed(response.url)
        body = response.read(LIMIT + 1)
        if len(body) > LIMIT:
            raise ValueError("Source exceeds the 8 MB download limit")
        size = response.headers.get("Content-Length")
        if size is not None and int(size) != len(body):
            raise ValueError("Incomplete source response")
        receipt = {"url": url, "first_observed_at_utc": iso(dt.datetime.now(UTC)),
                   "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                   "http_last_modified": response.headers.get("Last-Modified")}
    return body, receipt


def _save(key, body, receipt):
    CACHE.mkdir(parents=True, exist_ok=True)
    for suffix, content in ((".body", body), (".json", json.dumps(receipt).encode())):
        with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as output:
            output.write(content)
            temporary = Path(output.name)
        temporary.replace(CACHE / (key + suffix))


def _cached(key):
    metadata = CACHE / (key + ".json")
    if not metadata.exists():
        return None
    receipt = json.loads(metadata.read_text())
    body = (CACHE / (key + ".body")).read_bytes()
    if hashlib.sha256(body).hexdigest() != receipt["sha256"]:
        raise ValueError("Cached source failed its SHA-256 check")
    return body, receipt


def _zip(body):
    archive = zipfile.ZipFile(io.BytesIO(body))
    if sum(item.file_size for item in archive.infolist()) > EXPANDED_LIMIT:
        archive.close()
        raise ValueError("Expanded report exceeds the 40 MB limit")
    return archive


def hour_slots(day):
    """ERCOT clock-hour ending + repeated flag mapped to real UTC intervals."""
    current = dt.datetime.combine(day, dt.time(), CENTRAL).astimezone(UTC)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), CENTRAL).astimezone(UTC)
    result = {}
    while current < end:
        local = current.astimezone(CENTRAL)
        result[(local.hour + 1, "Y" if local.fold else "N")] = current
        current += dt.timedelta(hours=1)
    return result


def parse_dam(body):
    """Validate every load-zone curve, including 23/25-hour delivery days."""
    with _zip(body) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one DAM CSV")
        reader = csv.DictReader(io.StringIO(archive.read(members[0]).decode("utf-8-sig")))
        expected = {"DeliveryDate", "HourEnding", "SettlementPoint", "SettlementPointPrice", "DSTFlag"}
        if not expected <= set(reader.fieldnames or ()):
            raise ValueError("Unexpected DAM CSV schema")
        rows, seen, days = [], set(), set()
        for raw in reader:
            zone = raw["SettlementPoint"].strip()
            if zone not in ZONES:
                continue
            day = dt.datetime.strptime(raw["DeliveryDate"], "%m/%d/%Y").date()
            hour = int(raw["HourEnding"].split(":")[0])
            repeated = raw["DSTFlag"].strip().upper()
            slots = hour_slots(day)
            if (hour, repeated) not in slots:
                raise ValueError("Invalid or nonexistent DAM clock-hour/repeated flag")
            key = (day, zone, hour, repeated)
            if key in seen:
                raise ValueError("Duplicate DAM interval")
            seen.add(key)
            days.add(day)
            price = float(raw["SettlementPointPrice"])
            if not math.isfinite(price):
                raise ValueError("Non-finite DAM price")
            start = slots[(hour, repeated)]
            rows.append({"settlement_point": zone, "delivery_date": day.isoformat(),
                         "hour_ending": hour, "repeated_hour": repeated,
                         "valid_start_utc": iso(start), "valid_end_utc": iso(start + dt.timedelta(hours=1)),
                         "dam_price": price})
    if len(days) != 1:
        raise ValueError("DAM report must contain one delivery day")
    day = next(iter(days))
    expected_keys = {(day, zone, hour, flag) for zone in ZONES for hour, flag in hour_slots(day)}
    if seen != expected_keys:
        raise ValueError("DAM report lacks complete curves for all eight load zones")
    return sorted(rows, key=lambda row: (row["valid_start_utc"], row["settlement_point"]))


def _dam(day, now):
    listing, _ = _get(DAM_LIST, now)
    documents = json.loads(listing)["ListDocsByRptTypeRes"]["DocumentList"]
    candidates = []
    for item in documents:
        document = item["Document"]
        published = instant(document["PublishDate"])
        if ("_csv" in document.get("FriendlyName", "").lower() and published <= now
                and day - dt.timedelta(days=1) <= published.astimezone(CENTRAL).date() <= day):
            candidates.append((published, document))
    # A normal day has one CSV; allow a small number of corrected/delayed runs.
    for published, document in sorted(candidates, key=lambda item: item[0], reverse=True)[:4]:
        doc_id = str(document["DocID"])
        if not doc_id.isdecimal():
            raise ValueError("Invalid ERCOT document ID")
        key = "dam_" + doc_id
        cached = _cached(key)
        url = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=" + doc_id
        body, receipt = cached or _get(url, now)
        rows = parse_dam(body)
        if cached is None:
            _save(key, body, receipt)
        if rows[0]["delivery_date"] == day.isoformat():
            return {"source_id": "dam", "status": "ready", "report_id": 12331,
                    "document_id": doc_id, "published_at_utc": iso(published),
                    **receipt, "rows": rows}
    return {"source_id": "dam", "status": "unavailable", "report_id": 12331,
            "url": DAM_LIST, "rows": [],
            "message": f"ERCOT has not supplied a complete DAM curve for {day.isoformat()}"}


def load_counties():
    data = json.loads((ROOT / "model_assets/counties.json").read_text())
    return [{"county_fips": fips, **county} for fips, county in data.items() if county["eligible"]]


def _county_rows(counties):
    result, seen = [], set()
    for row in counties:
        fips = str(row["county_fips"])
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if (len(fips) != 5 or not fips.isdecimal() or fips in seen
                or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            raise ValueError("Invalid county coordinates or duplicate FIPS")
        seen.add(fips)
        result.append((fips, row.get("name", fips), lon, lat, Point(lon, lat)))
    if not result:
        raise ValueError("At least one eligible county is required")
    return result


def _weather_source(kind, origin, county_rows, now):
    prefix = "spc" if kind == "C" else "wpc"
    day = origin.date()
    key = f"{prefix}_{day.isoformat()}"
    cached = _cached(key)
    end = day + dt.timedelta(days=1)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/gis/outlooks.py"
           f"?d=1&type={kind}&sts={day.isoformat()}T00:00Z&ets={end.isoformat()}T00:00Z")
    body, receipt = cached or _get(url, now)
    with _zip(body):
        pass
    # Reuse the exact training-time polygon and publication-cutoff selector.
    with tempfile.NamedTemporaryFile(suffix=".zip") as source:
        source.write(body)
        source.flush()
        chosen, risks = risk_by_county(source.name, kind, day.year, county_rows)
    product = chosen[day.isoformat()]
    metadata = {"source_id": prefix, **receipt, "status": "ready" if product else "unavailable"}
    if product:
        metadata.update(published_at_utc=stamp(product[0]), valid_start_utc=stamp(product[1]),
                        valid_end_utc=stamp(product[2]), product_id="/".join(product))
        if cached is None:
            _save(key, body, receipt)
        return metadata, risks[day.isoformat()]
    metadata["message"] = "No prior-issued Day 1 product covers the complete 12Z–12Z horizon"
    return metadata, [-1] * len(county_rows)


def pull_inputs(now=None, target_date=None, counties=None):
    """Return available inputs and explicit per-source failures; never replace UI data.

    Weather origin stays at the latest elapsed 12Z, even when retrieved later.
    It does not cover the whole following Central delivery day. The caller must
    preserve those timestamps and use its reserve fallback outside that horizon.
    """
    now = instant(now)
    day = target_day(target_date, now)
    county_rows = _county_rows(load_counties() if counties is None else counties)
    origin = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if origin > now:
        origin -= dt.timedelta(days=1)
    try:
        dam = _dam(day, now)
    except Exception as error:
        dam = {"source_id": "dam", "status": "error", "report_id": 12331,
               "url": DAM_LIST, "rows": [], "message": f"{type(error).__name__}: {error}"}
    weather = {"forecast_origin_utc": iso(origin), "valid_start_utc": iso(origin),
               "valid_end_utc": iso(origin + dt.timedelta(hours=24)), "sources": [],
               "counties": [{"county_fips": c[0], "latitude": c[3], "longitude": c[2]} for c in county_rows]}
    for kind, prefix in (("C", "spc"), ("E", "wpc")):
        try:
            metadata, risks = _weather_source(kind, origin, county_rows, now)
        except Exception as error:
            metadata = {"source_id": prefix, "status": "error", "message": f"{type(error).__name__}: {error}"}
            risks = [-1] * len(county_rows)
        weather["sources"].append(metadata)
        for county, risk in zip(weather["counties"], risks):
            county[prefix + "_risk"] = risk
            county[prefix + "_available"] = int(metadata["status"] == "ready")
    sources = [{key: value for key, value in dam.items() if key != "rows"}, *weather["sources"]]
    weather["status"] = "ready" if all(s["status"] == "ready" for s in weather["sources"]) else "unavailable"
    ready = all(source["status"] == "ready" for source in sources)
    status = ("ready" if ready else "source_error" if any(s["status"] == "error" for s in sources)
              else "waiting_for_dam" if dam["status"] != "ready" else "waiting_for_weather")
    signature = {"target_date": day.isoformat(), "dam_document": dam.get("document_id"),
                 "weather_origin": iso(origin), "products": [s.get("product_id") for s in weather["sources"]]}
    return {"target_date": day.isoformat(), "checked_at_utc": iso(now), "status": status,
            "input_id": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() if ready else None,
            "inputs_available_at_utc": max(s["first_observed_at_utc"] for s in sources) if ready else None,
            "dam": dam, "weather": weather, "sources": sources}


def probe_sources(now=None, target_date=None, counties=None):
    """Same checked input set with payload arrays omitted; immutable reports are cached."""
    result = pull_inputs(now, target_date, counties)
    result["dam"].pop("rows", None)
    result["weather"].pop("counties", None)
    return result


def fetch_rtm_labels(day, now=None):
    """Public daily RTM table, eight zones; only labels at least 48 hours old.

    The HTML has no DST-fold field, so transition days are explicitly skipped.
    This optional feedback must never block next-day inference.
    """
    now = instant(now)
    day = target_day(day, now)
    key = "rtm_" + day.isoformat()
    url = f"https://www.ercot.com/content/cdr/html/{day:%Y%m%d}_real_time_spp.html"
    try:
        if len(hour_slots(day)) != 24:
            return {"status": "unsupported_dst", "rows": [], "url": url,
                    "message": "Daily HTML has no repeated-hour flag; this day is excluded from adaptive feedback"}
        cached = _cached(key)
        body, receipt = cached or _get(url, now)
        parser = FirstTable()
        parser.feed(body.decode("utf-8-sig"))
        header, *table = parser.rows
        if not {"Oper Day", "Interval Ending", *ZONES} <= set(header):
            raise ValueError("Unexpected RTM daily table schema")
        expected = {f"{minute // 60:02}{minute % 60:02}" for minute in range(15, 1441, 15)}
        seen, rows = set(), []
        start_day = dt.datetime.combine(day, dt.time(), CENTRAL).astimezone(UTC)
        for cells in table:
            if len(cells) != len(header):
                raise ValueError("Malformed RTM daily table")
            raw = dict(zip(header, cells))
            label = raw["Interval Ending"]
            if raw["Oper Day"] != day.strftime("%m/%d/%Y") or label not in expected or label in seen:
                raise ValueError("Invalid or duplicate RTM interval")
            seen.add(label)
            minutes = 60 * int(label[:2]) + int(label[2:])
            end = start_day + dt.timedelta(minutes=minutes)
            for zone in ZONES:
                price = float(raw[zone])
                if not math.isfinite(price):
                    raise ValueError("Non-finite RTM label")
                if end + dt.timedelta(hours=48) <= now:
                    rows.append({"settlement_point": zone, "valid_start_utc": iso(end - dt.timedelta(minutes=15)),
                                 "valid_end_utc": iso(end), "rtm_price": price})
        if seen != expected:
            raise ValueError("RTM daily table is incomplete")
        if cached is None:
            _save(key, body, receipt)
        return {"status": "ready", **receipt, "rows": rows, "label_delay_hours": 48,
                "price_version": "first observed public daily table; subsequent corrections not replayed"}
    except Exception as error:
        return {"status": "error", "rows": [], "url": url, "message": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Show metadata without the input arrays")
    parser.add_argument("--target-date", help="Delivery date; defaults to tomorrow in America/Chicago")
    args = parser.parse_args()
    result = probe_sources(target_date=args.target_date) if args.probe else pull_inputs(target_date=args.target_date)
    print(json.dumps(result, indent=2, allow_nan=False))
