"""Run with: .venv/bin/python -m unittest test_forecast_sources -v"""

import csv
import datetime as dt
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import forecast_sources as source
from build_outlook_features import self_check as check_outlook_cutoffs


def dam_zip(day, *, missing=False, duplicate=False):
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["DeliveryDate", "HourEnding", "SettlementPoint", "SettlementPointPrice", "DSTFlag"])
    rows = [[day.strftime("%m/%d/%Y"), f"{hour:02}:00", zone, 25.5, flag]
            for zone in source.ZONES for hour, flag in source.hour_slots(day)]
    writer.writerows(rows[:-1] if missing else rows)
    if duplicate:
        writer.writerow(rows[0])
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("prices.csv", table.getvalue())
    return body.getvalue()


def receipt(body, url, now):
    return {"url": url, "sha256": hashlib.sha256(body).hexdigest(),
            "first_observed_at_utc": source.iso(now), "bytes": len(body)}


class ForecastSourcesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = patch.object(source, "CACHE", Path(self.temp.name))
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.now = source.instant("2026-09-27T19:00:00Z")
        self.day = dt.date(2026, 9, 28)
        self.counties = [{"county_fips": "48113", "latitude": 32.7, "longitude": -96.7}]

    def test_complete_normal_spring_and_fall_curves(self):
        for day, hours in ((self.day, 24), (dt.date(2026, 3, 8), 23), (dt.date(2026, 11, 1), 25)):
            rows = source.parse_dam(dam_zip(day))
            self.assertEqual(len(rows), hours * 8)
            starts = {row["valid_start_utc"] for row in rows}
            self.assertEqual(len(starts), hours)
            self.assertEqual(sum(r["repeated_hour"] == "Y" for r in rows), 8 if hours == 25 else 0)

    def test_reject_incomplete_and_duplicate_dam(self):
        for kwargs in ({"missing": True}, {"duplicate": True}):
            with self.assertRaises(ValueError):
                source.parse_dam(dam_zip(self.day, **kwargs))

    def mock_listing(self, published):
        return {"ListDocsByRptTypeRes": {"DocumentList": [{"Document": {
            "PublishDate": published, "FriendlyName": "DAMSPNP4190_csv", "DocID": "123"}}]}}

    def test_publication_is_observed_not_assumed_from_nominal_time(self):
        body = json.dumps(self.mock_listing("2026-09-27T15:00:00-05:00")).encode()
        with patch.object(source, "_get", return_value=(body, receipt(body, source.DAM_LIST, self.now))):
            self.assertEqual(source._dam(self.day, self.now)["status"], "unavailable")

    def test_reject_wrong_delivery_date_and_keep_receipt_on_cache_hit(self):
        listing = json.dumps(self.mock_listing("2026-09-27T12:40:00-05:00")).encode()
        payload = dam_zip(self.day)

        def get(url, now):
            body = listing if url == source.DAM_LIST else payload
            return body, receipt(body, url, now)

        with patch.object(source, "_get", side_effect=get) as download:
            first = source._dam(self.day, self.now)
            second = source._dam(self.day, self.now + dt.timedelta(minutes=15))
            self.assertEqual(first["first_observed_at_utc"], second["first_observed_at_utc"])
            self.assertEqual(download.call_count, 3)  # two listings, one immutable CSV
        with patch.object(source, "_get", side_effect=get):
            self.assertEqual(source._dam(self.day - dt.timedelta(days=1), self.now)["status"], "unavailable")

    def weather(self, kind, origin, counties, now):
        return {"source_id": "spc" if kind == "C" else "wpc", "status": "ready",
                "product_id": kind + "/" + source.iso(origin),
                "first_observed_at_utc": "2026-09-27T18:55:00Z"}, [2] * len(counties)

    def test_ready_time_is_latest_actual_input_receipt(self):
        dam = {"status": "ready", "document_id": "123", "source_id": "dam", "rows": [],
               "first_observed_at_utc": "2026-09-27T18:59:00Z"}
        with patch.object(source, "_dam", return_value=dam), patch.object(source, "_weather_source", side_effect=self.weather):
            result = source.pull_inputs(self.now, self.day, self.counties)
        self.assertEqual(result["inputs_available_at_utc"], "2026-09-27T18:59:00Z")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["weather"]["forecast_origin_utc"], "2026-09-27T12:00:00Z")
        self.assertEqual(result["weather"]["counties"][0]["spc_risk"], 2)

    def test_weather_can_be_ready_when_tomorrow_dam_is_missing(self):
        with patch.object(source, "_dam", return_value={"source_id": "dam", "status": "unavailable", "rows": []}), \
                patch.object(source, "_weather_source", side_effect=self.weather):
            result = source.pull_inputs(self.now, counties=self.counties)
        self.assertEqual(result["target_date"], "2026-09-28")
        self.assertEqual(result["status"], "waiting_for_dam")
        self.assertEqual(result["weather"]["status"], "ready")
        self.assertIsNone(result["input_id"])

    def test_network_failure_does_not_become_zero_risk(self):
        with patch.object(source, "_dam", side_effect=TimeoutError("test timeout")), \
                patch.object(source, "_weather_source", side_effect=TimeoutError("test timeout")):
            result = source.pull_inputs(self.now, counties=self.counties)
        self.assertEqual(result["status"], "source_error")
        county = result["weather"]["counties"][0]
        self.assertEqual((county["spc_risk"], county["spc_available"]), (-1, 0))
        self.assertIsNone(result["inputs_available_at_utc"])

    def test_actual_weather_selector_rejects_late_and_short_horizon_products(self):
        check_outlook_cutoffs()

    def test_utc_morning_origin_stays_on_previous_day(self):
        with patch.object(source, "_dam", return_value={"status": "unavailable", "rows": []}), \
                patch.object(source, "_weather_source", side_effect=self.weather):
            result = source.probe_sources("2026-09-27T10:00:00Z", counties=self.counties)
        self.assertEqual(result["weather"]["forecast_origin_utc"], "2026-09-26T12:00:00Z")
        self.assertNotIn("counties", result["weather"])

    def test_bad_cache_and_redirect_are_rejected(self):
        source._save("test", b"correct", receipt(b"correct", source.DAM_LIST, self.now))
        (source.CACHE / "test.body").write_bytes(b"wrong")
        with self.assertRaises(ValueError):
            source._cached("test")
        for url in ("http://www.ercot.com/", "https://127.0.0.1/", "https://www.ercot.com@evil.test/"):
            with self.assertRaises(ValueError):
                source._url_allowed(url)

    def test_rtm_has_48_hour_delay_and_explicit_dst_skip(self):
        day = dt.date(2026, 9, 25)
        rows = [["Oper Day", "Interval Ending", *source.ZONES]]
        rows.extend(["09/25/2026", f"{m // 60:02}{m % 60:02}", *(["42.5"] * 8)] for m in range(15, 1441, 15))
        body = ("<table>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</table>").encode()
        with patch.object(source, "_get", return_value=(body, receipt(body, "https://www.ercot.com/test", self.now))):
            result = source.fetch_rtm_labels(day, self.now)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["rows"]), 14 * 4 * 8)  # 19Z = 14 local hours, two days later
        self.assertTrue(all(source.instant(r["valid_end_utc"]) + dt.timedelta(hours=48) <= self.now for r in result["rows"]))
        self.assertEqual(source.fetch_rtm_labels("2026-11-01", self.now)["status"], "unsupported_dst")


if __name__ == "__main__":
    unittest.main()
