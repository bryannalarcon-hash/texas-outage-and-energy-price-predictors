"""Public-clone replay and causal/DST contract checks; no private data needed."""
from datetime import datetime, timedelta
import unittest
from zoneinfo import ZoneInfo

import numpy as np

import forecast_models as model


class FrozenInferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = model.asset('replay.json')

    def test_existing_predictions_replay(self):
        f = self.fixture
        price = model.price_forecast(f['hourly_dam_rows'], f['price_issued_at_utc'])
        expected = {(r['settlement_point'], r['interval_start_utc']): r for r in f['expected_price']}
        self.assertEqual(price['horizon'], 'full_day')
        self.assertEqual(len(price['records']), 768)
        for row in price['records']:
            old = expected[(row['settlement_point'], row['interval_start_utc'])]
            for key in ['exp002_mean_usd_mwh', 'rtm_mean_usd_mwh', 'rtm_p10_usd_mwh', 'rtm_p50_usd_mwh', 'rtm_p90_usd_mwh']:
                self.assertAlmostEqual(row[key], old[key], places=10)
        outage = model.outage_forecast(f['county_rows'], f['outage_issued_at_utc'], forecast_origin_utc=f['outage_issued_at_utc'])
        for row in outage['records']:
            np.testing.assert_allclose(row['p_first_start_by_hour'], f['expected_outage'][row['county_fips']], rtol=0, atol=1e-14)
            self.assertAlmostEqual(sum(row['p_first_start_by_hour']), row['p_any_next_24h'], places=14)
            self.assertEqual(row['coverage'], 'scenario')
            self.assertTrue(row['at_risk_assumed'])

    def test_previous_outage_is_original_forecast_available_before_day_ahead_plan(self):
        f = self.fixture
        origin = f['previous_outage_issued_at_utc']
        self.assertLess(model.utc(origin), model.utc(f['price_issued_at_utc']))
        self.assertLess(model.utc(f['price_issued_at_utc']), model.utc(f['hourly_dam_rows'][0]['interval_start_utc']))
        self.assertLess(model.utc(f['hourly_dam_rows'][0]['interval_start_utc']), model.utc(f['outage_issued_at_utc']))
        run = model.outage_forecast(f['previous_county_rows'], origin, forecast_origin_utc=origin)
        self.assertEqual(len(run['records']), 4)
        for row in run['records']:
            np.testing.assert_allclose(row['p_first_start_by_hour'], f['previous_expected_outage'][row['county_fips']], rtol=0, atol=1e-14)

    def test_dst_is_elapsed_utc_and_repeated_hour_is_exact(self):
        for day, count in [('2026-03-08', 92), ('2026-11-01', 100)]:
            start = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('America/Chicago'))
            end = (start + timedelta(days=1)).astimezone(model.UTC)
            hour = start.astimezone(model.UTC)
            rows = []
            while hour < end:
                rows.append(dict(settlement_point='LZ_HOUSTON', interval_start_utc=model.stamp(hour), dam_spp_usd_mwh=hour.hour - 2.))
                hour += timedelta(hours=1)
            result = model.price_forecast(rows, model.stamp(start.astimezone(model.UTC) - timedelta(hours=8)))
            self.assertEqual(len(result['records']), count)
            self.assertEqual(sum(r['repeated_hour_flag'] == 'Y' for r in result['records']), 4 if count == 100 else 0)
            self.assertEqual(result['records'][0]['interval_start_utc'], model.stamp(start))
            self.assertEqual(result['records'][-1]['interval_end_utc'], model.stamp(end))
            # Cropping across either clock change uses UTC and keeps the exact
            # predictions produced with the original full-day DAM context.
            issue = start.astimezone(model.UTC) + timedelta(hours=1, minutes=50)
            cropped = model.price_forecast(rows, model.stamp(issue))
            future = [r for r in result['records'] if model.utc(r['interval_start_utc']) >= issue]
            self.assertEqual(cropped['horizon'], 'remaining_day')
            self.assertEqual(cropped['records'], future)
            self.assertEqual(cropped['records'][0]['interval_start_utc'], model.stamp(start.astimezone(model.UTC) + timedelta(hours=2)))
            self.assertEqual(sum(r['repeated_hour_flag'] == 'Y' for r in cropped['records']), 4 if count == 100 else 0)

    def test_same_day_crop_preserves_actual_issue_and_complete_dam_features(self):
        f = self.fixture
        original = model.price_forecast(f['hourly_dam_rows'], f['price_issued_at_utc'])
        for issue, expected_start in [('2025-07-01T13:07:01Z', '2025-07-01T13:15:00Z'),
                                      ('2025-07-01T13:15:00Z', '2025-07-01T13:15:00Z')]:
            result = model.price_forecast(f['hourly_dam_rows'], issue, input_cutoff_utc=f['price_issued_at_utc'],
                                          blend_state=original['provenance']['blend'])
            self.assertEqual(result['horizon'], 'remaining_day')
            self.assertEqual(result['issued_at_utc'], issue)
            self.assertEqual(result['input_cutoff_utc'], f['price_issued_at_utc'])
            self.assertEqual(result['records'][0]['interval_start_utc'], expected_start)
            self.assertEqual(result['records'], [r for r in original['records'] if r['interval_start_utc'] >= expected_start])
            self.assertEqual(result['records'][-1]['interval_end_utc'], '2025-07-02T05:00:00Z')
        # Removing an already elapsed hour would change daily extrema/neighbors.
        with self.assertRaises(ValueError):
            model.price_forecast(f['hourly_dam_rows'][1:], '2025-07-01T13:07:01Z')

    def test_last_aligned_quarter_is_allowed_and_later_issues_are_rejected(self):
        result = model.price_forecast(self.fixture['hourly_dam_rows'], '2025-07-02T04:45:00Z')
        self.assertEqual(len(result['records']), 8)
        self.assertTrue(all(r['interval_start_utc'] == '2025-07-02T04:45:00Z' for r in result['records']))
        self.assertTrue(all(r['interval_end_utc'] == '2025-07-02T05:00:00Z' for r in result['records']))
        for issue in ['2025-07-02T04:45:01Z', '2025-07-02T05:00:00Z', '2025-07-03T12:00:00Z']:
            with self.assertRaises(ValueError):
                model.price_forecast(self.fixture['hourly_dam_rows'], issue)

    def test_incomplete_and_duplicate_dam_are_rejected(self):
        rows = self.fixture['hourly_dam_rows']
        for bad in ([], rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                model.price_forecast(bad, self.fixture['price_issued_at_utc'])

    def test_blend_does_not_admit_newer_or_duplicate_outcomes(self):
        issue = '2025-07-01T22:00:00Z'
        seed = model.blend_weight('2025-06-30T22:00:00Z')
        row = dict(settlement_point='LZ_HOUSTON', model_issue_utc='2025-06-28T22:00:00Z',
                   interval_end_utc='2025-06-29T22:00:00Z', dam_spp_usd_mwh=10., exp002_mean=12., rtm_spp_usd_mwh=13.)
        newer = dict(row, interval_end_utc='2025-06-30T22:00:00Z')
        late_publication = dict(row, interval_end_utc='2025-06-29T21:00:00Z', actual_available_at_utc='2025-07-02T00:00:00Z')
        state = model.blend_weight(issue, blend_state=seed, history_rows=[row, newer, late_publication])
        self.assertEqual(state['new_outcome_rows'], 1)
        self.assertEqual(state['rows'], seed['rows'] + 1)
        self.assertAlmostEqual(state['numerator'], seed['numerator'] + 6)
        self.assertAlmostEqual(state['denominator'], seed['denominator'] + 4)
        self.assertEqual(state['adaptation_status'], 'updated_from_delayed_actuals')
        with self.assertRaises(ValueError):
            model.blend_weight(issue, blend_state=seed, history_rows=[row, row])
        with self.assertRaises(ValueError):
            model.blend_weight('2025-06-30T21:00:00Z', blend_state=seed)

    def test_outage_origin_and_unknown_state_remain_explicit(self):
        f = self.fixture
        row = f['county_rows'][0]
        result = model.outage_forecast([row], '2025-07-01T18:10:00Z', forecast_origin_utc=f['outage_issued_at_utc'])
        self.assertEqual(result['issued_at_utc'], '2025-07-01T18:10:00Z')
        self.assertEqual(result['intervals'][0]['interval_start_utc'], '2025-07-01T12:00:00Z')
        for unknown in [dict(row, at_risk_assumed=False), dict(row, active_outage={'detected': True})]:
            result = model.outage_forecast([unknown], f['outage_issued_at_utc'], forecast_origin_utc=f['outage_issued_at_utc'])
            self.assertEqual(result['records'][0]['coverage'], 'unknown')
            self.assertIsNone(result['records'][0]['p_any_next_24h'])
        with self.assertRaises(ValueError):
            model.outage_forecast([row], '2025-07-01T18:10:00Z', forecast_origin_utc='2025-07-01T18:00:00Z')
        with self.assertRaises(ValueError):
            model.outage_forecast([dict(row, wpc_available=0, wpc_risk=0)], f['outage_issued_at_utc'], forecast_origin_utc=f['outage_issued_at_utc'])


if __name__ == '__main__':
    unittest.main()
