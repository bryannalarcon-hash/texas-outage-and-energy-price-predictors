import { readFile } from 'node:fs/promises';
import { validateGeometry } from './core.mjs';

const read = async name => JSON.parse(await readFile(new URL(`data/${name}`, import.meta.url), 'utf8'));
export const geometry = validateGeometry({
  energy: await read('texas-load-zones.json'),
  outages: await read('texas-counties.json')
});
export const dummy = await read('dummy-data.json');
export const now = Date.parse(dummy.generated_at_utc);

// An in-memory MODEL fixture, never a production model-output.json export.
export function modelPayload() {
  const payload = structuredClone(dummy);
  payload.kind = 'model';
  payload.decisions.is_demo = true;
  for (const row of payload.outage.records) {
    if (row.coverage === 'unknown') Object.assign(row, {
      coverage: 'observed', active_outage: null,
      p_first_start_by_hour: Array(24).fill(0), p_any_next_24h: 0
    });
  }
  const decisions = payload.decisions.records;
  for (const decision of decisions) {
    if (decision.relationship?.method === 'illustrative_pair') decision.relationship = null;
  }
  const keys = new Set(decisions.map(row => `${row.mode}|${row.region_id}|${row.interval_start_utc}`));
  const addHold = (mode, region_id, interval_start_utc) => {
    if (!keys.has(`${mode}|${region_id}|${interval_start_utc}`)) decisions.push({
      mode, region_id, interval_start_utc, action: 'hold', strength: 0,
      reserve_constraint: true, reason_codes: ['no_clear_opportunity'], relationship: null
    });
  };
  for (const row of payload.price.records) {
    if (row.availability === 'available') addHold('energy', row.settlement_point, row.interval_start_utc);
  }
  for (const row of payload.outage.records) {
    for (const interval of payload.outage.intervals) addHold('outages', row.county_fips, interval.interval_start_utc);
  }
  return payload;
}

const iso = value => new Date(value).toISOString().slice(0, 19) + 'Z';
const central = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
});

// Returns a complete price run; callers supply a known Central midnight and day length.
export function priceDay(date, startUtc, count) {
  const start = Date.parse(startUtc);
  const seen = new Set();
  const records = [];
  for (let index = 0; index < count; index++) {
    const time = start + index * 900000;
    const { hour, minute } = Object.fromEntries(central.formatToParts(time).map(part => [part.type, part.value]));
    const label = `${hour}:${minute}`;
    const repeated_hour_flag = seen.has(label) ? 'Y' : 'N';
    seen.add(label);
    for (const region of geometry.energy.regions) records.push({
      settlement_point: region.id, interval_start_utc: iso(time), interval_end_utc: iso(time + 900000),
      delivery_date: date, hour_ending: Number(hour) + 1, quarter: Number(minute) / 15 + 1,
      repeated_hour_flag, availability: 'available', dam_spp_usd_mwh: 20,
      rtm_mean_usd_mwh: 30, rtm_p10_usd_mwh: 10, rtm_p50_usd_mwh: 25, rtm_p90_usd_mwh: 40
    });
  }
  return {
    status: 'available', model_version: 'test-price-v1', issued_at_utc: iso(start - 43200000),
    input_cutoff_utc: iso(start - 43200000), max_age_hours: 48, target_date: date, records
  };
}
