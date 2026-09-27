export const MODES = ['energy', 'outages'];
export const SOURCES = ['dummy', 'model'];
export const STATUSES = ['loading', 'current', 'stale', 'partial', 'empty', 'error'];
export const EVENTS = ['LOAD', 'RESOLVE', 'REJECT', 'MODE', 'TIME', 'PREVIEW', 'UNPREVIEW', 'PIN', 'CLEAR', 'RULES', 'RETURN', 'RESTORE', 'TICK'];
export const ACTIONS = { charge: 'Charge', hold: 'Hold reserve', discharge: 'Discharge' };
export const REASONS = {
  reserve_protection: 'Outage protection takes priority over the price opportunity.',
  low_price: 'The example price is below the demo charging threshold.',
  price_opportunity: 'The example price clears the demo discharge threshold.',
  no_clear_opportunity: 'The example price is between the demo thresholds.',
  no_price_relationship: 'Keep reserve; no verified price relationship is available for this county.',
  active_scenario: 'A county scenario was ongoing at issue time. Protect reserve.',
  device_constraint: 'A supplied battery constraint limits the action.',
  missing_input: 'A required input is unavailable; the exported rule protects reserve.',
  policy_threshold: 'The versioned policy threshold determines this action.'
};
export const RAMPS = {
  energy: { metric: 'Expected RTM · USD/MWh', thresholds: [25, 80, 150, 300], labels: ['< $25', '$25–80', '$80–150', '$150–300', '≥ $300'] },
  outages: { metric: 'County scenario · first onset in this hour', thresholds: [0.005, 0.02, 0.05, 0.1], labels: ['< 0.5%', '0.5–2%', '2–5%', '5–10%', '≥ 10%'] }
};
export const MAX_BYTES = 10 * 1024 * 1024;
export const REQUEST_TIMEOUT = 8000;
const utcPattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
const localParts = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const localLabel = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Chicago', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
const moneyLabel = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });

function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

function record(value, name) {
  ensure(value && typeof value === 'object' && !Array.isArray(value), `${name} must be an object.`);
}

function text(value, name, maximum = 300) {
  ensure(typeof value === 'string' && value.trim().length > 0 && value.length <= maximum, `${name} must be nonempty text (≤ ${maximum} characters).`);
}

function number(value, name, minimum = -Infinity, maximum = Infinity) {
  ensure(typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum, `${name} must be a finite number from ${minimum} to ${maximum}.`);
}

function timestamp(value, name) {
  ensure(typeof value === 'string' && utcPattern.test(value) && Number.isFinite(Date.parse(value)), `${name} must be an explicit UTC timestamp.`);
  ensure(new Date(value).toISOString().replace('.000Z', 'Z') === value.replace('.000Z', 'Z'), `${name} is not a real calendar timestamp.`);
  return Date.parse(value);
}

function parts(value) {
  return Object.fromEntries(localParts.formatToParts(new Date(value)).filter(part => part.type !== 'literal').map(part => [part.type, part.value]));
}

function dayOf(value) {
  const local = parts(value);
  return `${local.year}-${local.month}-${local.day}`;
}

export function formatTime(value) {
  return value ? localLabel.format(new Date(value)) : 'Unavailable';
}

export function formatMoney(value) {
  return moneyLabel.format(value);
}

export function colorClass(mode, value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'unknown';
  const index = RAMPS[mode].thresholds.findIndex(threshold => value < threshold);
  return `level-${index === -1 ? 4 : index}`;
}

export function validateGeometry(geometry) {
  for (const mode of MODES) {
    const layer = geometry[mode];
    record(layer, `${mode} geometry`);
    ensure(Array.isArray(layer.view_box) && layer.view_box.length === 4 && layer.view_box.every(Number.isFinite) && layer.view_box[2] > 0 && layer.view_box[3] > 0, 'Invalid map view box.');
    ensure(Array.isArray(layer.regions) && layer.regions.length === (mode === 'energy' ? 8 : 254), `${mode} geometry has the wrong region count.`);
    const seen = new Set();
    for (const region of layer.regions) {
      text(region.id, 'Geometry ID', 30);
      text(region.name, 'Region name', 100);
      ensure(!seen.has(region.id), 'Duplicate geometry ID.');
      ensure(mode === 'energy' ? /^LZ_(WEST|NORTH|SOUTH|HOUSTON|AEN|CPS|LCRA|RAYBN)$/.test(region.id) : /^48\d{3}$/.test(region.id), 'Invalid geometry ID.');
      ensure(typeof region.path === 'string' && region.path.length > 6 && /^[MLZ\d.,\s-]+$/i.test(region.path), 'Invalid SVG boundary.');
      if (mode === 'energy') ensure(Array.isArray(region.label) && region.label.length === 2 && region.label.every(Number.isFinite), 'Energy geometry requires finite label coordinates.');
      seen.add(region.id);
    }
  }
  return geometry;
}

function runMetadata(run, name, generated, now) {
  record(run, name);
  ensure(['available', 'unavailable'].includes(run.status), `${name}.status is unsupported.`);
  if (run.status === 'unavailable') {
    text(run.reason, `${name}.reason`);
    ensure(!run.records || (Array.isArray(run.records) && run.records.length === 0), `${name} cannot carry records while unavailable.`);
    return;
  }
  text(run.model_version, `${name}.model_version`, 100);
  const issue = timestamp(run.issued_at_utc, `${name}.issued_at_utc`);
  const cutoff = timestamp(run.input_cutoff_utc, `${name}.input_cutoff_utc`);
  ensure(cutoff <= issue, `${name} cutoff must be no later than issue time.`);
  if (run.forecast_origin_utc !== undefined) {
    const origin = timestamp(run.forecast_origin_utc, `${name}.forecast_origin_utc`);
    ensure(name === 'outage' && cutoff <= origin && origin <= issue, 'Outage forecast origin must fall between input cutoff and publication.');
  }
  ensure(issue <= generated && issue <= now + 300000, `${name} issue time is in the future or after export time.`);
  number(run.max_age_hours, `${name}.max_age_hours`, 0.01, 168);
  ensure(Array.isArray(run.records), `${name}.records must be an array.`);
}

export function validatePayload(payload, geometry, wallClock = Date.now()) {
  record(payload, 'Payload');
  ensure(payload.schema_version === 1, 'Unsupported schema_version. Expected 1.');
  ensure(SOURCES.includes(payload.kind === 'model' ? 'model' : payload.kind), 'Payload kind must be dummy or model.');
  const generated = timestamp(payload.generated_at_utc, 'generated_at_utc');
  const now = payload.kind === 'dummy' ? generated : wallClock;
  ensure(generated <= now + 300000, 'Export time is in the future.');
  runMetadata(payload.price, 'price', generated, now);
  runMetadata(payload.outage, 'outage', generated, now);
  const ids = Object.fromEntries(MODES.map(mode => [mode, new Set(geometry[mode].regions.map(region => region.id))]));
  const priceIndex = new Map();
  const outageIndex = new Map();
  const decisionIndex = new Map();
  const timelines = { energy: [], outages: [] };
  let available = 0;
  let partial = false;
  const price = payload.price;
  if (price.status === 'available') {
    text(price.target_date, 'price.target_date', 10);
    ensure(/^\d{4}-\d{2}-\d{2}$/.test(price.target_date), 'Invalid target date.');
    const horizon = price.horizon === undefined ? 'full_day' : price.horizon;
    ensure(['full_day', 'remaining_day'].includes(horizon), 'Unsupported price horizon.');
    const issue = Date.parse(price.issued_at_utc);
    if (horizon === 'remaining_day') ensure(dayOf(issue) === price.target_date, 'Remaining-day prices must be issued during their target Central operating day.');
    ensure(price.records.length > 0 && price.records.length <= 800, 'Price records must contain 1–800 rows.');
    const byStart = new Map();
    const damByHour = new Map();
    for (const row of price.records) {
      record(row, 'Price row');
      ensure(ids.energy.has(row.settlement_point), 'Unknown settlement_point.');
      const start = timestamp(row.interval_start_utc, 'Price interval start');
      const end = timestamp(row.interval_end_utc, 'Price interval end');
      ensure(end - start === 900000 && start % 900000 === 0, 'Price intervals must be aligned 15-minute intervals.');
      ensure(start >= Date.parse(price.issued_at_utc), 'Price interval precedes issue time.');
      ensure(row.delivery_date === price.target_date && dayOf(start) === price.target_date, 'Price delivery date disagrees with Central Time.');
      const local = parts(start);
      ensure(row.hour_ending === Number(local.hour) + 1 && row.quarter === Number(local.minute) / 15 + 1, 'Price hour_ending or quarter disagrees with UTC interval.');
      ensure(['N', 'Y'].includes(row.repeated_hour_flag), 'Invalid repeated_hour_flag.');
      const key = `${row.settlement_point}|${row.interval_start_utc}`;
      ensure(!priceIndex.has(key), 'Duplicate price identity.');
      ensure(['available', 'unavailable'].includes(row.availability), 'Price availability must be explicit.');
      if (row.availability === 'available') {
        for (const field of ['dam_spp_usd_mwh', 'rtm_mean_usd_mwh', 'rtm_p10_usd_mwh', 'rtm_p50_usd_mwh', 'rtm_p90_usd_mwh']) number(row[field], field);
        ensure(row.rtm_p10_usd_mwh <= row.rtm_p50_usd_mwh && row.rtm_p50_usd_mwh <= row.rtm_p90_usd_mwh, 'Price quantiles must satisfy P10 ≤ P50 ≤ P90.');
        const hourKey = `${row.settlement_point}|${Math.floor(start / 3600000)}`;
        ensure(!damByHour.has(hourKey) || damByHour.get(hourKey) === row.dam_spp_usd_mwh, 'DAM must repeat across matching hourly quarters.');
        damByHour.set(hourKey, row.dam_spp_usd_mwh);
        available++;
      } else {
        text(row.reason, 'Unavailable price reason');
        partial = true;
      }
      priceIndex.set(key, row);
      if (byStart.has(start)) ensure(byStart.get(start).repeated_hour_flag === row.repeated_hour_flag, 'Inconsistent repeated-hour flags across zones.');
      byStart.set(start, row);
    }
    const ordered = [...byStart.keys()].sort((first, second) => first - second);
    const first = parts(ordered[0]);
    const after = parts(ordered.at(-1) + 900000);
    ensure(after.hour === '00' && after.minute === '00' && dayOf(ordered.at(-1) + 900000) !== price.target_date, 'Energy timeline must end at the next Central midnight.');
    if (horizon === 'remaining_day') {
      ensure(ordered[0] === Math.ceil(issue / 900000) * 900000, 'Remaining-day prices must begin at the first aligned interval at or after issue.');
    } else {
      ensure([92, 96, 100].includes(ordered.length), 'An energy day must have 92, 96, or 100 intervals.');
      ensure(first.hour === '00' && first.minute === '00', 'Energy timeline must span the full Central operating day.');
    }
    for (const [index, start] of ordered.entries()) {
      ensure(index === 0 || start - ordered[index - 1] === 900000, 'Energy timeline contains a gap.');
      const local = parts(start);
      const previousHour = parts(start - 3600000);
      const repeated = dayOf(start - 3600000) === price.target_date && local.hour === previousHour.hour && local.minute === previousHour.minute;
      const row = byStart.get(start);
      ensure(row.repeated_hour_flag === (repeated ? 'Y' : 'N'), 'Incorrect DST repeated_hour_flag.');
      timelines.energy.push({ interval_start_utc: row.interval_start_utc, interval_end_utc: row.interval_end_utc });
    }
    if (price.records.length !== ordered.length * ids.energy.size) partial = true;
  } else partial = true;
  const outage = payload.outage;
  if (outage.status === 'available') {
    ensure(outage.grain === 'county_scenario', 'Outage grain must be county_scenario; home forecasts require another interface.');
    ensure(outage.probability_kind === 'first_onset', 'Outage probability_kind must be first_onset, not conditional hazard.');
    text(outage.scenario_definition, 'outage.scenario_definition');
    ensure(Array.isArray(outage.intervals) && outage.intervals.length === 24, 'Outage forecast requires 24 hourly intervals.');
    ensure(outage.records.length <= 254, 'Too many county records.');
    for (const [index, interval] of outage.intervals.entries()) {
      record(interval, 'Outage interval');
      const start = timestamp(interval.interval_start_utc, 'Outage interval start');
      const end = timestamp(interval.interval_end_utc, 'Outage interval end');
      ensure(end - start === 3600000 && start % 3600000 === 0, 'Outage intervals must be aligned one-hour intervals.');
      const origin = Date.parse(outage.forecast_origin_utc ?? outage.issued_at_utc);
      ensure(index === 0 ? start >= origin && start - origin < 3600000 : start === Date.parse(outage.intervals[index - 1].interval_end_utc), 'Outage intervals must be contiguous from the forecast origin hour.');
    }
    timelines.outages = outage.intervals;
    for (const row of outage.records) {
      record(row, 'County row');
      ensure(ids.outages.has(row.county_fips), 'Unknown county FIPS.');
      ensure(!outageIndex.has(row.county_fips), 'Duplicate county record.');
      ensure(['observed', 'scenario', 'unknown'].includes(row.coverage), 'County coverage must be observed, scenario, or unknown.');
      if (row.coverage === 'scenario') {
        ensure(row.at_risk_assumed === true && row.active_outage === null, 'A conditional scenario must explicitly assume no active outage.');
        partial = true;
      }
      if (row.coverage === 'unknown') {
        ensure(row.p_first_start_by_hour === null && row.p_any_next_24h === null && row.active_outage === null, 'Unknown coverage cannot contain measured risk or an active scenario.');
        partial = true;
      } else if (row.active_outage !== null) {
        record(row.active_outage, 'active_outage');
        ensure(row.p_first_start_by_hour === null && row.p_any_next_24h === null, 'An active scenario cannot carry pre-outage onset predictions.');
        const active = row.active_outage;
        const started = timestamp(active.outage_started_at, 'outage_started_at');
        const elapsed = (Date.parse(outage.issued_at_utc) - started) / 60000;
        number(active.elapsed_minutes, 'elapsed_minutes', 0);
        ensure(elapsed >= 0 && Math.abs(elapsed - active.elapsed_minutes) < 1, 'Outage age disagrees with its start and issue time.');
        let previous = 1;
        for (const hours of [1, 4, 12, 24]) {
          const probability = active[`p_remaining_gt_${hours}h`];
          number(probability, 'Remaining-time probability', 0, 1);
          ensure(probability <= previous, 'Remaining-time probabilities must be nonincreasing.');
          previous = probability;
        }
        available++;
      } else {
        ensure(Array.isArray(row.p_first_start_by_hour) && row.p_first_start_by_hour.length === 24, 'A county needs 24 first-onset probabilities.');
        row.p_first_start_by_hour.forEach(value => number(value, 'Hourly county probability', 0, 1));
        number(row.p_any_next_24h, 'p_any_next_24h', 0, 1);
        const expected = row.p_first_start_by_hour.reduce((total, probability) => total + probability, 0);
        ensure(expected <= 1 + 0.000001 && Math.abs(expected - row.p_any_next_24h) < 0.000001, 'First-onset probabilities must sum to p_any_next_24h and at most 1.');
        available++;
      }
      outageIndex.set(row.county_fips, row);
    }
    if (outage.records.length !== ids.outages.size) partial = true;
  } else partial = true;
  record(payload.decisions, 'decisions');
  text(payload.decisions.rule_version, 'rule_version', 100);
  ensure(typeof payload.decisions.is_demo === 'boolean', 'decisions.is_demo must be explicit.');
  ensure(payload.kind !== 'dummy' || payload.decisions.is_demo, 'Dummy actions must be labeled Demo rules.');
  ensure(Array.isArray(payload.decisions.records) && payload.decisions.records.length <= 7000, 'Invalid decisions array.');
  for (const row of payload.decisions.records) {
    record(row, 'Decision');
    ensure(MODES.includes(row.mode) && ids[row.mode].has(row.region_id), 'Unknown decision geography.');
    text(row.action, 'Decision action', 20);
    ensure(Object.hasOwn(ACTIONS, row.action), 'Unknown decision action.');
    number(row.strength, 'Decision strength', 0, 1);
    ensure(typeof row.reserve_constraint === 'boolean', 'reserve_constraint must be boolean.');
    ensure(!row.reserve_constraint || row.action !== 'discharge', 'A reserve-constrained decision cannot recommend discharge.');
    ensure(Array.isArray(row.reason_codes) && row.reason_codes.length > 0 && row.reason_codes.length <= 8 && row.reason_codes.every(reason => Object.hasOwn(REASONS, reason)), 'Unknown or missing decision reason code.');
    ensure(timelines[row.mode].some(interval => interval.interval_start_utc === row.interval_start_utc), 'Decision interval is absent from its model timeline.');
    const key = `${row.mode}|${row.region_id}|${row.interval_start_utc}`;
    ensure(!decisionIndex.has(key), 'Duplicate decision identity.');
    const primary = row.mode === 'energy' ? priceIndex.get(`${row.region_id}|${row.interval_start_utc}`) : outageIndex.get(row.region_id);
    ensure(primary && (row.mode === 'energy' ? primary.availability === 'available' : ['observed', 'scenario'].includes(primary.coverage)), 'Decision is missing an available primary signal.');
    if (row.relationship !== null) {
      record(row.relationship, 'relationship');
      const relationship = row.relationship;
      ensure(MODES.includes(relationship.mode) && relationship.mode !== row.mode && ids[relationship.mode].has(relationship.region_id), 'Invalid relationship geography.');
      ensure(['verified_pair', 'documented_aggregate', 'illustrative_pair', 'representative_scenario'].includes(relationship.method), 'Unknown relationship method.');
      text(relationship.description, 'Relationship description');
      ensure(relationship.method !== 'illustrative_pair' || payload.kind === 'dummy', 'Illustrative relationships are allowed only in dummy data.');
      ensure(relationship.method !== 'representative_scenario' || payload.decisions.scope === 'simulated_household', 'Representative pairings require an explicit simulated household scope.');
      if (relationship.method !== 'illustrative_pair') text(relationship.evidence, 'Relationship evidence', 1000);
    }
    if (payload.decisions.scope === 'simulated_household') {
      ensure(typeof row.locked === 'boolean', 'A simulated action must declare whether it is locked history.');
      const planned = timestamp(row.planned_at_utc, 'Action plan time');
      const basis = timestamp(row.basis_outage_issued_at_utc, 'Action outage basis');
      ensure(basis <= planned && planned <= Date.parse(row.interval_start_utc) && planned <= generated, 'Action plan must precede its interval and use an already issued outage forecast.');
      for (const field of ['charge_kwh', 'discharge_kwh']) number(row[field], field, -1e-7, 1.2500001);
      for (const field of ['stored_energy_start_kwh', 'stored_energy_end_kwh', 'reserve_kwh']) number(row[field], field, -1e-7, 25.0000001);
      ensure(typeof row.risk_window_covered === 'boolean', 'Risk-window coverage must be explicit.');
      ensure(row.charge_kwh * row.discharge_kwh < 1e-7, 'A simulated battery cannot charge and discharge together.');
      const expectedAction = row.charge_kwh > 1e-6 ? 'charge' : row.discharge_kwh > 1e-6 ? 'discharge' : 'hold';
      ensure(row.action === expectedAction, 'Simulated action disagrees with its energy flows.');
      const expectedEnergy = row.stored_energy_start_kwh + Math.sqrt(.9) * row.charge_kwh - row.discharge_kwh / Math.sqrt(.9);
      ensure(Math.abs(row.stored_energy_end_kwh - expectedEnergy) < 1e-6 && row.stored_energy_end_kwh >= row.reserve_kwh - 1e-6, 'Simulated battery energy or reserve is inconsistent.');
    }
    decisionIndex.set(key, row);
  }
  if (payload.decisions.scope === 'simulated_household') {
    record(payload.decisions.battery, 'Simulated battery');
    const battery = payload.decisions.battery;
    for (const field of ['capacity_kwh', 'power_kw', 'initial_and_terminal_kwh']) number(battery[field], field, 0.01, 1000);
    number(battery.round_trip_efficiency, 'round_trip_efficiency', 0.01, 1);
    const terminal = battery.initial_and_terminal_kwh;
    number(terminal, 'Daily starting and ending inventory', 0, 25);
    for (const zone of ids.energy) {
      // UTC adjacency includes DST and locked history; missing actions stay unknown.
      const rows = timelines.energy.map(interval => decisionIndex.get(`energy|${zone}|${interval.interval_start_utc}`));
      const firstUnlocked = rows.findIndex(row => row && !row.locked);
      const planStart = firstUnlocked < 0 ? null : rows[firstUnlocked].stored_energy_start_kwh;
      for (const [index, row] of rows.entries()) {
        if (!row) continue;
        const local = parts(row.interval_start_utc);
        if (index === 0 && local.hour === '00' && local.minute === '00') ensure(Math.abs(row.stored_energy_start_kwh - terminal) < 1e-6, 'Simulated battery must start the day at its declared inventory.');
        if (index === rows.length - 1) ensure(Math.abs(row.stored_energy_end_kwh - terminal) < 1e-6, 'Simulated battery must end the day at its declared inventory.');
        const previous = rows[index - 1];
        if (previous) ensure(Math.abs(previous.stored_energy_end_kwh - row.stored_energy_start_kwh) < 1e-6, 'Simulated battery energy must be continuous between adjacent intervals.');
        if (row.locked) continue;
        ensure(index >= firstUnlocked && row.relationship?.mode === 'outages' && row.relationship.method === 'representative_scenario', 'An unlocked simulated action needs its representative county scenario.');
        ensure(row.basis_outage_issued_at_utc === outage.issued_at_utc, 'An unlocked simulated action must use the published outage run.');
        const county = outageIndex.get(row.relationship.region_id);
        ensure(county && Array.isArray(county.p_first_start_by_hour), 'The representative county needs first-onset probabilities.');
        const boundary = Date.parse(row.interval_start_utc) + 900000;
        const offset = (boundary - Date.parse(outage.forecast_origin_utc ?? outage.issued_at_utc)) / 3600000;
        const covered = offset >= 0 && offset + 6 <= 24;
        let mass = 0;
        if (covered) for (let hour = 0; hour < 24; hour++) {
          mass += county.p_first_start_by_hour[hour] * Math.max(0, Math.min(offset + 6, hour + 1) - Math.max(offset, hour));
        }
        const target = 5 + (covered ? 10 * Math.min(1, mass / .25) : 0);
        const reachable = planStart + Math.sqrt(battery.round_trip_efficiency) * battery.power_kw * .25 * (index - firstUnlocked + 1);
        ensure(row.risk_window_covered === covered && Math.abs(row.reserve_kwh - Math.min(target, reachable)) < 1e-6, 'Simulated battery reserve disagrees with its representative county risk.');
      }
    }
    for (const row of payload.decisions.records.filter(row => row.mode === 'outages' && !row.locked)) {
      ensure(row.relationship?.mode === 'energy' && row.relationship.method === 'representative_scenario', 'An unlocked county action needs its representative energy plan.');
      const energy = decisionIndex.get(`energy|${row.relationship.region_id}|${row.interval_start_utc}`);
      if (!energy) continue;
      ensure(energy && energy.relationship?.region_id === row.region_id, 'Representative county and energy actions must be paired.');
      for (const field of ['charge_kwh', 'discharge_kwh', 'stored_energy_start_kwh', 'stored_energy_end_kwh', 'reserve_kwh']) ensure(Math.abs(row[field] - energy[field]) < 1e-6, 'Representative county and energy actions disagree.');
      ensure(row.risk_window_covered === energy.risk_window_covered && row.basis_outage_issued_at_utc === energy.basis_outage_issued_at_utc, 'Representative county and energy risk bases disagree.');
    }
  }
  if (available && decisionIndex.size < priceIndex.size + outageIndex.size * timelines.outages.length) partial = true;
  const stale = isStale(payload, timelines, now);
  const status = !available ? 'empty' : stale ? 'stale' : partial ? 'partial' : 'current';
  return { payload, status, partial, stale, timelines, priceIndex, outageIndex, decisionIndex };
}

function isStale(payload, timelines, now) {
  return [payload.price, payload.outage].some(run => run.status === 'available' && now - Date.parse(run.forecast_origin_utc ?? run.issued_at_utc) > run.max_age_hours * 3600000)
    || Object.values(timelines).some(timeline => timeline.length && now >= Date.parse(timeline.at(-1).interval_end_utc));
}

export function unavailablePayload(reason) {
  return { schema_version: 1, kind: 'model', generated_at_utc: new Date().toISOString().slice(0, 19) + 'Z', price: { status: 'unavailable', reason }, outage: { status: 'unavailable', reason }, decisions: { rule_version: 'not-configured', is_demo: false, records: [] } };
}

export function initialState() {
  return { mode: 'energy', timeIndex: 0, previewRegion: null, selectedRegion: null, dataSource: 'dummy', payloadStatus: 'loading', page: 'map', requestId: 0, data: null, error: null, saved: null };
}

export function snapshot(state) {
  return { version: 1, mode: state.mode, dataSource: state.dataSource, time: state.data?.timelines[state.mode][state.timeIndex]?.interval_start_utc ?? null, selectedRegion: state.selectedRegion };
}

export function readSnapshot(raw) {
  try {
    const value = JSON.parse(raw);
    if (!value || value.version !== 1 || !MODES.includes(value.mode) || !SOURCES.includes(value.dataSource)) return null;
    if (value.time !== null) timestamp(value.time, 'Saved interval');
    if (value.selectedRegion !== null && (typeof value.selectedRegion !== 'string' || value.selectedRegion.length > 30)) return null;
    return value;
  } catch { return null; }
}

function restored(state, saved, geometry) {
  if (!saved) return state;
  const mode = saved.mode;
  const timeline = state.data?.timelines[mode] ?? [];
  const selectedTime = Date.parse(saved.time);
  const found = timeline.findIndex(interval => Date.parse(interval.interval_start_utc) <= selectedTime && selectedTime < Date.parse(interval.interval_end_utc));
  const selectedRegion = found >= 0 && geometry[mode].regions.some(region => region.id === saved.selectedRegion) ? saved.selectedRegion : null;
  return { ...state, mode, timeIndex: Math.max(0, found), selectedRegion, previewRegion: null, saved: null, page: 'map' };
}

export function transition(state, event, context) {
  if (!event || !EVENTS.includes(event.type)) return state;
  const timeline = state.data?.timelines[state.mode] ?? [];
  const ready = state.page === 'map' && ['current', 'partial', 'stale'].includes(state.payloadStatus) && timeline.length > 0;
  const validRegion = region => ready && context.geometry[state.mode].regions.some(item => item.id === region);
  switch (event.type) {
    case 'LOAD':
      if (!SOURCES.includes(event.source)) return state;
      return { ...state, dataSource: event.source, payloadStatus: 'loading', data: null, error: null, requestId: state.requestId + 1, previewRegion: null, selectedRegion: null, timeIndex: 0, saved: event.saved ?? null };
    case 'RESOLVE': {
      if (event.requestId !== state.requestId || state.payloadStatus !== 'loading') return state;
      try {
        const data = validatePayload(event.payload, context.geometry, context.now);
        ensure(event.payload.kind === state.dataSource, 'The selected source disagrees with payload kind.');
        const resolved = { ...state, data, payloadStatus: data.status, error: null };
        if (state.saved) return restored(resolved, state.saved, context.geometry);
        const intervals = data.timelines[state.mode];
        const target = intervals.findIndex(interval => state.dataSource === 'dummy' ? parts(interval.interval_start_utc).hour === '15' : Date.parse(interval.interval_end_utc) > (context.now ?? Date.now()));
        return { ...resolved, timeIndex: target >= 0 ? target : Math.max(0, intervals.length - 1) };
      } catch (error) {
        return { ...state, payloadStatus: 'error', data: null, error: error.message.slice(0, 240), selectedRegion: null, previewRegion: null, saved: null };
      }
    }
    case 'REJECT':
      if (event.requestId !== state.requestId || state.payloadStatus !== 'loading') return state;
      return { ...state, payloadStatus: 'error', data: null, error: String(event.message || 'Forecast could not be loaded.').slice(0, 240), saved: null };
    case 'MODE': {
      if (!MODES.includes(event.mode) || state.page !== 'map' || event.mode === state.mode) return state;
      const selectedTime = Date.parse(timeline[state.timeIndex]?.interval_start_utc);
      const nextTimeline = state.data?.timelines[event.mode] ?? [];
      const timeIndex = nextTimeline.findIndex(interval => Date.parse(interval.interval_start_utc) <= selectedTime && selectedTime < Date.parse(interval.interval_end_utc));
      const saved = state.saved ? { ...state.saved, mode: event.mode, selectedRegion: null } : null;
      return { ...state, mode: event.mode, timeIndex: Math.max(0, timeIndex), selectedRegion: null, previewRegion: null, saved };
    }
    case 'TIME':
      if (!ready || !Number.isInteger(event.index) || event.index < 0 || event.index >= timeline.length) return state;
      return { ...state, timeIndex: event.index, previewRegion: null };
    case 'PREVIEW':
      return validRegion(event.region) ? { ...state, previewRegion: event.region } : state;
    case 'UNPREVIEW':
      return !event.region || event.region === state.previewRegion ? { ...state, previewRegion: null } : state;
    case 'PIN':
      return validRegion(event.region) ? { ...state, selectedRegion: event.region, previewRegion: null } : state;
    case 'CLEAR':
      return { ...state, previewRegion: validRegion(event.region) ? event.region : null, selectedRegion: null };
    case 'RULES':
      return { ...state, page: 'rules', previewRegion: null, saved: snapshot(state) };
    case 'RETURN':
      return restored({ ...state, page: 'map' }, state.saved, context.geometry);
    case 'RESTORE': {
      const saved = readSnapshot(event.raw);
      if (!saved) return state;
      return transition({ ...state, mode: saved.mode }, { type: 'LOAD', source: saved.dataSource, saved }, context);
    }
    case 'TICK': {
      if (state.dataSource === 'dummy' || !state.data || !Number.isFinite(event.now) || event.now < Date.parse(state.data.payload.generated_at_utc)) return state;
      const stale = isStale(state.data.payload, state.data.timelines, event.now);
      const payloadStatus = state.payloadStatus === 'empty' ? 'empty' : stale ? 'stale' : state.data.partial ? 'partial' : 'current';
      if (payloadStatus === state.payloadStatus) return state;
      return { ...state, data: { ...state.data, status: payloadStatus, stale }, payloadStatus };
    }
    default:
      return state;
  }
}

export function viewSelection(state) {
  if (!state.data) return null;
  const region = state.previewRegion ?? state.selectedRegion;
  const interval = state.data.timelines[state.mode][state.timeIndex];
  if (!region || !interval) return null;
  const decision = state.data.decisionIndex.get(`${state.mode}|${region}|${interval.interval_start_utc}`) ?? null;
  const relationship = decision?.relationship;
  const energyId = state.mode === 'energy' ? region : relationship?.mode === 'energy' ? relationship.region_id : null;
  const outageId = state.mode === 'outages' ? region : relationship?.mode === 'outages' ? relationship.region_id : null;
  const target = Date.parse(interval.interval_start_utc);
  const energyInterval = state.data.timelines.energy.find(item => Date.parse(item.interval_start_utc) <= target && target < Date.parse(item.interval_end_utc));
  const outageHour = state.data.timelines.outages.findIndex(item => Date.parse(item.interval_start_utc) <= target && target < Date.parse(item.interval_end_utc));
  return { region, interval, decision, relationship, energyId, outageId,
    energy: energyId && energyInterval ? state.data.priceIndex.get(`${energyId}|${energyInterval.interval_start_utc}`) ?? null : null,
    outage: outageId && outageHour >= 0 ? state.data.outageIndex.get(outageId) ?? null : null,
    outageHour, energyInterval };
}

export async function readJsonResponse(response) {
  ensure(response.ok, `Forecast request failed (HTTP ${response.status}).`);
  const declared = Number(response.headers.get('content-length'));
  ensure(!Number.isFinite(declared) || declared <= MAX_BYTES, 'Forecast file exceeds 10 MB.');
  let content = '';
  let total = 0;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      total += chunk.value.byteLength;
      ensure(total <= MAX_BYTES, 'Forecast file exceeds 10 MB.');
      content += decoder.decode(chunk.value, { stream: true });
    }
    content += decoder.decode();
    try { return JSON.parse(content); } catch { throw new Error('Forecast file is not valid JSON.'); }
  } finally { await reader.cancel().catch(() => {}); }
}
