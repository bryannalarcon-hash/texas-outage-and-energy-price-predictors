import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODES, SOURCES, STATUSES,
  validateGeometry, validatePayload, initialState, transition, snapshot, readSnapshot,
  viewSelection, unavailablePayload
} from './core.mjs';
import { geometry, dummy, now, modelPayload, priceDay } from './test-data.mjs';

const context = { geometry, now };
const step = (state, event) => transition(state, event, context);
const validate = (payload, wallClock = now) => validatePayload(payload, geometry, wallClock);
const emptyPayload = () => ({ ...unavailablePayload('No forecast export'), generated_at_utc: dummy.generated_at_utc });
function loaded(payload = modelPayload(), mode = 'energy', wallClock = now) {
  const state = step({ ...initialState(), mode }, { type: 'LOAD', source: payload.kind });
  return transition(state, { type: 'RESOLVE', requestId: state.requestId, payload }, { geometry, now: wallClock });
}

// Finite equivalence classes: both sources × both modes × every payload status.
const classes = new Map();
function stateClass(source, mode, status) {
  const key = `${source}/${mode}/${status}`;
  if (classes.has(key)) return classes.get(key);
  let state = step({ ...initialState(), mode }, { type: 'LOAD', source });
  if (status === 'error') state = step(state, { type: 'REJECT', requestId: state.requestId, message: 'Network failed' });
  else if (status !== 'loading') {
    const payload = status === 'empty' ? emptyPayload() : modelPayload();
    payload.kind = source;
    payload.decisions.is_demo = true;
    if (status === 'partial') payload.decisions.records.pop();
    if (status === 'stale') payload.price.max_age_hours = 0.01;
    state = step(state, { type: 'RESOLVE', requestId: state.requestId, payload });
  }
  assert.equal(state.payloadStatus, status, `Fixture ${key}: ${state.error ?? 'wrong status'}`);
  classes.set(key, state);
  return state;
}

test('fixtures: validated geometry has eight unique load zones and 254 unique counties', () => {
  assert.equal(validateGeometry(geometry), geometry);
  for (const [mode, count] of [['energy', 8], ['outages', 254]]) {
    assert.equal(geometry[mode].regions.length, count);
    assert.equal(new Set(geometry[mode].regions.map(row => row.id)).size, count);
  }
  const missingLabel = structuredClone(geometry);
  delete missingLabel.energy.regions[0].label;
  assert.throws(() => validateGeometry(missingLabel), /label coordinates/);
});

test('fixtures: original dummy is partial and MODEL is current with every interval decision', () => {
  assert.equal(validate(dummy).status, 'partial');
  const original = structuredClone(dummy);
  const payload = modelPayload();
  const data = validate(payload);
  assert.equal(payload.kind, 'model');
  assert.equal(payload.decisions.is_demo, true);
  assert.equal(data.status, 'current');
  assert.equal(data.partial, false);
  assert.equal(data.stale, false);
  assert.equal(data.priceIndex.size, 8 * 96);
  assert.equal(data.outageIndex.size, 254);
  assert.equal(data.decisionIndex.size, 8 * 96 + 254 * 24);
  assert.equal(payload.decisions.records.some(row => row.relationship?.method === 'illustrative_pair'), false);
  for (const row of original.outage.records.filter(row => row.coverage === 'unknown')) {
    const filled = data.outageIndex.get(row.county_fips);
    assert.equal(filled.coverage, 'observed');
    assert.equal(filled.p_any_next_24h, 0);
    assert.deepEqual(filled.p_first_start_by_hour, Array(24).fill(0));
    for (const interval of data.timelines.outages) {
      const decision = data.decisionIndex.get(`outages|${row.county_fips}|${interval.interval_start_utc}`);
      assert.equal(decision.action, 'hold');
      assert.equal(decision.reserve_constraint, true);
    }
  }
  payload.price.records[0].rtm_mean_usd_mwh = 999;
  assert.deepEqual(dummy, original, 'Fixture mutation must not alter imported JSON');
  assert.deepEqual(modelPayload().price.records, original.price.records);
});

test('contract: dummy identifies first-onset probabilities and sums every observed county', () => {
  assert.equal(dummy.outage.probability_kind, 'first_onset');
  assert.equal(typeof dummy.outage.scenario_definition, 'string');
  assert.ok(dummy.outage.scenario_definition.trim().length > 0);
  for (const row of dummy.outage.records.filter(row => row.coverage === 'observed' && !row.active_outage)) {
    assert.equal(row.p_first_start_by_hour.length, 24, row.county_fips);
    assert.ok(row.p_first_start_by_hour.every(value => Number.isFinite(value) && value >= 0 && value <= 1), row.county_fips);
    const sum = row.p_first_start_by_hour.reduce((total, value) => total + value, 0);
    assert.ok(sum <= 1, `${row.county_fips}: first-onset mass ${sum} exceeds 1`);
    assert.ok(Math.abs(sum - row.p_any_next_24h) <= 1e-6, `${row.county_fips}: aggregate must equal sum, not product of hazards`);
  }
});

test('contract: decision actions must be typed strings before policy checks', () => {
  const payload = modelPayload();
  payload.decisions.records[0].action = ['discharge'];
  payload.decisions.records[0].reserve_constraint = true;
  assert.throws(() => validate(payload), /Decision action must be nonempty text/);
});

test('FSM: initial state, snapshot and unknown events are deterministic', () => {
  assert.deepEqual(initialState(), {
    mode: 'energy', timeIndex: 0, previewRegion: null, selectedRegion: null,
    dataSource: 'dummy', payloadStatus: 'loading', page: 'map', requestId: 0,
    data: null, error: null, saved: null
  });
  const state = initialState();
  assert.deepEqual(snapshot(state), { version: 1, mode: 'energy', dataSource: 'dummy', time: null, selectedRegion: null });
  for (const event of [null, undefined, {}, { type: 'BOGUS' }]) assert.equal(step(state, event), state);
  assert.equal(viewSelection(state), null);
});

for (const source of SOURCES) for (const mode of MODES) for (const status of STATUSES) {
  test(`FSM ${source}/${mode}/${status}: selection, rules, mode, load and response guards`, () => {
    const state = stateClass(source, mode, status);
    const ready = ['current', 'partial', 'stale'].includes(status);
    const region = geometry[mode].regions[0].id;
    const otherMode = mode === 'energy' ? 'outages' : 'energy';
    for (const type of ['PIN', 'PREVIEW']) {
      const result = step(state, { type, region });
      if (ready) assert.equal(result[type === 'PIN' ? 'selectedRegion' : 'previewRegion'], region);
      else assert.equal(result, state, `${type} must be guarded while ${status}`);
      for (const invalid of [undefined, '', 'invalid', geometry[otherMode].regions[0].id]) assert.equal(step(state, { type, region: invalid }), state);
    }
    const moved = step(state, { type: 'TIME', index: 1 });
    if (ready) assert.equal(moved.timeIndex, 1);
    else assert.equal(moved, state);
    const rules = step(state, { type: 'RULES' });
    assert.equal(rules.page, 'rules');
    assert.deepEqual(rules.saved, snapshot(state));
    for (const event of [{ type: 'TIME', index: 1 }, { type: 'PIN', region }, { type: 'PREVIEW', region }, { type: 'MODE', mode: otherMode }]) {
      assert.equal(step(rules, event), rules, `${event.type} must be guarded on rules page`);
    }
    const returned = step(rules, { type: 'RETURN' });
    assert.equal(returned.page, 'map');
    assert.equal(returned.mode, mode);
    assert.equal(returned.dataSource, source);
    assert.equal(returned.timeIndex, state.timeIndex);
    assert.equal(returned.payloadStatus, status);
    assert.equal(returned.saved, null);
    assert.equal(step(state, { type: 'MODE', mode }), state);
    assert.equal(step(state, { type: 'MODE', mode: 'invalid' }), state);
    const changed = step(state, { type: 'MODE', mode: otherMode });
    assert.equal(changed.mode, otherMode);
    assert.equal(changed.dataSource, source);
    assert.equal(changed.payloadStatus, status);
    assert.equal(changed.selectedRegion, null);
    assert.equal(changed.previewRegion, null);
    assert.equal(step(state, { type: 'LOAD', source: 'invalid' }), state);
    const reloading = step(state, { type: 'LOAD', source });
    assert.equal(reloading.payloadStatus, 'loading');
    assert.equal(reloading.requestId, state.requestId + 1);
    for (const key of ['data', 'error', 'selectedRegion', 'previewRegion', 'saved']) assert.equal(reloading[key], null);
    assert.equal(reloading.timeIndex, 0);
    for (const type of ['RESOLVE', 'REJECT']) {
      assert.equal(step(state, { type, requestId: state.requestId - 1, payload: null }), state);
      if (status !== 'loading') assert.equal(step(state, { type, requestId: state.requestId, payload: null }), state);
    }
    assert.equal(step(state, { type: 'RESTORE', raw: '{broken' }), state);
    if (!state.data) assert.equal(step(state, { type: 'TICK', now: now + 86400000 }), state);
  });
}

for (const source of SOURCES) for (const mode of MODES) {
  test(`FSM ${source}/${mode}: preview overrides pin temporarily; exit, pin and clear are distinct`, () => {
    const [first, second] = geometry[mode].regions.map(row => row.id);
    let state = stateClass(source, mode, 'current');
    assert.equal(viewSelection(state), null);
    state = step(state, { type: 'PIN', region: first });
    assert.equal(viewSelection(state).region, first);
    state = step(state, { type: 'PREVIEW', region: second });
    assert.equal(state.selectedRegion, first);
    assert.equal(viewSelection(state).region, second);
    assert.equal(step(state, { type: 'UNPREVIEW', region: first }), state, 'Late exit must not clear another preview');
    state = step(state, { type: 'UNPREVIEW', region: second });
    assert.equal(state.previewRegion, null);
    assert.equal(viewSelection(state).region, first);
    state = step(state, { type: 'PREVIEW', region: second });
    assert.equal(step(state, { type: 'UNPREVIEW' }).previewRegion, null);
    state = step(state, { type: 'PIN', region: second });
    assert.equal(state.selectedRegion, second);
    assert.equal(state.previewRegion, null);
    state = step(state, { type: 'PREVIEW', region: first });
    state = step(state, { type: 'CLEAR' });
    assert.equal(state.selectedRegion, null);
    assert.equal(state.previewRegion, null);
    assert.equal(viewSelection(state), null);
    assert.deepEqual(step(state, { type: 'CLEAR' }), state);
    state = step(state, { type: 'CLEAR', region: first });
    assert.equal(state.selectedRegion, null);
    assert.equal(state.previewRegion, first);
    assert.equal(viewSelection(state).region, first);
  });

  test(`FSM ${source}/${mode}: TIME accepts endpoints, rejects invalid indices and preserves pin`, () => {
    let state = stateClass(source, mode, 'current');
    const [first, second] = geometry[mode].regions.map(row => row.id);
    const length = state.data.timelines[mode].length;
    state = step(step(state, { type: 'PIN', region: first }), { type: 'PREVIEW', region: second });
    for (const index of [0, length - 1, 0]) {
      state = step(state, { type: 'TIME', index });
      assert.equal(state.timeIndex, index);
      assert.equal(state.selectedRegion, first);
      assert.equal(state.previewRegion, null);
      assert.equal(viewSelection(state).interval, state.data.timelines[mode][index]);
    }
    for (const index of [-1, length, 1.5, '1', NaN, Infinity, undefined]) assert.equal(step(state, { type: 'TIME', index }), state);
  });

  test(`FSM ${source}/${mode}: RULES/RETURN and RESTORE preserve exact UTC time and pin, never hover`, () => {
    let state = stateClass(source, mode, 'current');
    state = step(state, { type: 'TIME', index: 7 });
    state = step(state, { type: 'PIN', region: geometry[mode].regions[0].id });
    state = step(state, { type: 'PREVIEW', region: geometry[mode].regions[1].id });
    const saved = snapshot(state);
    assert.equal(saved.time, state.data.timelines[mode][7].interval_start_utc);
    assert.equal(Object.hasOwn(saved, 'previewRegion'), false);
    assert.deepEqual(readSnapshot(JSON.stringify(saved)), saved);
    const rules = step(state, { type: 'RULES' });
    assert.equal(rules.previewRegion, null);
    const returned = step(rules, { type: 'RETURN' });
    assert.deepEqual(snapshot(returned), saved);
    assert.equal(returned.previewRegion, null);
    let restored = step(initialState(), { type: 'RESTORE', raw: JSON.stringify(saved) });
    assert.equal(restored.mode, mode);
    assert.equal(restored.dataSource, source);
    assert.equal(restored.payloadStatus, 'loading');
    assert.equal(restored.selectedRegion, null);
    restored = step(restored, { type: 'RESOLVE', requestId: restored.requestId, payload: state.data.payload });
    assert.equal(restored.payloadStatus, 'current', restored.error);
    assert.deepEqual(snapshot(restored), saved);
    assert.equal(restored.saved, null);
    assert.equal(restored.previewRegion, null);
  });
}

test('FSM: freshness ticks reuse validated indexes until status changes', () => {
  const state = loaded();
  const frozenDummy = loaded(dummy);
  assert.equal(step(frozenDummy, { type: 'TICK', now: now + 172800000 }), frozenDummy);
  assert.equal(step(state, { type: 'TICK', now: now + 1000 }), state);
  const stale = step(state, { type: 'TICK', now: now + 172800000 });
  assert.equal(stale.payloadStatus, 'stale');
  assert.equal(stale.data.priceIndex, state.data.priceIndex);
  assert.equal(stale.data.outageIndex, state.data.outageIndex);
  assert.equal(stale.data.decisionIndex, state.data.decisionIndex);
  assert.equal(step(stale, { type: 'TICK', now: now + 172860000 }), stale);
});

for (const source of SOURCES) {
  test(`FSM ${source}: MODE uses half-open UTC overlap for every quarter and hour`, () => {
    const energy = stateClass(source, 'energy', 'current');
    for (let index = 0; index < 96; index++) {
      const selected = step(step(step(energy, { type: 'TIME', index }), { type: 'PIN', region: 'LZ_WEST' }), { type: 'PREVIEW', region: 'LZ_HOUSTON' });
      const county = step(selected, { type: 'MODE', mode: 'outages' });
      assert.equal(county.timeIndex, Math.floor(index / 4), `quarter ${index} must select containing UTC hour`);
      assert.equal(county.selectedRegion, null);
      assert.equal(county.previewRegion, null);
      assert.equal(step(county, { type: 'MODE', mode: 'energy' }).timeIndex, Math.floor(index / 4) * 4);
    }
  });

  test(`FSM ${source}: late opposite-source responses and failures cannot beat the latest request`, () => {
    const other = source === 'dummy' ? 'model' : 'dummy';
    const first = step(initialState(), { type: 'LOAD', source: other });
    const latest = step(first, { type: 'LOAD', source });
    const oldPayload = other === 'dummy' ? dummy : modelPayload();
    assert.equal(step(latest, { type: 'RESOLVE', requestId: first.requestId, payload: oldPayload }), latest);
    assert.equal(step(latest, { type: 'REJECT', requestId: first.requestId, message: 'Aborted previous fetch' }), latest);
    const payload = source === 'dummy' ? dummy : modelPayload();
    const resolved = step(latest, { type: 'RESOLVE', requestId: latest.requestId, payload });
    assert.equal(resolved.payloadStatus, source === 'dummy' ? 'partial' : 'current', resolved.error);
    assert.equal(resolved.data.payload, payload);
    assert.equal(step(resolved, { type: 'RESOLVE', requestId: first.requestId, payload: oldPayload }), resolved);
    assert.equal(step(resolved, { type: 'REJECT', requestId: latest.requestId, message: 'Late failure' }), resolved);
    const mismatch = step(latest, { type: 'RESOLVE', requestId: latest.requestId, payload: oldPayload });
    assert.equal(mismatch.payloadStatus, 'error');
    assert.match(mismatch.error, /source.*kind/i);
    assert.equal(mismatch.data, null);
  });

  test(`FSM ${source}: retry after failure gets a new ID and retains explicitly saved context`, () => {
    let selected = stateClass(source, 'outages', 'current');
    selected = step(step(selected, { type: 'TIME', index: 9 }), { type: 'PIN', region: '48201' });
    const saved = snapshot(selected);
    const loading = step(selected, { type: 'LOAD', source, saved });
    const failed = step(loading, { type: 'REJECT', requestId: loading.requestId, message: 'Timed out' });
    assert.equal(failed.payloadStatus, 'error');
    assert.equal(failed.error, 'Timed out');
    assert.equal(failed.saved, null);
    const retry = step(failed, { type: 'LOAD', source, saved });
    assert.equal(retry.requestId, loading.requestId + 1);
    assert.equal(retry.error, null);
    assert.equal(step(retry, { type: 'RESOLVE', requestId: loading.requestId, payload: selected.data.payload }), retry);
    const resolved = step(retry, { type: 'RESOLVE', requestId: retry.requestId, payload: selected.data.payload });
    assert.equal(resolved.payloadStatus, 'current', resolved.error);
    assert.deepEqual(snapshot(resolved), saved);
    assert.equal(step(loading, { type: 'REJECT', requestId: loading.requestId }).error, 'Forecast could not be loaded.');
    assert.equal(step(loading, { type: 'REJECT', requestId: loading.requestId, message: 'x'.repeat(500) }).error.length, 240);
  });

  test(`FSM ${source}: a mode chosen during reload overrides the saved mode`, () => {
    let selected = stateClass(source, 'energy', 'current');
    selected = step(step(selected, { type: 'TIME', index: 61 }), { type: 'PIN', region: 'LZ_HOUSTON' });
    const loading = step(selected, { type: 'LOAD', source, saved: snapshot(selected) });
    const changed = step(loading, { type: 'MODE', mode: 'outages' });
    const payload = source === 'dummy' ? dummy : modelPayload();
    const resolved = step(changed, { type: 'RESOLVE', requestId: changed.requestId, payload });
    assert.equal(resolved.mode, 'outages');
    assert.equal(resolved.timeIndex, 15);
    assert.equal(resolved.selectedRegion, null);
  });
}

test('selection: an explicit county relationship outside its horizon is temporal unavailability, not unknown coverage', () => {
  let state = loaded(dummy, 'energy', now);
  state = step(step(state, { type: 'TIME', index: 0 }), { type: 'PIN', region: 'LZ_HOUSTON' });
  const data = { ...state.data, timelines: { ...state.data.timelines, outages: state.data.timelines.outages.slice(1) } };
  const selection = viewSelection({ ...state, data });
  assert.equal(selection.outageId, '48201');
  assert.equal(selection.outageHour, -1);
  assert.equal(selection.outage, null);
});

test('FSM: malformed RESOLVE clears saved context, reports the field and can recover on retry', () => {
  const loading = step(initialState(), { type: 'LOAD', source: 'model', saved: snapshot(initialState()) });
  const payload = modelPayload();
  payload.price.records[0].rtm_p10_usd_mwh = Infinity;
  const error = step(loading, { type: 'RESOLVE', requestId: loading.requestId, payload });
  assert.equal(error.payloadStatus, 'error');
  assert.match(error.error, /rtm_p10_usd_mwh.*finite/);
  for (const key of ['data', 'selectedRegion', 'previewRegion', 'saved']) assert.equal(error[key], null);
  const retry = step(error, { type: 'LOAD', source: 'model' });
  assert.equal(step(retry, { type: 'RESOLVE', requestId: retry.requestId, payload: modelPayload() }).payloadStatus, 'current');
});

test('contract: delayed publication preserves the fixed 12Z outage origin and its age', () => {
  const payload = modelPayload();
  const origin = payload.outage.issued_at_utc;
  payload.outage.forecast_origin_utc = origin;
  payload.outage.issued_at_utc = new Date(Date.parse(origin) + 2 * 3600000).toISOString().replace('.000Z', 'Z');
  payload.generated_at_utc = payload.outage.issued_at_utc;
  for (const row of payload.outage.records) if (row.active_outage) row.active_outage.elapsed_minutes += 120;
  payload.outage.max_age_hours = 1;
  const result = validate(payload, Date.parse(payload.generated_at_utc));
  assert.equal(result.status, 'stale', 'Publishing later must not renew the forecast age');
  payload.outage.forecast_origin_utc = new Date(Date.parse(payload.outage.issued_at_utc) + 3600000).toISOString().replace('.000Z', 'Z');
  assert.throws(() => validate(payload, Date.parse(payload.generated_at_utc)), /origin/);
});

test('contract: conditional county scenarios cannot impersonate observed outage status', () => {
  const payload = modelPayload();
  const row = payload.outage.records.find(row => row.coverage === 'observed' && !row.active_outage);
  row.coverage = 'scenario';
  row.at_risk_assumed = true;
  assert.doesNotThrow(() => validate(payload));
  row.at_risk_assumed = false;
  assert.throws(() => validate(payload), /assum/);
});

function remainingDay(date = '2025-07-01', start = '2025-07-01T05:00:00Z', count = 96, issue = '2025-07-01T18:07:13Z') {
  const price = priceDay(date, start, count);
  Object.assign(price, { horizon: 'remaining_day', issued_at_utc: issue, input_cutoff_utc: issue });
  price.records = price.records.filter(row => Date.parse(row.interval_start_utc) >= Math.ceil(Date.parse(issue) / 900000) * 900000);
  return { ...emptyPayload(), generated_at_utc: issue, price };
}

test('contract: remaining-day prices start at the next quarter, retain midnight, and allow partial zones', () => {
  const payload = remainingDay();
  const data = validate(payload, Date.parse(payload.generated_at_utc));
  assert.equal(data.timelines.energy.length, 43);
  assert.equal(data.timelines.energy[0].interval_start_utc, '2025-07-01T18:15:00Z');
  assert.equal(data.timelines.energy.at(-1).interval_end_utc, '2025-07-02T05:00:00Z');
  const partial = structuredClone(payload);
  partial.price.records.shift();
  assert.equal(validate(partial, Date.parse(payload.generated_at_utc)).partial, true);
  const aligned = remainingDay(undefined, undefined, undefined, '2025-07-01T18:15:00Z');
  assert.equal(validate(aligned, Date.parse(aligned.generated_at_utc)).timelines.energy[0].interval_start_utc, aligned.generated_at_utc);
});

test('contract: remaining-day prices reject backdating, missing edges, gaps, wrong issue days and unknown horizons', () => {
  const original = remainingDay();
  for (const [mutate, message] of [
    [p => { p.price.records.unshift(priceDay('2025-07-01', '2025-07-01T05:00:00Z', 96).records.find(r => r.interval_start_utc === '2025-07-01T18:00:00Z')); }, /precedes issue/],
    [p => { p.price.records = p.price.records.filter(r => r.interval_start_utc !== '2025-07-01T18:15:00Z'); }, /first aligned/],
    [p => { p.price.records = p.price.records.filter(r => r.interval_start_utc !== '2025-07-01T19:00:00Z'); }, /gap/],
    [p => { p.price.records = p.price.records.filter(r => r.interval_start_utc !== '2025-07-02T04:45:00Z'); }, /midnight/],
    [p => { p.price.issued_at_utc = p.price.input_cutoff_utc = '2025-07-01T04:59:59Z'; }, /target Central operating day/],
    [p => { delete p.price.horizon; }, /92, 96, or 100/],
    [p => { p.price.horizon = 'arbitrary'; }, /Unsupported price horizon/]
  ]) {
    const payload = structuredClone(original);
    mutate(payload);
    assert.throws(() => validate(payload, Date.parse(payload.generated_at_utc)), message);
  }
  const empty = remainingDay(undefined, undefined, undefined, '2025-07-02T04:45:01Z');
  assert.throws(() => validate(empty, Date.parse(empty.generated_at_utc)), /1–800 rows/);
});

test('contract: remaining-day DST horizons preserve elapsed intervals and the true repeated-hour flag', () => {
  for (const [date, start, count, issue, expected, flag] of [
    ['2026-03-08', '2026-03-08T06:00:00Z', 92, '2026-03-08T07:50:00Z', 84, 'N'],
    ['2026-11-01', '2026-11-01T05:00:00Z', 100, '2026-11-01T07:10:00Z', 91, 'Y']
  ]) {
    const payload = remainingDay(date, start, count, issue);
    const data = validate(payload, Date.parse(issue));
    assert.equal(data.timelines.energy.length, expected);
    assert.equal(payload.price.records[0].repeated_hour_flag, flag);
    const full = { ...payload, price: priceDay(date, start, count) };
    assert.equal(validate(full, Date.parse(issue)).timelines.energy.length, count);
    const first = payload.price.records[0].interval_start_utc;
    for (const row of payload.price.records) if (row.interval_start_utc === first) row.repeated_hour_flag = flag === 'Y' ? 'N' : 'Y';
    assert.throws(() => validate(payload, Date.parse(issue)), /Incorrect DST/);
  }
});
