import { ACTIONS, REASONS, RAMPS, REQUEST_TIMEOUT, initialState, transition, snapshot, readSnapshot, validateGeometry, formatTime, formatMoney, colorClass, viewSelection, readJsonResponse, unavailablePayload } from './core.mjs';

const get = id => document.getElementById(id);
const svgNamespace = 'http://www.w3.org/2000/svg';
const storageKey = 'bpc-map-context-v1';
const mobileLayout = window.matchMedia('(max-width: 760px)');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const shortTime = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Chicago', hour: 'numeric', minute: '2-digit' });
const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const percentLabel = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const percent = value => `${percentLabel.format(value * 100)}%`;
let state = initialState();
let geometry = null;
let terrain = null;
let controller = null;
let renderedMode = null;
let lastPanel = '';
let lastAnnouncement = '';
let gradientFrame = null;
let renderedTimeline = null;
let zoomLevel = 1;
let mapView = null;
let pan = null;
let didPan = false;
let keyboardMapFocus = false;
let pointerFocusPending = false;

function dispatch(event) {
  const nextState = transition(state, event, { geometry, now: Date.now() });
  if (nextState === state) return;
  state = nextState;
  render();
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
const rgb = hex => hex.match(/[a-f\d]{2}/gi).map(channel => parseInt(channel, 16));
const mix = (from, to, amount) => from.map((channel, index) => Math.round(channel + (to[index] - channel) * amount));
const theme = getComputedStyle(document.documentElement);
const palettes = Object.fromEntries(['energy', 'outages'].map(mode => [mode, Array.from({ length: 5 }, (_, index) => rgb(theme.getPropertyValue(`--${mode}-${index}`).trim()))]));

function gradientColor(mode, level, position) {
  const color = palettes[mode][level];
  return mode === 'energy'
    ? mix(mix(color, [255, 236, 174], .38 * (1 - position)), [88, 28, 13], position * .24)
    : mix(mix(color, [12, 37, 66], .43 * (1 - position)), [212, 237, 249], position * .45);
}

function paintGradients(from = state.mode, progress = 1) {
  const atmosphere = [];
  for (const [level, gradient] of [...get('map-gradients').children].entries()) {
    for (const [index, stop] of [...gradient.children].entries()) {
      const position = index / (gradient.children.length - 1);
      const front = progress * 1.9 - .45;
      const distance = front - position;
      const blend = progress === 1 ? 1 : clamp(distance / .22 + .5, 0, 1);
      let color = mix(gradientColor(from, level, position), gradientColor(state.mode, level, position), blend);
      if (progress > 0 && progress < 1) {
        const band = Math.max(0, 1 - Math.abs(distance) / .32) * Math.sin(progress * Math.PI);
        const rays = state.mode === 'outages' ? .38 + .16 * Math.sin(position * 58 - progress * 22) : .55;
        color = mix(color, state.mode === 'outages' ? [218, 234, 240] : [255, 239, 182], band * rays);
      }
      stop.setAttribute('stop-color', `rgb(${color.join(' ')})`);
      if (level === 1) atmosphere.push(`rgb(${color.join(' ')} / .28) ${position * 100}%`);
    }
  }
  get('workspace').style.setProperty('--map-atmosphere', `linear-gradient(158deg, ${atmosphere.join(',')})`);
}

function animateGradients(from) {
  cancelAnimationFrame(gradientFrame);
  document.body.classList.remove('mode-flow');
  if (reducedMotion.matches) { paintGradients(); return; }
  document.body.classList.add('mode-flow');
  const started = performance.now();
  paintGradients(from, 0);
  const frame = now => {
    const progress = Math.min(1, (now - started) / 1100);
    paintGradients(from, progress);
    if (progress < 1) gradientFrame = requestAnimationFrame(frame);
    else document.body.classList.remove('mode-flow');
  };
  gradientFrame = requestAnimationFrame(frame);
}

function updateMapView() {
  if (!mapView) return;
  const { left, top, width, height } = mapView;
  const zoomWidth = width / zoomLevel;
  const zoomHeight = height / zoomLevel;
  mapView.centerX = clamp(mapView.centerX, left + zoomWidth / 2, left + width - zoomWidth / 2);
  mapView.centerY = clamp(mapView.centerY, top + zoomHeight / 2, top + height - zoomHeight / 2);
  get('texas-map').setAttribute('viewBox', [mapView.centerX - zoomWidth / 2, mapView.centerY - zoomHeight / 2, zoomWidth, zoomHeight].join(' '));
  get('texas-map').classList.toggle('is-zoomed', zoomLevel > 1);
  get('zoom-reset').textContent = `${Math.round(zoomLevel * 100)}%`;
  get('zoom-in').disabled = zoomLevel >= 2.5;
  get('zoom-out').disabled = zoomLevel <= 1;
  get('zoom-hint').textContent = zoomLevel > 1 ? 'Drag to pan · select 100% to reset the view.' : 'Zoom in to explore terrain and smaller regions.';
}

function zoomMap(level) {
  if (!mapView) return;
  if (zoomLevel === 1 && level > 1 && state.selectedRegion) {
    const bounds = get('texas-map').querySelector(`[data-region="${state.selectedRegion}"]`).getBBox();
    const point = new DOMPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2).matrixTransform(get('map-object').transform.baseVal.consolidate().matrix);
    mapView.centerX = point.x;
    mapView.centerY = point.y;
  }
  zoomLevel = clamp(level, 1, 2.5);
  updateMapView();
}

async function loadSource(source, saved = null) {
  controller?.abort();
  controller = new AbortController();
  const operation = controller;
  dispatch({ type: 'LOAD', source, saved });
  const requestId = state.requestId;
  const timeout = setTimeout(() => operation.abort(), REQUEST_TIMEOUT);
  try {
    const filename = source === 'dummy' ? 'dummy-data.json' : 'model-output.json';
    const geometryRequest = geometry ? Promise.resolve(geometry) : Promise.all(['texas-load-zones.json', 'texas-counties.json'].map(async name => readJsonResponse(await fetch(`data/${name}`, { signal: operation.signal })))).then(([energy, outages]) => validateGeometry({ energy, outages }));
    const payloadRequest = fetch(`data/${filename}`, { signal: operation.signal, cache: 'no-store' }).then(response => response.status === 404 && source === 'model'
      ? unavailablePayload('No compliant model-output.json has been exported. Historical research data is not a forecast.')
      : readJsonResponse(response));
    const terrainRequest = terrain ? Promise.resolve(terrain) : fetch('data/texas-terrain.json', { signal: operation.signal }).then(readJsonResponse).catch(() => null);
    const [nextGeometry, payload, nextTerrain] = await Promise.all([geometryRequest, payloadRequest, terrainRequest]);
    if (operation.signal.aborted) return;
    geometry = nextGeometry;
    terrain = nextTerrain;
    dispatch({ type: 'RESOLVE', requestId, payload });
  } catch (error) {
    dispatch({ type: 'REJECT', requestId, message: operation.signal.aborted ? 'Forecast loading timed out. Retry or switch data source.' : error.message });
  } finally { clearTimeout(timeout); }
}

function buildMap() {
  if (!geometry || renderedMode === state.mode) return;
  renderedMode = state.mode;
  const layer = geometry[state.mode];
  const map = get('texas-map');
  const [left, top, width, height] = layer.view_box;
  const unit = width / 900;
  const padding = 58 * unit;
  const centerX = left + width / 2;
  const centerY = top + height / 2;
  mapView = { left: left - padding, top: top - padding, width: width + padding * 2, height: height + padding * 2, centerX, centerY };
  zoomLevel = 1;
  updateMapView();
  get('map-object').setAttribute('transform', `translate(${centerX} ${centerY}) rotate(-1.2) skewX(-2) scale(1 .91) translate(${-centerX} ${-centerY})`);
  map.style.setProperty('--terrain-depth', `${31 * unit}px`);
  map.style.setProperty('--region-seam-depth', `${2 * unit}px`);
  map.style.setProperty('--region-side-depth', `${31 * unit}px`);
  map.style.setProperty('--region-pin-lift', `${25 * unit}px`);
  map.style.setProperty('--region-preview-lift', `${29 * unit}px`);
  map.style.setProperty('--map-unit', unit);
  get('survey-grid').setAttribute('patternTransform', `scale(${unit})`);
  const gradients = get('map-gradients');
  gradients.replaceChildren();
  for (let level = 0; level < 5; level++) {
    const gradient = svgElement('linearGradient', { id: `data-fill-${level}`, gradientUnits: 'userSpaceOnUse', x1: left + width * .22, y1: top, x2: left + width * .64, y2: top + height });
    for (let index = 0; index <= 24; index++) gradient.append(svgElement('stop', { offset: index / 24 }));
    gradients.append(gradient);
  }
  paintGradients();
  for (const id of ['terrain-pattern', 'terrain-image']) {
    for (const [attribute, value] of Object.entries({ x: left, y: top, width, height })) get(id).setAttribute(attribute, value);
  }
  get('terrain-image').setAttribute('href', `data/texas-terrain-${state.mode}.webp`);
  for (const id of ['terrain-halo', 'terrain-grid']) {
    for (const [attribute, value] of Object.entries({ cx: left + width * .51, cy: top + height * .56, rx: width * .46, ry: height * .45 })) get(id).setAttribute(attribute, value);
  }
  const group = get('regions');
  const utilityGroup = get('utility-regions');
  group.replaceChildren();
  utilityGroup.replaceChildren();
  get('region-targets').replaceChildren();
  get('utility-targets').replaceChildren();
  get('map-cities').replaceChildren();
  get('map-labels').replaceChildren();
  const outline = layer.outline ?? layer.regions.map(region => region.path).join('');
  get('state-outline').setAttribute('d', outline);
  get('terrain-base').setAttribute('d', outline);
  get('terrain-mid').setAttribute('d', outline);
  get('map-texture').setAttribute('d', outline);
  const options = document.createDocumentFragment();
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = state.mode === 'energy' ? 'Select a load zone' : 'Select a county';
  options.append(placeholder);
  const utilityOrder = { LZ_LCRA: 1, LZ_RAYBN: 2, LZ_CPS: 3, LZ_AEN: 4 };
  const renderRegions = state.mode === 'energy' ? [...layer.regions].sort((first, second) => (utilityOrder[first.id] ?? 0) - (utilityOrder[second.id] ?? 0)) : layer.regions;
  for (const [index, region] of renderRegions.entries()) {
    const utility = state.mode === 'energy' && ['LZ_AEN', 'LZ_CPS', 'LZ_LCRA', 'LZ_RAYBN'].includes(region.id);
    const piece = svgElement('g', { 'data-region-piece': region.id, class: `region-piece${utility ? ' utility-region' : ''}` });
    const path = svgElement('path', { d: region.path, 'data-region': region.id, class: 'region', role: 'button', tabindex: index === 0 ? '0' : '-1', 'aria-label': region.name, 'aria-pressed': 'false' });
    const title = svgElement('title');
    title.textContent = region.name;
    path.append(title);
    piece.append(svgElement('path', { d: region.path, class: 'region-side' }), svgElement('path', { d: region.path, class: 'region-fill unknown' }), svgElement('path', { d: region.path, class: 'region-surface' }));
    (utility ? utilityGroup : group).append(piece);
    get(utility ? 'utility-targets' : 'region-targets').append(path);
    const option = document.createElement('option');
    option.value = region.id;
    option.textContent = state.mode === 'energy' ? `${region.name} · ${region.id}` : `${region.name} County`;
    options.append(option);
    if (state.mode === 'energy') {
      const label = svgElement('text', { x: region.label[0], y: region.label[1], 'text-anchor': 'middle', 'data-label': region.id });
      if (utility) label.classList.add('utility-label');
      label.textContent = ({ LZ_AEN: 'Austin', LZ_CPS: 'CPS', LZ_LCRA: 'LCRA', LZ_RAYBN: 'Rayburn' })[region.id] ?? region.name;
      get('map-labels').append(label);
    }
  }
  const cityGroup = get('map-cities');
  const omittedCities = state.mode === 'energy' ? ['Austin', 'San Antonio', 'Houston', 'Fort Worth', 'Waco', 'Beaumont'] : ['Fort Worth', 'Waco', 'Beaumont'];
  for (const city of terrain?.cities ?? []) {
    const coordinates = city[state.mode];
    if (omittedCities.includes(city.name) || !Array.isArray(coordinates) || coordinates.length !== 2 || !coordinates.every(Number.isFinite)) continue;
    const [cityX, cityY] = coordinates;
    const marker = svgElement('g', { class: 'city-marker' });
    const cityLabel = svgElement('text', { x: cityX + 6 * unit, y: cityY + 4 * unit });
    cityLabel.textContent = city.name;
    marker.append(svgElement('circle', { cx: cityX, cy: cityY, r: 2.3 * unit }), cityLabel);
    const target = [...get('utility-targets').children, ...get('region-targets').children].find(path => path.isPointInFill(new DOMPoint(cityX, cityY)));
    if (target) marker.dataset.cityRegion = target.dataset.region;
    cityGroup.append(marker);
  }
  get('region-select').replaceChildren(options);
}

function regionValue(regionId) {
  if (!state.data) return { value: null, label: 'Coverage unknown' };
  const interval = state.data.timelines[state.mode][state.timeIndex];
  if (!interval) return { value: null, label: 'No forecast for this mode' };
  if (state.mode === 'energy') {
    const row = state.data.priceIndex.get(`${regionId}|${interval.interval_start_utc}`);
    return row?.availability === 'available' ? { value: row.rtm_mean_usd_mwh, label: `${formatMoney(row.rtm_mean_usd_mwh)} per MWh expected RTM` } : { value: null, label: 'Coverage unknown' };
  }
  const row = state.data.outageIndex.get(regionId);
  if (!row || row.coverage === 'unknown') return { value: null, label: 'Coverage unknown' };
  if (row.active_outage) return { value: null, active: true, label: 'County scenario ongoing at issue time' };
  const value = row.p_first_start_by_hour[state.timeIndex];
  return { value, label: `${percent(value)} chance of first county scenario onset in this hour` };
}

function renderMap() {
  buildMap();
  if (!geometry) return;
  const interactive = ['current', 'partial', 'stale'].includes(state.payloadStatus) && state.data.timelines[state.mode].length > 0;
  const activeId = state.previewRegion ?? state.selectedRegion;
  const groups = [get('regions'), get('utility-regions')];
  for (const group of groups) group.classList.toggle('has-active', Boolean(activeId));
  const focused = document.activeElement?.dataset.region;
  for (const piece of groups.flatMap(group => [...group.children])) {
    const path = get('texas-map').querySelector(`[data-region="${piece.dataset.regionPiece}"]`);
    const region = geometry[state.mode].regions.find(item => item.id === path.dataset.region);
    const data = regionValue(region.id);
    const selectionClass = `${region.id === activeId ? ' active' : ''}${region.id === state.selectedRegion ? ' pinned' : ''}`;
    path.setAttribute('class', `region${selectionClass}`);
    piece.querySelector('.region-fill').setAttribute('class', `region-fill ${data.active ? 'active-scenario' : colorClass(state.mode, data.value)}`);
    piece.setAttribute('class', `region-piece${piece.classList.contains('utility-region') ? ' utility-region' : ''}${data.value === null && !data.active ? ' unknown' : ''}${region.id === focused && keyboardMapFocus ? ' focused' : ''}${selectionClass}`);
    path.setAttribute('aria-pressed', String(region.id === state.selectedRegion));
    path.setAttribute('aria-disabled', String(!interactive));
    path.setAttribute('tabindex', interactive && region.id === (focused ?? state.selectedRegion ?? geometry[state.mode].regions[0].id) ? '0' : '-1');
    const label = `${region.name}${state.mode === 'outages' ? ' County' : ''}: ${data.label}${state.payloadStatus === 'stale' ? ', stale forecast' : ''}`;
    path.setAttribute('aria-label', label);
    path.firstElementChild.textContent = label;
  }
  for (const id of [state.selectedRegion, activeId]) {
    const piece = id && get('texas-map').querySelector(`[data-region-piece="${id}"]`);
    if (piece) piece.parentElement.append(piece);
  }
  const previewId = state.previewRegion;
  for (const label of get('map-labels').children) {
    label.classList.toggle('preview', label.dataset.label === previewId);
    label.classList.toggle('pinned', label.dataset.label === state.selectedRegion);
  }
  for (const marker of get('map-cities').children) {
    marker.classList.toggle('preview', marker.dataset.cityRegion === previewId);
    marker.classList.toggle('pinned', marker.dataset.cityRegion === state.selectedRegion);
  }
  get('region-select').disabled = !interactive;
  get('region-select').value = state.selectedRegion ?? '';
  for (const button of get('utility-zones').querySelectorAll('button')) {
    button.disabled = !interactive;
    button.setAttribute('aria-pressed', String(button.dataset.region === state.selectedRegion));
  }
  get('utility-zones').hidden = state.mode !== 'energy';
  get('map-empty-note').hidden = interactive;
  get('map-empty-note').textContent = state.payloadStatus === 'loading' ? 'Loading forecast…' : state.payloadStatus === 'error' ? 'Forecast unavailable' : 'No forecast for this view';
  get('map-title').textContent = state.mode === 'energy' ? 'Energy outlook' : 'County outage outlook';
  get('map-subtitle').textContent = state.mode === 'energy' ? 'Eight ERCOT settlement load zones' : '254 counties · county scenario risk, not home probability';
  get('region-noun').textContent = state.mode === 'energy' ? 'load zone' : 'county';
  get('map-description').textContent = `${state.mode === 'energy' ? 'Schematic ERCOT load zones' : 'Texas county scenario risk'}. Focus previews; Enter or Space pins. Arrow keys move between regions. An equivalent region selector follows the map.`;
  get('geometry-note').textContent = (state.mode === 'energy' ? 'Schematic ERCOT zones; terrain alignment is approximate. Utility footprints are simplified, not an address lookup.' : 'County boundaries do not identify load zones, feeders, or individual homes. Missing coverage is hatched, not zero.') + ' Shaded relief shows land elevation, not forecast intensity. Lift indicates selection.';
  const ramp = RAMPS[state.mode];
  get('legend-title').textContent = interactive ? ramp.metric : 'No quantitative forecast available';
  get('legend-scale').hidden = !interactive;
  get('legend-scale').innerHTML = ramp.labels.map((label, index) => `<span class="legend-stop"><i style="background:linear-gradient(115deg,rgb(${gradientColor(state.mode, index, 0).join(' ')}),rgb(${gradientColor(state.mode, index, 1).join(' ')}))"></i>${label}</span>`).join('');
  get('outside-key').hidden = state.mode !== 'energy';
  get('active-key').hidden = state.mode !== 'outages';
}

function geographyName(mode, id) {
  if (!id) return 'No geographic relationship';
  const region = geometry[mode].regions.find(item => item.id === id);
  return `${region?.name ?? id}${mode === 'outages' ? ' County' : ''}`;
}

function energyBlock(selection) {
  const row = selection.energy;
  const content = row?.availability === 'available'
    ? `<div class="metric-value">${escape(formatMoney(row.rtm_mean_usd_mwh))} <small>USD/MWh</small></div><p class="metric-subtext">Expected RTM · DAM ${escape(formatMoney(row.dam_spp_usd_mwh))}<br>Median (P50) ${escape(formatMoney(row.rtm_p50_usd_mwh))}</p><div class="uncertainty-band"><span>P10 ${escape(formatMoney(row.rtm_p10_usd_mwh))}</span><span>P90 ${escape(formatMoney(row.rtm_p90_usd_mwh))}</span></div><p class="small-note">P10–P90 is an 80% band only if calibrated. Wholesale prices, not a household rate or profit.</p><p class="small-note">${escape(formatTime(selection.energyInterval.interval_start_utc))} – ${escape(formatTime(selection.energyInterval.interval_end_utc))}</p>`
    : `<p class="unavailable">${selection.energyId ? 'Price forecast unavailable for this region and interval.' : 'Energy signal unavailable. No explicit load-zone relationship was supplied.'}</p>`;
  return `<section class="model-block${state.mode === 'energy' ? ' emphasis' : ''}" aria-label="Energy model"><h3>Energy signal</h3><p class="geography">${escape(geographyName('energy', selection.energyId))}${selection.energyId ? ` · ${escape(selection.energyId)}` : ''}</p>${content}</section>`;
}

function outageBlock(selection) {
  const row = selection.outage;
  let content;
  if (!selection.outageId) {
    content = '<p class="unavailable">Outage signal unavailable. No explicit county relationship was supplied.</p>';
  } else if (selection.outageHour < 0) {
    const timeline = state.data.timelines.outages;
    const window = timeline.length ? `${formatTime(timeline[0].interval_start_utc)} to ${formatTime(timeline.at(-1).interval_end_utc)}` : 'No outage forecast window was supplied.';
    content = `<p class="unavailable">Outside the outage forecast window. Available window: ${escape(window)}</p>`;
  } else if (!row || row.coverage === 'unknown') {
    content = '<p class="unavailable">Coverage unknown. Missing data does not mean zero risk.</p>';
  } else if (row.active_outage) {
    const active = row.active_outage;
    content = `<div class="metric-value" style="font-size:21px">Ongoing at issue</div><p class="metric-subtext">County scenario age: ${escape(active.elapsed_minutes)} minutes.</p><p class="small-note">Chance the county scenario remains unresolved for more than:</p><div class="uncertainty-band">${[1, 4, 12, 24].map(hours => `<span>${hours}h<br>${percent(active[`p_remaining_gt_${hours}h`])}</span>`).join('')}</div><p class="small-note">As of ${escape(formatTime(state.data.payload.outage.issued_at_utc))}. County persistence, not a home's restoration time.</p>`;
  } else {
    content = `<div class="metric-value">${percent(row.p_first_start_by_hour[selection.outageHour])} <small>this hour</small></div><p class="metric-subtext">First onset in this hour<br>${percent(row.p_any_next_24h)} any onset in the full 24-hour horizon</p><p class="small-note">Hourly first-onset chances sum to the 24-hour chance. County scenario risk, not a home's outage probability.</p>`;
  }
  return `<section class="model-block${state.mode === 'outages' ? ' emphasis' : ''}" aria-label="Outage model"><h3>County scenario risk</h3><p class="geography">${escape(geographyName('outages', selection.outageId))}${selection.outageId ? ` · FIPS ${escape(selection.outageId)}` : ''}</p>${content}</section>`;
}

function metadata(selection) {
  const payload = state.data.payload;
  const rows = [['Last export', formatTime(payload.generated_at_utc)], ['Rule version', payload.decisions.rule_version]];
  if (payload.outage.status === 'available') rows.push(['County scenario', payload.outage.scenario_definition]);
  for (const [name, run] of [['Price', payload.price], ['Outage', payload.outage]]) {
    if (run.status === 'available') rows.push([`${name} issued`, formatTime(run.issued_at_utc)], [`${name} input cutoff`, formatTime(run.input_cutoff_utc)], [`${name} model`, run.model_version], [`${name} expires after`, `${run.max_age_hours} hours from issue`]);
    else rows.push([`${name} model`, run.reason]);
  }
  if (selection.decision) rows.push(['Rule strength (not probability)', `${selection.decision.strength}`], ['Reserve constraint', selection.decision.reserve_constraint ? 'Active' : 'Not active']);
  return `<details class="metadata"><summary>Issue times, versions &amp; freshness</summary><dl>${rows.map(([label, value]) => `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>`).join('')}</dl></details>`;
}

function renderPanel() {
  const selection = viewSelection(state);
  get('clear-selection').hidden = !state.selectedRegion && !state.previewRegion;
  get('selection-state').textContent = state.previewRegion ? 'Preview · choose to pin' : state.selectedRegion ? 'Pinned region' : 'Explore a region';
  let content;
  if (!selection) {
    const emptyStates = {
      empty: ['Model output<br>is not available.', 'No compliant forecast is available for this view. Export the model contract and reload, or explore the complete dummy scenario.'],
      error: ['This forecast<br>could not load.', 'The data was not accepted. Retry the export or switch to Dummy data to explore.'],
      loading: ['Loading<br>the forecast.', 'The request has an eight-second limit. You can switch sources while it loads.']
    };
    const [heading, copy] = emptyStates[state.payloadStatus] ?? ['Every place has<br>a different signal.', 'Choose a region to see the forecast and the rule behind its battery recommendation.'];
    content = `<div class="panel-empty"><div class="battery-drawing" aria-hidden="true"><i></i><i></i><i></i></div><h2 id="panel-title">${heading}</h2><p>${copy}</p><div class="action-guide"><span>↓ Charge</span><span>Ⅱ Hold reserve</span><span>↑ Discharge</span></div><p class="small-note">Focus or hover to preview. Click, tap, Enter, or Space to pin.</p></div>`;
  } else {
    const { decision } = selection;
    const isDemo = state.data.payload.decisions.is_demo;
    const stale = state.payloadStatus === 'stale';
    const main = state.mode === 'energy' ? selection.energy : selection.outage;
    const unknown = !main || (state.mode === 'energy' ? main.availability !== 'available' : main.coverage === 'unknown');
    const title = unknown ? 'Coverage unknown' : decision ? ACTIONS[decision.action] : 'No recommendation';
    const icons = { charge: '↓', hold: 'Ⅱ', discharge: '↑' };
    const reasons = decision ? decision.reason_codes.map(code => {
      if (!isDemo && ['low_price', 'price_opportunity', 'no_clear_opportunity'].includes(code)) return 'The exported policy determines this price-based action.';
      return REASONS[code];
    }).join(' ') : unknown ? 'There is no trustworthy signal for this region and interval.' : 'A versioned decision was not supplied for this interval. Model values remain available below.';
    const modelBlocks = state.mode === 'energy' ? energyBlock(selection) + outageBlock(selection) : outageBlock(selection) + energyBlock(selection);
    const detailsOpen = mobileLayout.matches ? '' : ' open';
    content = `<div class="decision-heading"><span class="tag">${isDemo ? 'Demo rules' : 'Exported rules'}${stale ? ' · Stale' : ''}</span><p>${escape(geographyName(state.mode, selection.region))}</p>${stale ? '<p class="old-action">Prior recommendation — not current guidance</p>' : ''}<h2 id="panel-title">${decision && !unknown ? `<span class="decision-icon" aria-hidden="true">${icons[decision.action]}</span> ` : ''}${escape(title)}</h2><p>${escape(reasons)}</p><p class="small-note">${escape(formatTime(selection.interval.interval_start_utc))} – ${escape(formatTime(selection.interval.interval_end_utc))}</p></div><details class="signal-details"${detailsOpen}><summary>Model details and freshness</summary><div class="signal-details-body">${modelBlocks}<p class="relationship-note">${escape(selection.relationship?.description ?? 'No cross-geography pairing supplied. Load zones and counties do not nest.')}</p>${metadata(selection)}</div></details>`;
    const announcement = `${geographyName(state.mode, selection.region)}. ${stale ? 'Stale. Prior recommendation: ' : ''}${title}. ${formatTime(selection.interval.interval_start_utc)}.`;
    if (announcement !== lastAnnouncement) {
      get('panel-announcement').textContent = announcement;
      lastAnnouncement = announcement;
    }
  }
  if (content !== lastPanel) {
    const detailsOpen = get('panel-content').querySelector('.metadata')?.open;
    get('panel-content').innerHTML = content;
    if (detailsOpen && get('panel-content').querySelector('.metadata')) get('panel-content').querySelector('.metadata').open = true;
    lastPanel = content;
  }
}

function renderTimeline() {
  const timeline = state.data?.timelines[state.mode] ?? [];
  const interval = timeline[state.timeIndex];
  const interactive = Boolean(interval) && ['current', 'partial', 'stale'].includes(state.payloadStatus);
  get('time-range').disabled = !interactive;
  get('time-range').max = String(Math.max(0, timeline.length - 1));
  get('time-range').value = String(state.timeIndex);
  const label = interval ? `${formatTime(interval.interval_start_utc)} – ${formatTime(interval.interval_end_utc)}` : 'No available interval';
  get('selected-time').textContent = label;
  get('time-range').setAttribute('aria-valuetext', label);
  get('previous').disabled = !interactive || state.timeIndex === 0;
  get('next').disabled = !interactive || state.timeIndex >= timeline.length - 1;
  get('timeline-detail').textContent = state.mode === 'energy' ? `${timeline.length || 'No'} 15-minute intervals · Central Time` : `${timeline.length || 'No'} hourly intervals · Central Time`;
  get('interval-select').disabled = !interactive;
  if (renderedTimeline !== timeline) {
    renderedTimeline = timeline;
    get('interval-select').replaceChildren(...timeline.map((entry, index) => new Option(formatTime(entry.interval_start_utc), String(index))));
    const indices = timeline.length ? [...new Set(Array.from({ length: 5 }, (_, index) => Math.min(timeline.length - 1, Math.round(index * timeline.length / 4))))] : [];
    get('time-ticks').innerHTML = indices.map(index => `<button type="button" data-time="${index}" aria-label="Jump to ${escape(formatTime(timeline[index].interval_start_utc))}">${escape(shortTime.format(new Date(timeline[index].interval_start_utc)))}</button>`).join('');
  }
  get('interval-select').value = String(state.timeIndex);
  get('scrub-time').textContent = interval ? shortTime.format(new Date(interval.interval_start_utc)) : 'No interval';
  get('time-range').parentElement.style.setProperty('--time-progress', `${timeline.length > 1 ? state.timeIndex / (timeline.length - 1) * 100 : 0}%`);
  for (const tick of get('time-ticks').children) {
    tick.disabled = !interactive;
    tick.setAttribute('aria-pressed', String(Number(tick.dataset.time) === state.timeIndex));
  }
}

function render() {
  document.body.dataset.mode = state.mode;
  document.body.dataset.source = state.dataSource;
  document.body.dataset.status = state.payloadStatus;
  document.body.classList.toggle('has-map-selection', Boolean(state.previewRegion ?? state.selectedRegion));
  const pageTitle = state.mode === 'energy' ? 'Electricity Price Prediction by Location' : 'Outage Prediction by Location';
  get('page-title').textContent = pageTitle;
  document.title = `${pageTitle} · Base Power Hackathon`;
  for (const button of document.querySelectorAll('button[data-mode]')) button.setAttribute('aria-pressed', String(button.dataset.mode === state.mode));
  for (const button of document.querySelectorAll('button[data-source]')) button.setAttribute('aria-pressed', String(button.dataset.source === state.dataSource));
  get('model-tools').hidden = state.dataSource !== 'model';
  get('status-badge').textContent = ({ loading: 'Loading', current: 'Current', partial: 'Partial coverage', stale: 'Stale forecast', empty: 'No model output', error: 'Data error' })[state.payloadStatus];
  get('status-badge').dataset.status = state.payloadStatus;
  get('status-badge').hidden = state.payloadStatus === 'partial';
  const messages = {
    loading: 'Loading forecast… You can switch data sources at any time.',
    current: state.dataSource === 'dummy' ? 'Demo scenario. All prices and risk estimates are invented.' : 'Forecast export loaded. Recommendations follow the supplied rule version.',
    partial: `${state.dataSource === 'dummy' ? 'Demo scenario. ' : ''}Partial coverage: hatched regions have no verified signal; some model pairings or decisions are unavailable.`,
    stale: 'Stale forecast. Prior values remain visible for inspection; recommendations are not current guidance.',
    empty: 'Model output unavailable. Add a compliant forecast export, reload, or choose Dummy data. Historical observations are never used as predictions.',
    error: `Forecast rejected: ${state.error ?? 'Unable to load the map data.'}`
  };
  get('status-copy').textContent = messages[state.payloadStatus];
  get('retry').hidden = !['error', 'empty', 'stale'].includes(state.payloadStatus);
  const timeline = state.data?.timelines[state.mode] ?? [];
  const interval = timeline[state.timeIndex];
  get('context-date').textContent = interval ? `${state.dataSource === 'dummy' ? 'Demo · ' : ''}${formatTime(interval.interval_start_utc)}` : 'Texas · Central Time';
  get('context-window').textContent = state.data ? `Last export ${formatTime(state.data.payload.generated_at_utc)}` : 'Waiting for forecast window';
  get('workspace').setAttribute('aria-busy', String(state.payloadStatus === 'loading'));
  renderMap();
  renderPanel();
  renderTimeline();
}

document.querySelectorAll('button[data-source]').forEach(button => button.addEventListener('click', () => loadSource(button.dataset.source)));
document.querySelectorAll('button[data-mode]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.mode === state.mode) return;
  const previousMode = state.mode;
  dispatch({ type: 'MODE', mode: button.dataset.mode });
  animateGradients(previousMode);
}));
get('retry').addEventListener('click', () => loadSource(state.dataSource, snapshot(state)));
get('reload').addEventListener('click', () => loadSource('model', snapshot(state)));
get('region-select').addEventListener('change', event => dispatch(event.target.value ? { type: 'PIN', region: event.target.value } : { type: 'CLEAR' }));
get('clear-selection').addEventListener('click', () => dispatch({ type: 'CLEAR' }));
get('time-range').addEventListener('input', event => dispatch({ type: 'TIME', index: Number(event.target.value) }));
get('interval-select').addEventListener('change', event => dispatch({ type: 'TIME', index: Number(event.target.value) }));
get('time-ticks').addEventListener('click', event => {
  const button = event.target.closest('button[data-time]');
  if (button) dispatch({ type: 'TIME', index: Number(button.dataset.time) });
});
get('time-range').addEventListener('keydown', event => {
  if (event.key === 'PageUp' || event.key === 'PageDown') {
    event.preventDefault();
    dispatch({ type: 'TIME', index: clamp(state.timeIndex + (event.key === 'PageUp' ? 1 : -1) * (state.mode === 'energy' ? 4 : 1), 0, Number(event.target.max)) });
  }
});
get('previous').addEventListener('click', () => dispatch({ type: 'TIME', index: state.timeIndex - 1 }));
get('next').addEventListener('click', () => dispatch({ type: 'TIME', index: state.timeIndex + 1 }));
get('zoom-in').addEventListener('click', () => zoomMap(zoomLevel + .25));
get('zoom-out').addEventListener('click', () => zoomMap(zoomLevel - .25));
get('zoom-reset').addEventListener('click', () => zoomMap(1));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') dispatch({ type: 'CLEAR', region: document.activeElement?.dataset.region });
});
get('utility-zones').addEventListener('click', event => {
  const button = event.target.closest('button[data-region]');
  if (button) dispatch({ type: 'PIN', region: button.dataset.region });
});
get('texas-map').addEventListener('pointerdown', event => {
  pointerFocusPending = true;
  keyboardMapFocus = false;
  didPan = false;
  if (zoomLevel > 1 && event.button === 0) pan = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, centerX: mapView.centerX, centerY: mapView.centerY, matrix: get('texas-map').getScreenCTM().inverse() };
});
get('texas-map').addEventListener('pointermove', event => {
  if (!pan || event.pointerId !== pan.pointerId) return;
  if (!didPan && Math.hypot(event.clientX - pan.x, event.clientY - pan.y) < 6) return;
  if (!didPan) {
    didPan = true;
    get('texas-map').setPointerCapture(event.pointerId);
    get('texas-map').classList.add('is-panning');
    dispatch({ type: 'UNPREVIEW' });
  }
  const start = new DOMPoint(pan.x, pan.y).matrixTransform(pan.matrix);
  const current = new DOMPoint(event.clientX, event.clientY).matrixTransform(pan.matrix);
  mapView.centerX = pan.centerX - (current.x - start.x);
  mapView.centerY = pan.centerY - (current.y - start.y);
  updateMapView();
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) get('texas-map').addEventListener(name, () => {
  pointerFocusPending = false;
  pan = null;
  get('texas-map').classList.remove('is-panning');
});
get('texas-map').addEventListener('pointerover', event => {
  const path = event.target.closest('.region');
  if (path && event.pointerType !== 'touch' && !keyboardMapFocus && !pan) dispatch({ type: 'PREVIEW', region: path.dataset.region });
});
get('workspace').addEventListener('pointerleave', () => dispatch({ type: 'UNPREVIEW' }));
get('texas-map').addEventListener('focusin', event => {
  if (event.target.matches('.region')) {
    keyboardMapFocus = !pointerFocusPending;
    pointerFocusPending = false;
    dispatch({ type: 'PREVIEW', region: event.target.dataset.region });
  }
});
get('texas-map').addEventListener('focusout', event => {
  if (event.target.matches('.region')) {
    keyboardMapFocus = false;
    dispatch({ type: 'UNPREVIEW', region: event.target.dataset.region });
  }
});
get('texas-map').addEventListener('click', event => {
  if (didPan) return;
  const path = event.target.closest('.region');
  if (path) dispatch({ type: 'PIN', region: path.dataset.region });
});
get('texas-map').addEventListener('keydown', event => {
  const path = event.target.closest('.region');
  if (!path) return;
  if (['Enter', ' '].includes(event.key)) {
    event.preventDefault();
    dispatch({ type: 'PIN', region: path.dataset.region });
  }
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
    event.preventDefault();
    const regions = geometry[state.mode].regions;
    const current = regions.findIndex(region => region.id === path.dataset.region);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? regions.length - 1 : (current + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1) + regions.length) % regions.length;
    get('texas-map').querySelector(`[data-region="${regions[index].id}"]`).focus();
  }
});
document.querySelectorAll('a[href^="rules.html"]').forEach(link => link.addEventListener('click', event => {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || link.target === '_blank') return;
  dispatch({ type: 'RULES' });
  try { sessionStorage.setItem(storageKey, JSON.stringify(state.saved)); } catch { }
}));
mobileLayout.addEventListener('change', event => {
  const details = get('panel-content').querySelector('.signal-details');
  if (details) details.open = !event.matches;
});
window.addEventListener('pageshow', event => {
  if (event.persisted) {
    dispatch({ type: 'TICK', now: Date.now() });
    dispatch({ type: 'RETURN' });
  }
});
setInterval(() => dispatch({ type: 'TICK', now: Date.now() }), 60000);
let saved = null;
try { saved = readSnapshot(sessionStorage.getItem(storageKey)); } catch { }
if (saved) state = { ...state, mode: saved.mode };
loadSource(saved?.dataSource ?? 'dummy', saved);
