const button = document.getElementById('pull-forecast');
const status = document.getElementById('forecast-job-status');
const sources = document.getElementById('forecast-source-list');
let version = null;
let polling = false;
let refreshing = false;
let epoch = 0;
let apiAvailable = true;
let pollDelay = 15000;

async function request(path, options = {}) {
  const response = await fetch(path, { ...options, cache: 'no-store', signal: AbortSignal.timeout(10000) });
  if (response.status === 404 || response.status === 405) throw Object.assign(new Error('Live prediction is not connected on this site. Select Dummy data to explore the map.'), { status: response.status });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Forecast service returned ${response.status}.`);
  return data;
}

function render(data) {
  button.disabled = data.running;
  button.textContent = data.running ? 'Updating forecast…' : 'Pull data & predict';
  status.textContent = data.message || 'Ready to pull forecast inputs.';
  pollDelay = data.running ? 1500 : 15000;
  const entries = [...(data.sources || [])];
  if (data.scheduled_for_utc) entries.push({name:'Automatic run', detail:new Date(data.scheduled_for_utc).toLocaleString()});
  if (data.last_success_utc) entries.push({name:'Last export', detail:new Date(data.last_success_utc).toLocaleString()});
  sources.replaceChildren(...entries.map(source => {
    const li = document.createElement('li');
    li.textContent = `${source.name}: ${source.detail}`;
    return li;
  }));
  if (data.export_version && version !== data.export_version) {
    if (version !== null || data.last_success_utc) window.dispatchEvent(new CustomEvent('forecast-ready'));
    version = data.export_version;
  }
}

async function poll() {
  if (polling || refreshing) { window.setTimeout(poll, pollDelay); return; }
  polling = true;
  const current = epoch;
  try { const data = await request('api/forecast/status'); if (current === epoch) { render(data); apiAvailable = true; } }
  catch (error) { if (current === epoch) { status.textContent = error.message; button.disabled = error.status === 404 || error.status === 405; apiAvailable = false; } }
  finally {
    polling = false;
    window.setTimeout(poll, apiAvailable ? pollDelay : 60000);
  }
}

button.addEventListener('click', async () => {
  const current = ++epoch;
  refreshing = true;
  button.disabled = true;
  status.textContent = 'Pulling published prices and weather outlooks…';
  try { const data = await request('api/forecast/refresh', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); if (current === epoch) render(data); }
  catch (error) { if (current === epoch) { status.textContent = error.message; button.disabled = error.status === 404 || error.status === 405; } }
  finally { refreshing = false; }
});
poll();
