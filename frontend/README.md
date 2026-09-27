# Base Power Hackathon frontend

The native HTML/CSS/JavaScript map consumes one versioned JSON bundle. Run the Python forecast service from the repository root; see [the project README](../README.md) for installation, live refresh, offline replay, and model provenance.

From a clean clone, install the pinned runtime and start the runner (Python 3.12+, Node.js 22+, and LightGBM's OpenMP runtime are required):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-forecast.txt
.venv/bin/python forecast_server.py
```

Open `http://127.0.0.1:8099`. The process polls every minute and predicts ten minutes after observing new input versions; the manual button runs immediately. Public reverse proxies must allow the two `/api/forecast/` endpoints and `/data/model-output.json`, and the runner must receive `--allowed-origin` with the public HTTPS origin. Manual pulls are limited to one per minute. For an offline check, stop the runner, use `.venv/bin/python forecast_server.py --replay`, then restart it with `--no-auto`; the genuine historical timestamps remain visibly stale. The checked-in models reproduce frozen inference and dispatch, while full retraining needs the original bulk archives.

- **Model output** is the default. It reads `data/model-output.json`, produced by the real frozen models and simulated battery optimizer. The pull button calls the local service; source status explains delayed publication and the ten-minute automatic run.
- **Dummy data** is a fixed synthetic fixture, evaluated at its own frozen clock. Append `?source=dummy` to open it directly.
- **How dispatch works** uses native diagrams, the observed financial comparison, considered/excluded factors, and future numerical feature tests. Source and map credits remain included.

`core.mjs` validates [the model contract](model-output-contract.md) and owns the pure state reducer. `app.mjs` renders the map and decisions. `forecast-controls.mjs` handles manual acquisition and job status. `validate-export.mjs` applies the same consumer contract before the Python service publishes a bundle.

The map supports keyboard preview/pinning, stable hit targets, time scrubbing, zoom, touch, and reduced motion. Prices remain wholesale USD/MWh. Conditional county scenarios explicitly assume no active outage; unknown counties remain different from zero risk. Four illustrative household locations use documented representative pairings, never inferred address membership. Simulated stored energy and flows appear with the action; previously fixed actions are labeled as history.

```bash
npm ci --prefix frontend
npx --prefix frontend playwright install chromium
npm test --prefix frontend
```

The test server uses explicit in-memory model fixtures and does not depend on any live export. Browser checks save screenshots under ignored `frontend/screenshots/`. `BPC_SCREENSHOT_DIR`, `CHROMIUM_EXECUTABLE`, and `PLAYWRIGHT_MODULE` can select existing test resources. With the real service running, use `BPC_FORECAST_URL=http://127.0.0.1:8099/ node frontend/forecast-browser-check.mjs` for the refresh/API flow.

Static hosting can display a saved export and the dummy scenario. Live pulls and automatic forecasts require the Python service. No battery commands are sent.
