import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { dummy, now, modelPayload } from './test-data.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('.', import.meta.url));
const screenshotDirectory = process.env.BPC_SCREENSHOT_DIR ? resolve(process.env.BPC_SCREENSHOT_DIR) : resolve(root, 'screenshots');
await mkdir(screenshotDirectory, { recursive: true });
const mime = { '.html': 'text/html', '.css': 'text/css', '.mjs': 'text/javascript', '.json': 'application/json', '.webp': 'image/webp', '.png': 'image/png' };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/data/model-output.json') { response.writeHead(404).end('Test server requires an explicit forecast fixture'); return; }
    const filename = resolve(root, `.${pathname.endsWith('/') ? `${pathname}index.html` : pathname}`);
    if (!filename.startsWith(root.endsWith(sep) ? root : `${root}${sep}`)) { response.writeHead(403).end(); return; }
    const body = await readFile(filename);
    response.writeHead(200, { 'Content-Type': mime[extname(filename)] || 'application/octet-stream' }).end(body);
  }
  catch { response.writeHead(404).end('Not found'); }
});
await new Promise(resolveListening => server.listen(0, '127.0.0.1', resolveListening));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
let checks = 0;
const animationFailures = [];
async function checkElevation(page, regionId) {
  const piece = page.locator(`[data-region-piece="${regionId}"]`);
  const target = page.locator(`#texas-map .region[data-region="${regionId}"]`);
  const texture = piece.locator('.region-surface');
  assert.equal(await piece.isVisible(), true);
  await page.waitForFunction(id => {
    const target = document.querySelector(`#texas-map .region[data-region="${id}"]`).getBoundingClientRect();
    const fill = document.querySelector(`[data-region-piece="${id}"] .region-fill`).getBoundingClientRect();
    return target.top - fill.top > 3;
  }, regionId);
  const surface = await piece.evaluate(element => {
    const style = getComputedStyle(element);
    const bounds = element.getBoundingClientRect();
    const map = document.querySelector('#texas-map').getBoundingClientRect();
    return {
      lift: -new DOMMatrixReadOnly(style.transform).m42,
      contained: bounds.left >= map.left && bounds.right <= map.right && bounds.top >= map.top && bounds.bottom <= map.bottom
    };
  });
  assert.ok(surface.lift > 3, `${regionId} must rise, not only cast a shadow`);
  assert.ok(surface.contained, `${regionId} must fit inside the SVG`);
  assert.equal(await texture.evaluate(element => getComputedStyle(element).pointerEvents), 'none');
  assert.equal(await texture.evaluate(element => getComputedStyle(element).fill), 'url("#terrain-pattern")');
  assert.equal(await texture.getAttribute('d'), await target.getAttribute('d'));
  assert.equal(await target.evaluate(element => getComputedStyle(element).transform), 'none');
  assert.equal(await target.evaluate(element => Boolean(element.closest('[data-region-piece]'))), false);
  assert.equal(await piece.locator('.region-fill').getAttribute('d'), await target.getAttribute('d'));
  checks += 9;
  return surface.lift;
}

async function stationaryClick(page, regionId, coordinates) {
  const target = page.locator(`#texas-map .region[data-region="${regionId}"]`);
  await target.scrollIntoViewIfNeeded();
  const before = await target.boundingBox();
  const point = await target.evaluate((element, coordinates) => {
    const bounds = element.getBBox();
    const local = coordinates ? new DOMPoint(...coordinates) : new DOMPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    const screen = local.matrixTransform(element.getScreenCTM());
    return { x: screen.x, y: screen.y, inside: element.isPointInFill(local) };
  }, coordinates);
  assert.equal(point.inside, true, `${regionId} test point must be inside the region`);
  await page.mouse.move(point.x, point.y);
  await page.waitForTimeout(300);
  await checkElevation(page, regionId);
  const after = await target.boundingBox();
  for (const property of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(after[property] - before[property]) < .5, `${regionId} hit target must not move on hover (${property})`);
  assert.equal(await page.evaluate(screenPoint => document.elementFromPoint(screenPoint.x, screenPoint.y)?.getAttribute('data-region'), point), regionId);
  await page.mouse.click(point.x, point.y);
  assert.equal(await page.locator('#region-select').inputValue(), regionId);
  assert.equal(await target.getAttribute('aria-pressed'), 'true');
  checks += 8;
}

async function checkWalls(page) {
  const walls = await page.locator('.region-walls:has(path)').evaluateAll(groups => groups.map(group => {
    const piece = group.closest('.region-piece');
    const fill = piece.querySelector('.region-fill');
    const unit = Number(getComputedStyle(document.querySelector('#texas-map')).getPropertyValue('--map-unit'));
    return {
      raised: piece.classList.contains('active') || piece.classList.contains('pinned'),
      count: group.childElementCount,
      joined: [...group.children].every((path, index) => path.getAttribute('d') === fill.getAttribute('d') && Math.abs(path.transform.baseVal.consolidate().matrix.f - (30 - index) * unit) < .0001),
      embedded: group.firstElementChild.transform.baseVal.consolidate().matrix.f + new DOMMatrixReadOnly(getComputedStyle(piece).transform).m42 > 0,
      pointerEvents: getComputedStyle(group).pointerEvents
    };
  }));
  assert.ok(walls.length > 0 && walls.length <= 2, 'Only active and pinned pieces should have populated walls'); checks++;
  for (const wall of walls) {
    assert.equal(wall.raised, true);
    assert.equal(wall.count, 30);
    assert.equal(wall.joined, true, 'Wall slices must span the entire gap below the matching cap');
    assert.equal(wall.embedded, true, 'Walls must remain embedded in the original map surface');
    assert.equal(wall.pointerEvents, 'none'); checks += 5;
  }
}

async function checkTimeline(page, step) {
  const slider = page.locator('#time-range');
  const saved = await slider.inputValue();
  const maximum = Number(await slider.getAttribute('max'));
  assert.equal(await page.locator('#interval-select option').count(), maximum + 1);
  await page.locator('#interval-select').selectOption('4');
  assert.equal(await slider.inputValue(), '4');
  const selected = await page.locator('#interval-select option:checked').innerText();
  assert.ok((await page.locator('#selected-time').innerText()).includes(selected));
  assert.equal(await slider.getAttribute('aria-valuetext'), await page.locator('#selected-time').innerText());
  const tick = page.locator('#time-ticks button').nth(2);
  const tickIndex = Number(await tick.getAttribute('data-time'));
  await tick.click();
  assert.equal(await slider.inputValue(), String(tickIndex));
  assert.equal(await page.locator('#interval-select').inputValue(), String(tickIndex));
  assert.equal(await tick.getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#scrub-time').innerText(), await tick.innerText());
  await slider.focus();
  await page.keyboard.press('PageUp');
  assert.equal(await slider.inputValue(), String(tickIndex + step));
  assert.equal(await page.locator('#interval-select').inputValue(), String(tickIndex + step));
  await page.keyboard.press('PageDown');
  assert.equal(await slider.inputValue(), String(tickIndex));
  await page.keyboard.press('End');
  await page.keyboard.press('PageUp');
  assert.equal(await slider.inputValue(), String(maximum));
  await page.keyboard.press('Home');
  await page.keyboard.press('PageDown');
  assert.equal(await slider.inputValue(), '0');
  await page.locator('#interval-select').selectOption(saved);
  checks += 13;
}

async function checkModeTransition(page, mode) {
  await page.evaluate(() => {
    window.gradientSamples = [];
    window.pillSamples = [];
    const samplePill = () => {
      window.pillSamples.push(getComputedStyle(document.querySelector('.mode-switch'), '::before').transform);
      window.pillSampleFrame = requestAnimationFrame(samplePill);
    };
    samplePill();
    window.gradientObserver = new MutationObserver(() => window.gradientSamples.push({
      flowing: document.body.classList.contains('mode-flow'),
      colors: [...document.querySelectorAll('#map-gradients stop')].map(stop => stop.getAttribute('stop-color')).join('|')
    }));
    window.gradientObserver.observe(document.querySelector('#map-gradients'), { attributes: true, childList: true, subtree: true });
  });
  const started = performance.now();
  await page.locator(`button[data-mode="${mode}"]`).click();
  await page.waitForFunction(() => !document.body.classList.contains('mode-flow'));
  const samples = await page.evaluate(() => {
    window.gradientObserver.disconnect();
    return window.gradientSamples;
  });
  const pillSamples = await page.evaluate(() => { cancelAnimationFrame(window.pillSampleFrame); return window.pillSamples; });
  try {
    assert.ok(samples.some(sample => sample.flowing), 'The mode transition must animate');
    assert.ok(new Set(samples.map(sample => sample.colors)).size >= 3, 'The actual gradient must include intermediate colors, not teleport between modes');
    assert.ok(new Set(pillSamples).size > 2, 'The mode toggle pill must slide through intermediate CSS transforms');
  } catch (error) {
    animationFailures.push(`${mode} at ${await page.locator('#zoom-reset').innerText()}: ${error.message} (${samples.length} observed frames)`);
    console.error(animationFailures.at(-1));
  }
  assert.equal(await page.locator('.map-stage').evaluate(element => getComputedStyle(element, '::after').content), 'none');
  console.log(`${mode} gradient transition: ${Math.round(performance.now() - started)}ms browser round trip, ${samples.length} gradient frames observed.`);
  checks += 4;
}
try {
  browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}) });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.clock.setFixedTime(new Date(now));
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const metadataPage = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await metadataPage.clock.setFixedTime(new Date(now));
  metadataPage.on('pageerror', error => errors.push(error.message));
  let metadataRequests = 0;
  await metadataPage.route('**/data/texas-terrain.json', () => { metadataRequests++; });
  const metadataStarted = performance.now();
  await metadataPage.goto(`${base}/?source=dummy`, { waitUntil: 'domcontentloaded' });
  await metadataPage.waitForFunction(() => document.body.dataset.status === 'partial', null, { timeout: 3500 });
  const metadataReadyMs = performance.now() - metadataStarted;
  assert.equal(metadataRequests, 1);
  assert.ok(metadataReadyMs < 3500, `A hung optional metadata request must not block forecasts (${Math.round(metadataReadyMs)}ms)`);
  assert.equal(await metadataPage.locator('#time-range').isEnabled(), true);
  assert.equal(await metadataPage.locator('#region-select').isEnabled(), true);
  assert.equal(await metadataPage.locator('#texas-map .region').count(), 8); checks += 5;
  console.log(`Hung terrain metadata: forecasts ready in ${Math.round(metadataReadyMs)}ms.`);
  await metadataPage.close();
  await page.bringToFront();
  await page.goto(`${base}/?source=dummy`);
  await page.waitForFunction(() => document.body.dataset.status === 'partial');
  assert.equal(await page.locator('.wordmark').innerText(), 'Base Power Hackathon');
  assert.equal(await page.locator('#page-title').innerText(), 'Electricity Price Prediction by Location');
  assert.equal(await page.locator('#page-title + p').count(), 0);
  assert.equal(await page.locator('#texture').count(), 0);
  assert.equal(await page.locator('#source-caption').count(), 0);
  await page.waitForFunction(() => document.querySelector('#forecast-job-status').textContent.includes('Live prediction is not connected'));
  assert.match(await page.locator('#forecast-job-status').textContent(), /Live prediction is not connected/);
  assert.equal(await page.locator('#pull-forecast').isDisabled(), true); checks += 2;
  assert.equal(await page.locator('#selection-state').isHidden(), true); checks++;
  assert.equal(await page.locator('.timeline').evaluate(element => Boolean(element.compareDocumentPosition(document.querySelector('#workspace')) & Node.DOCUMENT_POSITION_FOLLOWING)), true);
  assert.equal(await page.locator('#status-badge').isHidden(), true); checks += 6;
  assert.equal(await page.locator('#texas-map .region').count(), 8); checks++;
  assert.equal(await page.locator('#texas-map .region-surface').count(), 8);
  assert.equal(await page.locator('#texas-map .region-side').count(), 8);
  assert.equal(await page.locator('#utility-zones button').count(), 8);
  assert.equal(await page.locator('#map-labels text').count(), 8);
  assert.equal(await page.locator('#map-labels .utility-label').count(), 4);
  assert.equal(await page.locator('#terrain-glow stop').count(), 3);
  assert.equal(await page.locator('#terrain-halo').evaluate(element => getComputedStyle(element).fill), 'url("#terrain-glow")');
  assert.equal(await page.locator('#map-texture').isVisible(), true);
  assert.equal(await page.locator('#terrain-base').getAttribute('d'), await page.locator('#state-outline').getAttribute('d'));
  assert.equal(await page.locator('#terrain-mid').getAttribute('d'), await page.locator('#state-outline').getAttribute('d'));
  assert.equal(await page.locator('#texas-map .region-surface[role], #texas-map .region-surface[tabindex], #texas-map .region-side[role], #texas-map .region-side[tabindex]').count(), 0); checks += 10;
  assert.equal(await page.locator('#texas-map .region-surface').first().evaluate(element => getComputedStyle(element).fill), 'url("#terrain-pattern")'); checks++;
  assert.notEqual(await page.locator('#map-atmosphere').evaluate(element => getComputedStyle(element).backgroundImage), 'none'); checks++;
  assert.equal(await page.locator('.map-stage').evaluate(element => getComputedStyle(element, '::after').content), 'none');
  const terrain = await page.locator('#terrain-image').evaluate(async element => {
    const image = new Image();
    image.src = element.getAttribute('href');
    await image.decode();
    return { width: image.naturalWidth, height: image.naturalHeight };
  });
  assert.ok(terrain.width >= 1600 && terrain.height >= 1200, 'Terrain must load a detailed raster, not repeating linework');
  assert.equal(await page.locator('#regions .region, #utility-regions .region').count(), 0);
  assert.equal(await page.locator('#region-targets .region').count(), 4);
  assert.equal(await page.locator('#utility-targets .region').count(), 4); checks += 5;
  await page.screenshot({ path: `${screenshotDirectory}/energy-desktop.png`, fullPage: true });
  const houston = page.locator('#region-targets [data-region="LZ_HOUSTON"]');
  await houston.hover();
  await checkElevation(page, 'LZ_HOUSTON');
  assert.notEqual(await page.locator('[data-region-piece="LZ_HOUSTON"]').evaluate(element => getComputedStyle(element).transform), 'none', 'Hover must lift the region'); checks++;
  assert.equal(await houston.evaluate(element => element.classList.contains('active')), true); checks++;
  assert.match(await page.locator('#panel-content').innerText(), /Houston/); checks++;
  await page.screenshot({ path: `${screenshotDirectory}/energy-hover-desktop.png`, fullPage: true });
  assert.match(await page.locator('#selection-state').innerText(), /Preview/);
  assert.equal(await houston.getAttribute('aria-pressed'), 'false');
  await page.locator('.decision-panel').hover();
  assert.match(await page.locator('#selection-state').innerText(), /Preview/);
  assert.match(await page.locator('#panel-content').innerText(), /Harris County/); checks += 2;
  await houston.hover();
  await houston.click();
  await page.locator('#region-targets [data-region="LZ_WEST"]').hover();
  assert.match(await page.locator('#panel-content').innerText(), /West/);
  await page.locator('#page-title').hover();
  assert.match(await page.locator('#panel-content').innerText(), /Harris County/);
  await checkElevation(page, 'LZ_HOUSTON');
  assert.equal(await page.locator('[data-region-piece="LZ_HOUSTON"]').evaluate(element => element.classList.contains('pinned')), true); checks++;
  await page.keyboard.press('Escape'); checks += 4;
  await houston.focus();
  await checkElevation(page, 'LZ_HOUSTON');
  assert.match(await page.locator('#selection-state').innerText(), /Preview/);
  assert.equal(await houston.getAttribute('aria-pressed'), 'false');
  await page.keyboard.press('Enter');
  assert.equal(await houston.getAttribute('aria-pressed'), 'true');
  assert.match(await page.locator('#panel-title').innerText(), /Hold reserve/);
  assert.match(await page.locator('#panel-content').innerText(), /Harris County/); checks += 5;
  assert.equal(await page.locator('body').evaluate(element => element.classList.contains('has-map-selection')), true);
  assert.notEqual(await page.locator('.model-block.emphasis').evaluate(element => getComputedStyle(element).boxShadow), 'none'); checks += 2;
  assert.match(await page.locator('#panel-content').innerText(), /Demo rules/); checks++;
  await page.locator('#region-targets [data-region="LZ_WEST"]').focus();
  assert.match(await page.locator('#selection-state').innerText(), /Preview/);
  await page.locator('#region-select').focus();
  assert.match(await page.locator('#panel-content').innerText(), /Harris County/);
  await page.locator('#region-targets [data-region="LZ_WEST"]').focus();
  await page.keyboard.press(' ');
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_WEST');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#region-select').inputValue(), ''); checks += 4;
  await houston.focus();
  await page.keyboard.press('Home');
  assert.equal(await page.locator('#texas-map .region:focus').getAttribute('data-region'), 'LZ_WEST');
  await page.keyboard.press('End');
  assert.equal(await page.locator('#texas-map .region:focus').getAttribute('data-region'), 'LZ_RAYBN');
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#texas-map .region:focus').getAttribute('data-region'), 'LZ_WEST');
  await page.locator('#region-select').focus(); checks += 3;
  const utilityPoints = { LZ_LCRA: [1106, 825], LZ_RAYBN: [1250, 455], LZ_CPS: [985, 995], LZ_AEN: [1090, 875] };
  for (const utility of Object.keys(utilityPoints)) {
    await page.locator('#region-select').selectOption('LZ_SOUTH');
    await stationaryClick(page, utility, utilityPoints[utility]);
  }
  await page.waitForTimeout(300);
  const austinCap = await page.locator('[data-region-piece="LZ_AEN"] .region-fill').evaluate(element => {
    const local = new DOMPoint(1090, 875);
    const screen = local.matrixTransform(element.getScreenCTM());
    return { x: screen.x, y: screen.y, inside: element.isPointInFill(local) };
  });
  assert.equal(austinCap.inside, true);
  await page.mouse.move(austinCap.x, austinCap.y);
  await page.waitForTimeout(300);
  assert.match(await page.locator('#panel-content').innerText(), /Austin/);
  await page.mouse.click(austinCap.x, austinCap.y);
  await page.screenshot({ path: `${screenshotDirectory}/energy-austin-raised-cap.png`, fullPage: true });
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_AEN', 'Clicking the visible raised Austin cap must not select the fixed South footprint beneath it');
  assert.equal(await page.locator('#utility-targets [data-region="LZ_AEN"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#texas-map .region:focus').getAttribute('data-region'), 'LZ_AEN');
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_AEN', 'Keyboard activation after a cap click must keep Austin pinned'); checks += 2;
  await houston.hover();
  assert.match(await page.locator('#panel-content').innerText(), /Houston/);
  assert.match(await page.locator('#selection-state').innerText(), /Preview/);
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_AEN', 'Leaving the raised cap must preview another region without changing the pin'); checks += 7;
  await checkWalls(page);
  await page.locator('#region-select').selectOption('LZ_HOUSTON');
  const originalView = await page.locator('#texas-map').getAttribute('viewBox');
  assert.equal(await page.locator('#zoom-reset').innerText(), '100%');
  assert.equal(await page.locator('#zoom-out').isDisabled(), true);
  await page.evaluate(() => {
    window.zoomSamples = [];
    const sample = () => {
      const map = document.querySelector('#texas-map');
      if (map.classList.contains('is-zooming')) {
        const bounds = document.querySelector('#region-targets [data-region="LZ_HOUSTON"]').getBoundingClientRect();
        window.zoomSamples.push({
          zoom: document.querySelector('#zoom-reset').textContent,
          transform: getComputedStyle(map).transform,
          bounds: [bounds.x, bounds.y, bounds.width, bounds.height].map(value => value.toFixed(3)).join('|')
        });
      }
      window.zoomSampleFrame = requestAnimationFrame(sample);
    };
    window.zoomSampleFrame = requestAnimationFrame(sample);
  });
  for (let zoom = 125; zoom <= 250; zoom += 25) {
    await page.locator('#zoom-in').click();
    assert.equal(await page.locator('#zoom-reset').innerText(), `${zoom}%`);
    await page.waitForFunction(() => !document.querySelector('#texas-map').classList.contains('is-zooming')); checks++;
  }
  const zoomSamples = await page.evaluate(() => { cancelAnimationFrame(window.zoomSampleFrame); return window.zoomSamples; });
  assert.ok(new Set(zoomSamples.map(sample => sample.transform)).size > 6, 'Zoom must animate intermediate transforms, not teleport at each click');
  assert.ok(zoomSamples.some((sample, index) => index > 0 && sample.zoom === zoomSamples[index - 1].zoom && sample.bounds !== zoomSamples[index - 1].bounds), 'A fixed geographic target must move onscreen during the zoom animation');
  assert.equal(await page.locator('#texas-map').evaluate(element => getComputedStyle(element).transform), 'none'); checks += 3;
  assert.equal(await page.locator('#zoom-in').isDisabled(), true);
  const zoomedView = (await page.locator('#texas-map').getAttribute('viewBox')).split(' ').map(Number);
  const initialView = originalView.split(' ').map(Number);
  assert.ok(Math.abs(zoomedView[2] - initialView[2] / 2.5) < .001);
  assert.ok(Math.abs(zoomedView[3] - initialView[3] / 2.5) < .001);
  await page.locator('#zoom-out').click();
  assert.equal(await page.locator('#zoom-reset').innerText(), '225%');
  await page.waitForFunction(() => !document.querySelector('#texas-map').classList.contains('is-zooming'));
  await page.locator('#texas-map').scrollIntoViewIfNeeded();
  const dragPoint = await page.locator('#texas-map').evaluate(element => {
    const bounds = element.getBoundingClientRect();
    for (const vertical of [.5, .35, .65]) for (const horizontal of [.5, .35, .65]) {
      const point = { x: bounds.x + bounds.width * horizontal, y: bounds.y + bounds.height * vertical };
      const region = document.elementFromPoint(point.x, point.y)?.getAttribute('data-region');
      if (region && region !== 'LZ_HOUSTON') return point;
    }
    return null;
  });
  assert.ok(dragPoint, 'Pan must start over a clickable region other than the pinned region');
  const beforePan = await page.locator('#texas-map').getAttribute('viewBox');
  await page.mouse.move(dragPoint.x, dragPoint.y);
  await page.mouse.down();
  await page.mouse.move(dragPoint.x - 75, dragPoint.y + 25, { steps: 8 });
  await page.mouse.up();
  assert.notEqual(await page.locator('#texas-map').getAttribute('viewBox'), beforePan);
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_HOUSTON', 'Dragging must not pin the region under the pointer');
  assert.equal(await page.locator('#texas-map .region[aria-pressed="true"]').count(), 1);
  assert.equal(await page.locator('#texas-map').evaluate(element => element.classList.contains('is-panning')), false);
  await page.screenshot({ path: `${screenshotDirectory}/energy-zoom-pan-desktop.png`, fullPage: true });
  for (let step = 0; step < 3; step++) {
    await page.locator('#zoom-out').click();
    await page.waitForFunction(() => !document.querySelector('#texas-map').classList.contains('is-zooming'));
  }
  assert.equal(await page.locator('#zoom-reset').innerText(), '150%');
  const energyZoomRatio = await page.locator('#texas-map').evaluate(element => element.viewBox.baseVal.width / Number(document.querySelector('#terrain-pattern').getAttribute('width')));
  await checkModeTransition(page, 'outages');
  assert.equal(await page.locator('#zoom-reset').innerText(), '150%');
  const outageZoomRatio = await page.locator('#texas-map').evaluate(element => element.viewBox.baseVal.width / Number(document.querySelector('#terrain-pattern').getAttribute('width')));
  assert.ok(Math.abs(outageZoomRatio - energyZoomRatio) < .001, 'Mode switch must retain actual zoom, not only the zoom label');
  await page.locator('#zoom-out').click();
  await page.waitForFunction(() => !document.querySelector('#texas-map').classList.contains('is-zooming'));
  assert.equal(await page.locator('#zoom-reset').innerText(), '125%');
  const zoomBeforeEnergy = await page.locator('#texas-map').evaluate(element => element.viewBox.baseVal.width / Number(document.querySelector('#terrain-pattern').getAttribute('width')));
  await checkModeTransition(page, 'energy');
  assert.equal(await page.locator('#zoom-reset').innerText(), '125%');
  const zoomAfterEnergy = await page.locator('#texas-map').evaluate(element => element.viewBox.baseVal.width / Number(document.querySelector('#terrain-pattern').getAttribute('width')));
  assert.ok(Math.abs(zoomAfterEnergy - zoomBeforeEnergy) < .001, 'Returning to energy must preserve zoom'); checks += 6;
  await page.locator('#zoom-reset').click();
  assert.equal(await page.locator('#zoom-reset').innerText(), '100%');
  await page.waitForFunction(() => !document.querySelector('#texas-map').classList.contains('is-zooming'));
  assert.equal(await page.locator('#texas-map').getAttribute('viewBox'), originalView);
  assert.equal(await page.locator('#zoom-out').isDisabled(), true);
  assert.equal(await page.locator('#zoom-in').isEnabled(), true); checks += 15;
  await page.locator('#region-select').selectOption('LZ_HOUSTON');
  await page.locator('#next').click();
  const savedTime = await page.locator('#time-range').inputValue();
  await page.locator('.rules-link').click();
  assert.match(await page.title(), /How dispatch works/);
  await page.screenshot({ path: `${screenshotDirectory}/rules-desktop.png`, fullPage: true });
  await page.locator('.back-map').click();
  await page.waitForFunction(() => document.body.dataset.status === 'partial');
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_HOUSTON');
  assert.equal(await page.locator('#time-range').inputValue(), savedTime); checks += 3;
  const [rulesTab] = await Promise.all([
    page.context().waitForEvent('page'),
    page.locator('.rules-link').click({ modifiers: ['Control'] })
  ]);
  await rulesTab.waitForLoadState('domcontentloaded');
  await rulesTab.close();
  await page.bringToFront();
  await checkModeTransition(page, 'outages');
  assert.equal(await page.locator('body').getAttribute('data-mode'), 'outages');
  assert.equal(await page.locator('#page-title').innerText(), 'Outage Prediction by Location');
  await checkModeTransition(page, 'energy'); checks++;
  await page.locator('#region-select').selectOption('LZ_HOUSTON');
  await page.waitForFunction(() => !document.body.classList.contains('mode-flow'));
  await page.screenshot({ path: `${screenshotDirectory}/energy-selected-desktop.png`, fullPage: true });
  await page.locator('button[data-mode="outages"]').click();
  assert.equal(await page.locator('#page-title').innerText(), 'Outage Prediction by Location'); checks += 3;
  assert.equal(await page.locator('#region-targets .region').count(), 254);
  assert.match(await page.locator('#map-subtitle').innerText(), /county scenario risk, not home probability/);
  assert.equal(await page.locator('#time-range').inputValue(), '15');
  assert.equal(await page.locator('#region-select').inputValue(), '');
  await checkTimeline(page, 1);
  await stationaryClick(page, '48003');
  await page.locator('#clear-selection').click();
  const anderson = page.locator('#region-targets [data-region="48001"]');
  await anderson.hover();
  await checkElevation(page, '48001');
  assert.equal(await page.locator('#panel-title').innerText(), 'Coverage unknown');
  const unavailableBackground = await page.locator('.decision-heading[data-action="unavailable"]').evaluate(element => getComputedStyle(element).backgroundColor);
  assert.notEqual(await page.locator('[data-region-piece="48001"]').evaluate(element => getComputedStyle(element).transform), 'none');
  await page.locator('#region-targets [data-region="48003"]').focus();
  await checkElevation(page, '48003');
  assert.match(await page.locator('#panel-content').innerText(), /Andrews County/);
  await page.keyboard.press(' ');
  await page.locator('#region-select').focus();
  await checkElevation(page, '48003'); checks += 3;
  await page.locator('#region-select').selectOption('48453');
  assert.match(await page.locator('#panel-content').innerText(), /0%/);
  await page.locator('#region-select').selectOption('48001');
  assert.equal(await page.locator('#panel-title').innerText(), 'Coverage unknown');
  await page.locator('#region-select').selectOption('48167');
  assert.match(await page.locator('#panel-content').innerText(), /Ongoing at issue/);
  assert.match(await page.locator('#panel-content').innerText(), /not a home's restoration time/); checks += 8;
  await page.locator('#region-select').selectOption('48201');
  await checkElevation(page, '48201');
  await page.screenshot({ path: `${screenshotDirectory}/outages-desktop.png`, fullPage: true });
  await page.locator('#region-targets [data-region="48453"]').focus();
  await checkElevation(page, '48453');
  await checkWalls(page);
  await page.screenshot({ path: `${screenshotDirectory}/outages-focus-desktop.png`, fullPage: true });
  assert.equal(await page.locator('#map-texture').isVisible(), false);
  assert.equal(await page.locator('#map-relief').isVisible(), true);
  assert.equal(await page.locator('#map-relief').evaluate(element => getComputedStyle(element).pointerEvents), 'none');
  assert.equal(await page.locator('#map-relief').evaluate(element => getComputedStyle(element).fill), 'url("#terrain-pattern")');
  assert.ok(await page.locator('#relief-clip-path').getAttribute('d'));
  assert.equal(await page.locator('#utility-regions [data-region-piece="48453"] .region-surface').isVisible(), true);
  assert.equal(await page.locator('#utility-regions [data-region-piece="48201"] .region-surface').isVisible(), true);
  assert.equal(await page.locator('#regions .region-piece:not(.unknown) .region-surface').first().isVisible(), false); checks += 7;
  assert.match(await page.locator('[data-region-piece="48001"] .region-fill').evaluate(element => getComputedStyle(element).fill), /missing/);
  assert.equal(await page.locator('#terrain-base').isVisible(), true); checks += 3;
  await page.route('**/data/model-output.json', route => route.fulfill({ status: 404, body: 'No exported forecast' }));
  await page.locator('button[data-source="model"]').click();
  await page.waitForFunction(() => document.body.dataset.status === 'empty');
  assert.match(await page.locator('#status-copy').innerText(), /No forecast is ready for this view/);
  assert.equal(await page.locator('#region-select').isDisabled(), true);
  assert.equal(await page.locator('#interval-select').isDisabled(), true);
  assert.equal(await page.locator('#time-ticks button:not(:disabled)').count(), 0); checks += 2;
  await page.locator('#retry').click();
  await page.waitForFunction(() => document.body.dataset.status === 'empty');
  await page.screenshot({ path: `${screenshotDirectory}/model-unavailable.png`, fullPage: true }); checks += 3;
  await page.unroute('**/data/model-output.json');
  const model = structuredClone(dummy);
  model.kind = 'model';
  model.decisions.records.forEach(row => { row.relationship = null; });
  const bad = structuredClone(model);
  bad.price.records[0].rtm_p10_usd_mwh = bad.price.records[0].rtm_p90_usd_mwh + 1;
  await page.route('**/data/model-output.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(bad) }));
  await page.locator('#retry').click();
  await page.waitForFunction(() => document.body.dataset.status === 'error');
  assert.match(await page.locator('#status-copy').innerText(), /quantiles/); checks++;
  await page.unroute('**/data/model-output.json');
  model.price.max_age_hours = 0.01;
  await page.route('**/data/model-output.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(model) }));
  await page.locator('#retry').click();
  await page.waitForFunction(() => document.body.dataset.status === 'stale');
  await page.locator('#region-select').selectOption('48201');
  assert.match(await page.locator('#panel-content').innerText(), /Prior recommendation/);
  await page.screenshot({ path: `${screenshotDirectory}/model-stale.png`, fullPage: true }); checks++;
  const setModel = async payload => {
    await page.unroute('**/data/model-output.json');
    await page.route('**/data/model-output.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) }));
    await page.locator('#reload').click();
  };
  const current = modelPayload();
  await setModel(current);
  await page.waitForFunction(() => document.body.dataset.status === 'current');
  await page.locator('button[data-mode="energy"]').click();
  const priceIntervals = [...new Set(current.price.records.map(row => row.interval_start_utc))].sort();
  const decisionBackgrounds = [unavailableBackground];
  const decisionInks = [];
  const decisionIcons = [];
  for (const action of ['charge', 'hold', 'discharge']) {
    const decision = current.decisions.records.find(row => row.mode === 'energy' && row.action === action);
    await page.locator('#interval-select').selectOption(String(priceIntervals.indexOf(decision.interval_start_utc)));
    await page.locator('#region-select').selectOption(decision.region_id);
    assert.equal(await page.locator('.decision-heading').getAttribute('data-action'), action);
    assert.ok((await page.locator('#panel-title').innerText()).includes({ charge: 'Charge', hold: 'Hold reserve', discharge: 'Discharge' }[action]));
    decisionBackgrounds.push(await page.locator('.decision-heading').evaluate(element => getComputedStyle(element).backgroundColor));
    decisionInks.push(await page.locator('#panel-title').evaluate(element => getComputedStyle(element).color));
    decisionIcons.push(await page.locator('.decision-icon').innerText()); checks += 2;
  }
  assert.equal(new Set(decisionBackgrounds).size, 4, 'Charge, hold, discharge and unavailable must have distinct backgrounds');
  assert.equal(new Set(decisionInks).size, 3);
  assert.equal(new Set(decisionIcons).size, 3); checks += 3;
  await page.locator('#interval-select').selectOption('0');
  await page.locator('#region-select').selectOption('LZ_WEST');
  assert.equal(await page.locator('#time-range').inputValue(), '0');
  assert.equal(await page.locator('#previous').isDisabled(), true);
  assert.match(await page.locator('#panel-content').innerText(), /-\$/);
  await checkTimeline(page, 4);
  const selectedInterval = await page.locator('#interval-select option:checked').innerText();
  assert.ok((await page.locator('#panel-content').innerText()).includes(selectedInterval), 'Timeline selection must update the forecast panel'); checks++;
  await page.locator('#time-range').focus();
  await page.keyboard.press('End');
  assert.equal(await page.locator('#time-range').inputValue(), '95');
  assert.equal(await page.locator('#next').isDisabled(), true);
  await page.locator('#previous').click();
  assert.equal(await page.locator('#time-range').inputValue(), '94'); checks += 6;
  await page.locator('.rules-link').click();
  await page.locator('.back-map').click();
  await page.waitForFunction(() => document.body.dataset.status === 'current');
  assert.equal(await page.locator('body').getAttribute('data-source'), 'model');
  assert.equal(await page.locator('#region-select').inputValue(), 'LZ_WEST');
  assert.equal(await page.locator('#time-range').inputValue(), '94'); checks += 3;
  const partial = structuredClone(current);
  partial.price = { status: 'unavailable', reason: 'Price model has not issued a run.' };
  partial.decisions.records = partial.decisions.records.filter(row => row.mode === 'outages');
  await setModel(partial);
  await page.waitForFunction(() => document.body.dataset.status === 'partial');
  assert.equal(await page.locator('#region-select').isDisabled(), true);
  assert.equal(await page.locator('#time-range').isDisabled(), true);
  assert.equal(await page.locator('#interval-select').isDisabled(), true);
  assert.equal(await page.locator('#time-ticks button:not(:disabled)').count(), 0); checks += 2;
  assert.equal(await page.locator('#legend-scale').isVisible(), false);
  await page.locator('button[data-mode="outages"]').click();
  await page.locator('#region-select').selectOption('48453');
  assert.match(await page.locator('#panel-content').innerText(), /0%/);
  assert.match(await page.locator('#panel-content').innerText(), /No explicit load-zone relationship/); checks += 5;
  for (const [invalid, message] of [
    [{ meta: {}, zones: {} }, /schema_version/],
    [dummy, /source.*kind/],
    [{ ...current, decisions: { ...current.decisions, records: [{ ...current.decisions.records[0], action: 'sell' }] } }, /action/],
    [{ ...current, price: { ...current.price, records: [{ ...current.price.records[0], settlement_point: 'LZEW' }, ...current.price.records.slice(1)] } }, /settlement_point/]
  ]) {
    await setModel(invalid);
    await page.waitForFunction(() => document.body.dataset.status === 'error');
    assert.match(await page.locator('#status-copy').innerText(), message);
    assert.equal(await page.locator('#region-select').isDisabled(), true);
    assert.equal(await page.locator('#legend-scale').isVisible(), false); checks += 3;
  }
  await page.unroute('**/data/model-output.json');
  await page.route('**/data/model-output.json', route => route.fulfill({ status: 200, body: '{invalid JSON' }));
  await page.locator('#retry').click();
  await page.waitForFunction(() => document.body.dataset.status === 'error');
  assert.match(await page.locator('#status-copy').innerText(), /not valid JSON/); checks++;
  let releaseRequest;
  await page.unroute('**/data/model-output.json');
  await page.route('**/data/model-output.json', async route => {
    await new Promise(done => { releaseRequest = done; });
    await route.fulfill({ status: 503, body: 'Unavailable' }).catch(() => {});
  });
  await Promise.all([page.waitForRequest('**/data/model-output.json'), page.locator('#retry').click()]);
  assert.equal(await page.locator('body').getAttribute('data-status'), 'loading');
  assert.equal(await page.locator('#region-select').isDisabled(), true);
  assert.equal(await page.locator('#time-range').isDisabled(), true);
  assert.equal(await page.locator('#workspace').getAttribute('aria-busy'), 'true');
  await page.locator('button[data-mode="energy"]').click();
  releaseRequest();
  await page.waitForFunction(() => document.body.dataset.status === 'error');
  assert.match(await page.locator('#status-copy').innerText(), /HTTP 503/); checks += 5;
  await setModel(current);
  await page.waitForFunction(() => document.body.dataset.status === 'current');
  await page.locator('#region-select').selectOption('LZ_HOUSTON');
  assert.equal(await page.locator('#selection-state').innerText(), 'Pinned region'); checks++;
  await page.unroute('**/data/model-output.json');
  await page.route('**/data/model-output.json', () => {});
  await page.locator('#reload').click();
  await page.waitForFunction(() => document.body.dataset.status === 'error', null, { timeout: 12000 });
  assert.match(await page.locator('#status-copy').innerText(), /timed out/); checks++;
  await page.unroute('**/data/model-output.json');
  await page.route('**/data/model-output.json', async route => { await new Promise(done => setTimeout(done, 400)); await route.fulfill({ status: 500, body: 'Failure' }).catch(() => {}); });
  await page.locator('#reload').click();
  await page.locator('button[data-source="dummy"]').click();
  await page.waitForFunction(() => document.body.dataset.status === 'partial');
  await page.waitForTimeout(500);
  assert.equal(await page.locator('body').getAttribute('data-source'), 'dummy');
  assert.equal(await page.locator('body').getAttribute('data-status'), 'partial'); checks += 2;
  await page.unroute('**/data/model-output.json');
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 320, height: 740 }]) {
    await page.setViewportSize(viewport);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Overflow at ${viewport.width}`);
    await page.locator('button[data-mode="energy"]').click();
    await page.locator('#region-select').selectOption('LZ_HOUSTON');
    await checkElevation(page, 'LZ_HOUSTON');
    if (viewport.width <= 760) {
      assert.equal(await page.locator('.signal-details').getAttribute('open'), null);
      assert.equal(await page.locator('.signal-details > summary').isVisible(), true);
      checks += 2;
    }
    if (viewport.width === 390) {
      await page.setViewportSize({ width: 1024, height: 768 });
      await page.waitForFunction(() => document.querySelector('.signal-details')?.open);
      assert.equal(await page.locator('.signal-details').getAttribute('open'), '');
      assert.equal(await page.locator('.signal-details > summary').isVisible(), false);
      await page.setViewportSize(viewport);
      await page.waitForFunction(() => !document.querySelector('.signal-details')?.open);
      assert.equal(await page.locator('.signal-details').getAttribute('open'), null);
      checks += 3;
    }
    await page.waitForFunction(() => !document.body.classList.contains('mode-flow'));
    await page.screenshot({ path: `${screenshotDirectory}/energy-${viewport.width}.png`, fullPage: true });
    await page.locator('button[data-mode="outages"]').click();
    await page.locator('#region-select').selectOption('48201');
    await checkElevation(page, '48201');
    await page.waitForFunction(() => !document.body.classList.contains('mode-flow'));
    await page.screenshot({ path: `${screenshotDirectory}/outages-${viewport.width}.png`, fullPage: true });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Outage overflow at ${viewport.width}`);
    checks += 2;
  }
  const touch = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  const mobile = await touch.newPage();
  await mobile.clock.setFixedTime(new Date(now));
  await mobile.goto(`${base}/?source=dummy`);
  await mobile.waitForFunction(() => document.body.dataset.status === 'partial');
  await mobile.locator('#region-targets [data-region="LZ_HOUSTON"]').tap();
  await mobile.locator('#region-select').focus();
  await checkElevation(mobile, 'LZ_HOUSTON');
  assert.match(await mobile.locator('#panel-content').innerText(), /Houston/); checks++;
  await mobile.locator('button[data-mode="outages"]').tap();
  await mobile.locator('#region-targets [data-region="48201"]').tap();
  await mobile.locator('#region-select').focus();
  await checkElevation(mobile, '48201');
  assert.match(await mobile.locator('#panel-content').innerText(), /Harris County/); checks++;
  await mobile.locator('button[data-mode="energy"]').tap();
  await mobile.locator('#utility-zones [data-region="LZ_AEN"]').tap();
  assert.equal(await mobile.locator('#region-select').inputValue(), 'LZ_AEN');
  await mobile.locator('#next').tap();
  assert.equal(await mobile.locator('#time-range').inputValue(), '61');
  await mobile.locator('.rules-link').tap();
  await mobile.waitForURL('**/rules.html');
  assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await mobile.screenshot({ path: `${screenshotDirectory}/rules-mobile.png`, fullPage: true });
  await mobile.locator('.back-map').tap();
  await mobile.waitForFunction(() => document.body.dataset.status === 'partial');
  assert.equal(await mobile.locator('#region-select').inputValue(), 'LZ_AEN'); checks += 4;
  await touch.close();
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.locator('button[data-mode="energy"]').click();
  assert.equal(await page.locator('.mode-switch').evaluate(element => getComputedStyle(element, '::before').transitionDuration), '0s'); checks++;
  assert.equal(await page.locator('body').evaluate(element => element.classList.contains('mode-flow')), false);
  const reducedEnergy = await page.locator('#map-gradients stop').evaluateAll(stops => stops.map(stop => stop.getAttribute('stop-color')));
  await page.waitForTimeout(250);
  assert.deepEqual(await page.locator('#map-gradients stop').evaluateAll(stops => stops.map(stop => stop.getAttribute('stop-color'))), reducedEnergy);
  await page.locator('button[data-mode="outages"]').click();
  assert.equal(await page.locator('body').evaluate(element => element.classList.contains('mode-flow')), false);
  assert.notDeepEqual(await page.locator('#map-gradients stop').evaluateAll(stops => stops.map(stop => stop.getAttribute('stop-color'))), reducedEnergy); checks += 4;
  const reducedView = await page.locator('#texas-map').getAttribute('viewBox');
  await page.locator('#zoom-in').click();
  assert.equal(await page.locator('#texas-map').evaluate(element => element.classList.contains('is-zooming')), false);
  assert.equal(await page.locator('#texas-map').evaluate(element => getComputedStyle(element).transform), 'none'); checks++;
  assert.equal(await page.locator('#zoom-reset').innerText(), '125%');
  assert.notEqual(await page.locator('#texas-map').getAttribute('viewBox'), reducedView);
  await page.locator('#zoom-reset').click();
  assert.equal(await page.locator('#texas-map').getAttribute('viewBox'), reducedView); checks += 4;
  const firstCounty = page.locator('#region-targets .region').first();
  const firstCountyId = await firstCounty.getAttribute('data-region');
  const firstCountyPiece = page.locator(`[data-region-piece="${firstCountyId}"]`);
  await firstCounty.focus();
  assert.equal(await firstCountyPiece.evaluate(element => getComputedStyle(element).transform), 'none');
  assert.equal(await firstCountyPiece.evaluate(element => getComputedStyle(element).transitionDuration), '0s');
  assert.match(await page.locator('#selection-state').innerText(), /Preview/); checks += 3;
  await page.emulateMedia({ forcedColors: 'active' });
  assert.equal(await page.locator('#map-texture').isVisible(), false); checks += 2;
  assert.equal(await page.locator('#map-relief').isVisible(), false); checks++;
  assert.equal(await page.locator('#terrain-stage').isVisible(), false);
  assert.equal(await page.locator('#terrain-base').isVisible(), false);
  assert.equal(await page.locator('.region-surface').first().isVisible(), false);
  await page.keyboard.press('Enter');
  assert.equal(await firstCounty.getAttribute('aria-pressed'), 'true');
  assert.notEqual(await firstCountyPiece.locator('.region-fill').evaluate(element => getComputedStyle(element).fill), await page.locator('.region-piece:not(.active):not(.pinned):not(.focused) .region-fill').first().evaluate(element => getComputedStyle(element).fill)); checks += 5;
  assert.deepEqual(errors, []);
  assert.deepEqual(animationFailures, [], 'Mode transitions must show intermediate gradient frames at every tested zoom');
  console.log(`${checks} browser checks passed. Screenshots: frontend/screenshots/. Test server closed on exit.`);
} finally {
  try { await browser?.close(); }
  finally {
    server.closeAllConnections();
    await new Promise(resolveClosed => server.close(resolveClosed));
  }
}
