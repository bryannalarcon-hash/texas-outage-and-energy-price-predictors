import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const base = (process.env.BPC_FORECAST_URL || 'http://127.0.0.1:8099/').replace(/\/?$/, '/');
const output = new URL('screenshots/', import.meta.url);
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(()=>document.body.dataset.source==='model' && document.body.dataset.status!=='loading');
  await page.waitForFunction(()=>!document.querySelector('#pull-forecast').disabled,{timeout:30000});
  const response=page.waitForResponse(r=>r.url().endsWith('/api/forecast/refresh') && r.request().method()==='POST');
  await page.locator('#pull-forecast').click();
  assert.equal((await response).status(),202);
  await page.waitForFunction(()=>!document.querySelector('#pull-forecast').disabled,null,{timeout:45000});
  const service=await page.request.get(`${base}api/forecast/status`);
  assert.equal(service.status(),200);
  const state=await service.json();
  assert.equal(state.last_error,null);
  assert.ok(state.last_success_utc);
  const live=await (await page.request.get(`${base}data/model-output.json`)).json();
  assert.equal(live.kind,'model');
  assert.equal(live.outage.status,'available');
  assert.ok(live.outage.records.some(r=>r.coverage==='scenario'));
  await page.locator('button[data-mode="outages"]').click();
  await page.locator('#region-select').selectOption('48201');
  assert.match(await page.locator('#panel-content').innerText(),/Conditional scenario/);
  await page.waitForFunction(()=>!document.body.classList.contains('mode-flow'));
  await page.evaluate(()=>document.activeElement?.blur());
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:new URL('forecast-live-outages-desktop.png',output).pathname,fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:new URL('forecast-live-outages-mobile.png',output).pathname,fullPage:true});
  // Optional genuine historical bundle verifies the price→plan display without changing the live export.
  if(process.env.BPC_REPLAY_EXPORT) {
    const replay=await readFile(process.env.BPC_REPLAY_EXPORT,'utf8');
    await page.route('**/data/model-output.json',r=>r.fulfill({status:200,contentType:'application/json',body:replay}));
    await page.locator('#reload').click();
    await page.waitForFunction(()=>document.body.dataset.status==='stale');
    await page.locator('button[data-mode="energy"]').click();
    await page.locator('#region-select').selectOption('LZ_HOUSTON');
    await page.locator('#interval-select').selectOption('60');
    assert.match(await page.locator('#panel-content').innerText(),/Simulated battery plan/);
    assert.match(await page.locator('#panel-content').innerText(),/Stored.*Reserve/s);
    await page.locator('.signal-details > summary').click();
    assert.match(await page.locator('#panel-content').innerText(),/Mean: adaptive DAM \+ E2/);
    await page.waitForFunction(()=>!document.body.classList.contains('mode-flow'));
    await page.evaluate(()=>document.activeElement?.blur());
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:new URL('forecast-replay-energy-mobile.png',output).pathname,fullPage:true});
    await page.setViewportSize({width:1440,height:1050});
    await page.screenshot({path:new URL('forecast-replay-energy-desktop.png',output).pathname,fullPage:true});
  }
  assert.deepEqual(errors,[]);
  console.log('Forecast API, manual pull, conditional outage display, mobile layout and optional real-model replay passed.');
} finally { await browser.close(); }
