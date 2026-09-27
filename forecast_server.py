#!/usr/bin/env python3
"""Local forecast service, manual refresh, and durable publication-aware scheduling."""
import argparse
from datetime import datetime, timedelta, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

import forecast_sources as sources
from forecast_models import price_forecast, outage_forecast
from forecast_dispatch import decisions

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'forecast_cache'
UTC = timezone.utc


def now_utc():
    return datetime.now(UTC).replace(microsecond=0)


def iso(value):
    return value.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, allow_nan=False, separators=(',', ':'))
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def schedule_candidate(inputs):
    """Separate input versions, including a usable outage-only morning update."""
    ready = [s for s in inputs['sources'] if s['status'] == 'ready']
    usable = inputs['dam']['status'] == 'ready' or inputs['weather']['status'] == 'ready'
    if not usable or not ready:
        return None
    signature = [inputs['target_date'] if inputs['dam']['status'] == 'ready' else None, inputs['weather']['forecast_origin_utc'],
                 [(s['source_id'],s.get('document_id'),s.get('product_id'),s.get('sha256')) for s in ready]]
    key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    observed = max(datetime.fromisoformat(s['first_observed_at_utc']) for s in ready)
    return key, observed + timedelta(minutes=10)


def source_status(inputs):
    names = {'dam':'ERCOT day-ahead prices','spc':'SPC severe-weather outlook','wpc':'WPC rainfall outlook'}
    return [{'name':names.get(s['source_id'],s['source_id']),
             'detail':('Ready · published '+s.get('published_at_utc','at an unspecified time')) if s['status']=='ready' else s.get('message','Waiting for publication')[:180]}
            for s in inputs['sources']]


def feedback(issued):
    """Join our pre-interval forecasts with actuals only after 48 elapsed hours."""
    joined = []
    # Retain every ledger; use all available matured days, including catch-up after downtime.
    for path in sorted((CACHE/'predictions').glob('*.json')):
        saved = read_json(path,{})
        records = saved.get('records',[])
        if not records or datetime.fromisoformat(records[0]['interval_end_utc']) > issued-timedelta(hours=48):
            continue
        actual = sources.fetch_rtm_labels(saved['target_date'], now=issued)
        if actual['status'] != 'ready':
            continue
        by_key = {(r['settlement_point'],r['valid_end_utc']):r for r in actual['rows']}
        for row in records:
            # Calibration uses strictly pre-interval forecasts, including after a same-day startup.
            if datetime.fromisoformat(saved['issued_at_utc']) >= datetime.fromisoformat(row['interval_end_utc'])-timedelta(minutes=15):
                continue
            observed = by_key.get((row['settlement_point'],row['interval_end_utc']))
            if observed:
                receipt = datetime.fromisoformat(actual['first_observed_at_utc'])
                if receipt.microsecond:
                    receipt = receipt.replace(microsecond=0) + timedelta(seconds=1)
                joined.append({**row,'model_issue_utc':saved['issued_at_utc'], 'rtm_spp_usd_mwh':observed['rtm_price'],
                               'actual_available_at_utc':iso(receipt)})
    return joined


def build_bundle(inputs, issued, history=(), previous=None):
    if inputs['weather']['status'] != 'ready' and any(
            r['mode']=='energy' and datetime.fromisoformat(r['interval_start_utc'])+timedelta(minutes=15)>issued
            for r in (previous or {}).get('decisions',{}).get('records',[])):
        raise ValueError('Weather inputs are incomplete; keeping the existing forecast and simulated battery state.')
    missing = lambda reason: {'status':'unavailable','reason':reason[:300]}
    price = missing(inputs['dam'].get('message','Tomorrow’s complete DAM curve is not available.'))
    outage = missing('The complete issued weather inputs are not available for the fixed 12Z horizon.')
    if inputs['dam']['status'] == 'ready':
        rows = [{'settlement_point':r['settlement_point'], 'interval_start_utc':r['valid_start_utc'],
                 'dam_spp_usd_mwh':r['dam_price']} for r in inputs['dam']['rows']]
        price = price_forecast(rows,iso(issued),target_date=inputs['target_date'],history_rows=history)
    elif previous and previous['price']['status'] == 'available' and max(datetime.fromisoformat(r['interval_end_utc']) for r in previous['price']['records']) > issued:
        price = previous['price']
    if inputs['weather']['status'] == 'ready':
        rows = [{**r,'at_risk_assumed':True} for r in inputs['weather']['counties']]
        origin = inputs['weather']['forecast_origin_utc']
        outage = outage_forecast(rows,iso(issued),forecast_origin_utc=origin,input_cutoff_utc=origin)
    return {'schema_version':1,'kind':'model','generated_at_utc':iso(issued),'price':price,'outage':outage,
            'decisions':decisions(price,outage,previous=previous), 'source_receipts':inputs['sources']}


def publish(bundle, output):
    """The browser's real data contract is the publication gate, not a duplicate schema."""
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',dir=output.parent,delete=False) as stream:
        staged = Path(stream.name)
        json.dump(bundle,stream,allow_nan=False,separators=(',',':'))
    try:
        checked = subprocess.run(['node',str(ROOT/'frontend/validate-export.mjs'),str(staged)],
                                 capture_output=True,text=True,timeout=30)
        if checked.returncode:
            raise ValueError('Forecast failed the frontend data contract: '+checked.stderr[-800:])
        receipt = json.loads(checked.stdout)
        staged.replace(output)
        return receipt
    finally:
        staged.unlink(missing_ok=True)


class ForecastService:
    def __init__(self, output, automatic=True):
        self.output, self.automatic = output, automatic
        self.lock = threading.RLock()
        self.source_lock = threading.Lock()
        self.stop = threading.Event()
        self.next_manual_at = 0.
        saved = read_json(CACHE/'service.json',{})
        self.state = {**saved,'running':False,'automatic':automatic,'message':'Ready to pull inputs. Automatic publication checks run every minute.'}
        self.state.pop('scheduled_for_utc',None)
        self.completed = read_json(CACHE/'automatic-runs.json',[])
        published = read_json(output,{})
        if published.get('automatic_input_id') and published['automatic_input_id'] not in self.completed:
            self.completed.append(published['automatic_input_id'])
            atomic_json(CACHE/'automatic-runs.json',self.completed[-400:])

    def status(self):
        with self.lock:
            return dict(self.state)

    def update(self,**changes):
        with self.lock:
            self.state.update(changes)
            atomic_json(CACHE/'service.json',self.state)

    def start(self,automatic_key=None):
        with self.lock:
            if self.state['running']:
                return False
            if automatic_key is None:
                current = time.monotonic()
                if current < self.next_manual_at:
                    return False
                self.next_manual_at = current + 60
            self.update(running=True,message='Pulling ERCOT prices and issued weather outlooks…')
        threading.Thread(target=self.run,args=(automatic_key,),daemon=True).start()
        return True

    def run(self,automatic_key=None):
        publication_committed = False
        try:
            with self.source_lock:
                inputs = sources.pull_inputs()
                if any(s['status'] == 'error' for s in inputs['sources']):
                    raise RuntimeError('A source request failed; preserving the last complete export. Try the refresh again.')
                if automatic_key:
                    candidate = schedule_candidate(inputs)
                    if not candidate or candidate[0] != automatic_key or now_utc() < candidate[1]:
                        self.update(running=False, message='Inputs changed. Waiting ten minutes from observation of the new input set.')
                        return
                history = feedback(now_utc())
            self.update(sources=source_status(inputs),target_date=inputs['target_date'],message='Running price and county-risk models, then planning battery actions…')
            if inputs['dam']['status']!='ready' and inputs['weather']['status']!='ready':
                raise RuntimeError('Inputs are not ready. '+inputs['dam'].get('message','Please check source availability.'))
            issued = now_utc()
            bundle = build_bundle(inputs,issued,history,previous=read_json(self.output,None))
            if automatic_key:
                bundle['automatic_input_id'] = automatic_key
            checked = publish(bundle,self.output)
            publication_committed = True
            price = bundle['price']
            if price['status']=='available':
                path = CACHE/'predictions'/f"{price['target_date']}.json"
                # Keep the first as-issued daily curve for causal calibration; never overwrite it on manual retries.
                if not path.exists():
                    atomic_json(path,price)
            if automatic_key:
                actual = schedule_candidate(inputs)
                if actual and actual[0] == automatic_key and now_utc() >= actual[1]:
                    self.completed.append(automatic_key)
                    self.completed = self.completed[-400:]
                    atomic_json(CACHE/'automatic-runs.json',self.completed)
            message = ('Forecast ready for '+price['target_date']+'.' if price['status']=='available' else 'County outlook updated. Tomorrow’s prices are waiting for ERCOT’s DAM release.')
            if price.get('horizon') == 'remaining_day':
                message = 'Price forecast ready for the remaining intervals of '+price['target_date']+'.'
            if price['status']=='available' and (inputs.get('pending_day_ahead_date') or price['target_date'] != inputs['target_date']):
                message += ' The next day’s DAM release is still pending.'
            self.update(running=False,message=message,last_success_utc=iso(issued),export_version=hashlib.sha256(self.output.read_bytes()).hexdigest(),contract_status=checked['status'],last_error=None)
        except Exception as error:
            prefix = 'Forecast published, but saving its run history failed. ' if publication_committed else 'Update failed; the previous export is unchanged. '
            changes = {'running':False,'message':prefix+str(error)[:230],'last_error':str(error)[:500]}
            if publication_committed:
                changes.update(last_success_utc=iso(issued),export_version=hashlib.sha256(self.output.read_bytes()).hexdigest())
            self.update(**changes)

    def check_sources(self):
        if self.status()['running']:
            return
        with self.source_lock:
            inputs = sources.probe_sources()
        candidate = schedule_candidate(inputs)
        details = {'sources':source_status(inputs),'target_date':inputs['target_date']}
        if candidate and candidate[0] not in self.completed:
            key,due = candidate
            details['scheduled_for_utc'] = iso(due)
            if now_utc() >= due:
                self.update(**details)
                self.start(key)
            else:
                if not self.status().get('last_success_utc'):
                    details['message']='Inputs found. Automatic prediction is scheduled ten minutes after first observation.'
                self.update(**details)
        else:
            details['scheduled_for_utc']=None
            self.update(**details)

    def scheduler(self):
        while not self.stop.is_set():
            try:
                self.check_sources()
            except Exception as error:
                self.update(last_probe_error=str(error)[:250])
            self.stop.wait(60)


class Handler(SimpleHTTPRequestHandler):
    timeout = 10

    def send_json(self,code,data,retry_after=None):
        body=json.dumps(data,allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        if retry_after is not None:
            self.send_header('Retry-After',str(retry_after))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == '/api/forecast/status':
            return self.send_json(200,self.server.forecast.status())
        if urlsplit(self.path).path.startswith('/api/'):
            return self.send_json(404,{'error':'Unknown forecast endpoint'})
        return super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/api/forecast/refresh':
            return self.send_json(404,{'error':'Unknown forecast endpoint'})
        origin=self.headers.get('Origin','')
        if origin not in self.server.allowed_origins or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            return self.send_json(403,{'error':'Refresh requires an allowed same-origin browser request'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 1024 or self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                raise ValueError('Expected a small JSON object')
            data=json.loads(self.rfile.read(size))
            if data != {}:
                raise ValueError('Refresh accepts an empty object; sources are fixed by the server')
        except (ValueError,UnicodeDecodeError):
            return self.send_json(400,{'error':'Expected an empty JSON object'})
        service = self.server.forecast
        with service.lock:
            started = service.start()
            retry = max(0,math.ceil(service.next_manual_at-time.monotonic())) if not started and not service.state['running'] else 0
        if retry:
            return self.send_json(429,{'error':f'Please wait {retry} seconds before another manual refresh. The existing forecast is unchanged.','retry_after_seconds':retry},retry_after=retry)
        self.send_json(202,service.status())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8099)
    parser.add_argument('--allowed-origin',action='append',default=[])
    parser.add_argument('--no-auto',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--replay',action='store_true',help='Reproduce the historical fixture, with its original timestamps')
    args=parser.parse_args()
    output=ROOT/'frontend/data/model-output.json'
    if args.replay:
        example=read_json(ROOT/'model_assets/replay.json',{})
        price=price_forecast(example['hourly_dam_rows'],example['price_issued_at_utc'])
        prior_outage=outage_forecast(example['previous_county_rows'],example['previous_outage_issued_at_utc'],forecast_origin_utc=example['previous_outage_issued_at_utc'])
        previous={'price':price,'outage':prior_outage,'decisions':decisions(price,prior_outage)}
        outage=outage_forecast(example['county_rows'],example['outage_issued_at_utc'],forecast_origin_utc=example['outage_issued_at_utc'])
        bundle={'schema_version':1,'kind':'model','generated_at_utc':example['outage_issued_at_utc'],'price':price,'outage':outage,'decisions':decisions(price,outage,previous=previous)}
        print(json.dumps(publish(bundle,output)))
        return
    service=ForecastService(output,not args.no_auto)
    if args.once:
        service.run()
        print(json.dumps(service.status()))
        if service.status().get('last_error'):
            raise SystemExit(1)
        return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT/'frontend')))
    server.forecast=service
    server.allowed_origins={f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}',*args.allowed_origin}
    if service.automatic:
        threading.Thread(target=service.scheduler,daemon=True).start()
    print(f'Forecast service: http://127.0.0.1:{args.port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop.set()
        server.server_close()

if __name__=='__main__': main()
