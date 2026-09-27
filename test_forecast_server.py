import copy
from datetime import datetime, timedelta, timezone
from functools import partial
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import forecast_server as server


def fixture_inputs():
    x=json.loads((server.ROOT/'model_assets/replay.json').read_text())
    sources=[{'source_id':name,'status':'ready','first_observed_at_utc':'2025-06-30T21:50:00Z','published_at_utc':'2025-06-30T17:00:00Z','sha256':name,'document_id':'123' if name=='dam' else None,'product_id':name} for name in ('dam','spc','wpc')]
    return {'target_date':'2025-07-01','sources':sources,
            'dam':{**sources[0],'rows':[{'settlement_point':r['settlement_point'],'valid_start_utc':r['interval_start_utc'],'dam_price':r['dam_spp_usd_mwh']} for r in x['hourly_dam_rows']]},
            'weather':{'status':'ready','forecast_origin_utc':'2025-06-30T12:00:00Z','counties':x['county_rows']}}


class ForecastServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cache=patch.object(server,'CACHE',self.root/'cache')
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.inputs=fixture_inputs()
        self.issue=datetime(2025,6,30,22,tzinfo=timezone.utc)

    def test_real_models_dispatch_and_frontend_contract(self):
        bundle=server.build_bundle(self.inputs,self.issue)
        checked=server.publish(bundle,self.root/'output.json')
        self.assertEqual(checked['prices'],768)
        self.assertEqual(checked['counties'],4)
        self.assertGreater(checked['decisions'],384)
        self.assertEqual(checked['status'],'stale')
        old=(self.root/'output.json').read_bytes()
        bundle['price']['records'][0]['rtm_p10_usd_mwh']=1e12
        with self.assertRaises(ValueError): server.publish(bundle,self.root/'output.json')
        self.assertEqual(old,(self.root/'output.json').read_bytes())

    def test_scheduler_waits_ten_minutes_and_remembers_completion(self):
        service=server.ForecastService(self.root/'output.json')
        with patch.object(server.sources,'probe_sources',return_value=self.inputs), patch.object(server.sources,'pull_inputs',return_value=self.inputs), patch.object(server,'feedback',return_value=[]), patch.object(server,'now_utc',return_value=self.issue-timedelta(seconds=1)):
            service.check_sources()
            self.assertFalse(service.status()['running'])
            self.assertFalse(service.output.exists())
        with patch.object(server.sources,'probe_sources',return_value=self.inputs), patch.object(server.sources,'pull_inputs',return_value=self.inputs), patch.object(server,'feedback',return_value=[]), patch.object(server,'now_utc',return_value=self.issue):
            service.check_sources()
            deadline=time.monotonic()+15
            while service.status()['running'] and time.monotonic()<deadline: time.sleep(.05)
            self.assertFalse(service.status()['running'])
            self.assertIsNone(service.status().get('last_error'))
            self.assertEqual(len(service.completed),1)
            restarted=server.ForecastService(service.output)
            with patch.object(restarted,'start') as start:
                restarted.check_sources()
                start.assert_not_called()

    def test_publication_rejects_discontinuous_energy_and_wrong_daily_inventory(self):
        bundle=server.build_bundle(self.inputs,self.issue)
        output=self.root/'output.json'
        server.publish(bundle,output)
        original=output.read_bytes()
        energy=[r for r in bundle['decisions']['records'] if r['mode']=='energy' and r['region_id']=='LZ_HOUSTON']
        middle=next(r for r in energy[1:-1] if max(r['stored_energy_start_kwh'],r['stored_energy_end_kwh'])<24)
        for row,message in [(energy[0],'start the day'),(energy[-1],'end the day'),(middle,'continuous')]:
            with self.subTest(message=message):
                broken=copy.deepcopy(bundle)
                changed=next(r for r in broken['decisions']['records'] if r['mode']=='energy' and r['region_id']==row['region_id'] and r['interval_start_utc']==row['interval_start_utc'])
                changed['stored_energy_start_kwh']+=.1
                changed['stored_energy_end_kwh']+=.1
                changed['locked']=True
                with self.assertRaisesRegex(ValueError,message): server.publish(broken,output)
                self.assertEqual(output.read_bytes(),original)
        # Missing endpoints impose no invented starting SOC; input order is irrelevant.
        partial=copy.deepcopy(bundle)
        partial['decisions']['records']=[*reversed(energy[1:3]),*(r for r in bundle['decisions']['records'] if r['mode']=='outages')]
        server.publish(partial,self.root/'partial.json')
        partial['decisions']['records']=[]
        self.assertEqual(server.publish(partial,self.root/'empty-plan.json')['decisions'],0)

    def test_revision_cannot_skip_its_new_delay(self):
        service=server.ForecastService(self.root/'output.json')
        old_key,_=server.schedule_candidate(self.inputs)
        revised=copy.deepcopy(self.inputs)
        revised['sources'][0]['document_id']='456'
        revised['sources'][0]['first_observed_at_utc']=server.iso(self.issue)
        with patch.object(server.sources,'pull_inputs',return_value=revised), patch.object(server,'now_utc',return_value=self.issue), patch.object(server,'build_bundle') as build:
            service.run(old_key)
            build.assert_not_called()
        self.assertFalse(service.output.exists())
        self.assertFalse(service.completed)

    def test_source_failure_preserves_export(self):
        output=self.root/'output.json'; output.write_text('{}')
        service=server.ForecastService(output)
        with patch.object(server.sources,'pull_inputs',side_effect=OSError('source unavailable')):
            service.run()
        self.assertEqual(output.read_text(),'{}')
        self.assertIn('source unavailable',service.status()['last_error'])

    def test_manual_cooldown_coalesces_running_jobs_and_exempts_automatic_runs(self):
        service=server.ForecastService(self.root/'output.json')
        with patch.object(server.threading,'Thread') as thread, patch.object(server.time,'monotonic',return_value=100):
            self.assertTrue(service.start())
            self.assertFalse(service.start())
            thread.assert_called_once()
            service.update(running=False)
            before=service.status()
            self.assertFalse(service.start())
            self.assertEqual(service.status(),before)
            self.assertTrue(service.start('automatic-input'))
            self.assertEqual(service.next_manual_at,160)
            service.update(running=False)
            self.assertFalse(service.start())
        with patch.object(server.threading,'Thread'), patch.object(server.time,'monotonic',return_value=160):
            self.assertTrue(service.start())

    def test_same_weather_does_not_trigger_again_at_midnight(self):
        inputs=copy.deepcopy(self.inputs)
        inputs['dam']['status']='unavailable'
        inputs['sources'][0]['status']='unavailable'
        first=server.schedule_candidate(inputs)
        inputs['target_date']='2025-07-02'
        self.assertEqual(first,server.schedule_candidate(inputs))

    def test_partial_network_failure_keeps_prior_export(self):
        output=self.root/'output.json'; output.write_text('{}')
        service=server.ForecastService(output)
        broken=copy.deepcopy(self.inputs)
        broken['sources'][0]['status']='error'
        with patch.object(server.sources,'pull_inputs',return_value=broken): service.run()
        self.assertEqual(output.read_text(),'{}')
        self.assertIn('source request failed',service.status()['last_error'])

    def test_midday_refresh_keeps_earlier_actions_and_carried_energy(self):
        previous=server.build_bundle(self.inputs,self.issue)
        inputs=copy.deepcopy(self.inputs)
        inputs['target_date']='2025-07-02'
        inputs['dam']['status']='unavailable'
        inputs['weather']['forecast_origin_utc']='2025-07-01T12:00:00Z'
        issued=datetime(2025,7,1,12,1,tzinfo=timezone.utc)
        updated=server.build_bundle(inputs,issued,previous=previous)
        self.assertEqual(updated['price'],previous['price'])
        old={(r['region_id'],r['interval_start_utc']):r for r in previous['decisions']['records'] if r['mode']=='energy'}
        locked=[r for r in updated['decisions']['records'] if r.get('locked')]
        self.assertEqual(len(locked),4*29)
        for r in locked:
            expected=old[(r['region_id'],r['interval_start_utc'])]
            self.assertEqual(r['charge_kwh'],expected['charge_kwh'])
            self.assertEqual(r['discharge_kwh'],expected['discharge_kwh'])
        for zone in ['LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']:
            rows=[r for r in updated['decisions']['records'] if r['mode']=='energy' and r['region_id']==zone]
            for a,b in zip(rows,rows[1:]): self.assertAlmostEqual(a['stored_energy_end_kwh'],b['stored_energy_start_kwh'])
        self.assertEqual(server.publish(updated,self.root/'midday.json')['prices'],768)

    def test_published_automatic_key_recovers_after_a_crash(self):
        key,_=server.schedule_candidate(self.inputs)
        output=self.root/'output.json'
        output.write_text(json.dumps({'automatic_input_id':key}))
        service=server.ForecastService(output)
        self.assertIn(key,service.completed)
        with patch.object(server.sources,'probe_sources',return_value=self.inputs), patch.object(service,'start') as start:
            service.check_sources()
            start.assert_not_called()

    def test_feedback_rounds_receipt_up_without_backdating(self):
        path=server.CACHE/'predictions/2025-06-27.json'
        row={'settlement_point':'LZ_NORTH','interval_end_utc':'2025-06-28T00:00:00Z'}
        server.atomic_json(path,{'target_date':'2025-06-27','issued_at_utc':'2025-06-26T22:00:00Z','records':[row]})
        labels={'status':'ready','first_observed_at_utc':'2025-06-30T21:59:59.123456Z','rows':[{'settlement_point':'LZ_NORTH','valid_end_utc':row['interval_end_utc'],'rtm_price':42.}]}
        with patch.object(server.sources,'fetch_rtm_labels',return_value=labels):
            history=server.feedback(self.issue)
        self.assertEqual(history[0]['actual_available_at_utc'],'2025-06-30T22:00:00Z')

    def test_refresh_http_rejects_cross_origin_and_arbitrary_input(self):
        service=server.ForecastService(self.root/'output.json')
        httpd=server.ThreadingHTTPServer(('127.0.0.1',0),partial(server.Handler,directory=str(server.ROOT/'frontend')))
        httpd.forecast=service
        origin=f'http://127.0.0.1:{httpd.server_port}'
        httpd.allowed_origins={origin}
        worker=threading.Thread(target=httpd.serve_forever,daemon=True); worker.start()
        try:
            def post(body,origin_value):
                conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
                conn.request('POST','/api/forecast/refresh',body,{'Origin':origin_value,'Content-Type':'application/json'})
                reply=conn.getresponse(); code=reply.status; reply.read(); conn.close(); return code
            with patch.object(service,'start') as start:
                self.assertEqual(post('{}','https://other.example'),403)
                self.assertEqual(post('{"url":"https://other.example"}',origin),400)
                start.assert_not_called()
                self.assertEqual(post('{}',origin),202)
                start.assert_called_once()
            before=service.status()
            service.next_manual_at=time.monotonic()+60
            conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
            conn.request('POST','/api/forecast/refresh','{}',{'Origin':origin,'Content-Type':'application/json'})
            reply=conn.getresponse()
            self.assertEqual(reply.status,429)
            self.assertGreaterEqual(int(reply.getheader('Retry-After')),59)
            self.assertIn('existing forecast is unchanged',json.loads(reply.read())['error'])
            conn.close()
            self.assertEqual(service.status(),before)
            self.assertEqual(server.Handler.timeout,10)
            with patch.object(server.Handler,'timeout',.1):
                conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
                conn.request('POST','/api/forecast/refresh',headers={'Origin':origin,'Content-Type':'application/json','Content-Length':'2'})
                with self.assertRaises(http.client.RemoteDisconnected): conn.getresponse()
                conn.close()
        finally:
            httpd.shutdown(); httpd.server_close(); worker.join()

if __name__=='__main__': unittest.main()
