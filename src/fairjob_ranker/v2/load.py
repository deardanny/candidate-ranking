"""Scheduled constant/burst load with deadline success and model identity checks."""
import asyncio
from collections import Counter
import math
import os
from time import perf_counter

import httpx

from ..loadtest import Resources,percentiles


def arrival_offset(index, rate, burst_size):
    return (index // burst_size) * burst_size / rate


async def run_load(url, requests, rate, duration, burst_size=1, expected_version=None, pids=None, max_pending=256):
    if rate<=0 or duration<=0 or burst_size<1 or max_pending<1 or not requests:
        raise ValueError('Invalid load parameters')
    scheduled=math.ceil(rate*duration)
    records,pending=[],set()
    stopped=asyncio.Event()
    server=Resources(pids or [])
    generator=Resources([os.getpid()],include_children=False)
    async def monitor():
        while not stopped.is_set():
            await asyncio.to_thread(server.sample)
            await asyncio.to_thread(generator.sample)
            try:
                await asyncio.wait_for(stopped.wait(),1)
            except asyncio.TimeoutError:
                pass
    async with httpx.AsyncClient(base_url=url,timeout=5,trust_env=False,
                                limits=httpx.Limits(max_connections=max_pending,max_keepalive_connections=max_pending)) as client:
        async def send(i,due):
            sent=perf_counter()
            request=requests[i%len(requests)]
            record={'index':i,'issued':True,'send_lag_ms':(sent-due)*1000,'status':'network_error','degraded':False}
            try:
                response=await client.post('/rank',json=request)
                record['status']=str(response.status_code)
                if response.status_code==200:
                    body=response.json()
                    rankings=body['rankings']
                    ids=[r['product_id'] for r in rankings]
                    scores=[r['score'] for r in rankings]
                    valid=(body['request_id']==request['request_id'] and len(ids)==len(set(ids)) and
                           0<len(ids)<=request['top_k'] and set(ids)<=set(request['candidate_ids']) and
                           all(math.isfinite(s) and 0<=s<=1 for s in scores) and
                           rankings==sorted(rankings,key=lambda r:(-r['score'],r['product_id'])) and
                           [r['position'] for r in rankings]==list(range(1,len(ids)+1)))
                    if expected_version and body['attempted_model_version']!=expected_version:
                        valid=False
                    if not valid:
                        raise ValueError('Invalid response contract or model version')
                    record.update(degraded=body['degraded'],model_version=body['model_version'],timings_ms=body['timings_ms'])
            except httpx.TimeoutException:
                record['status']='timeout'
            except httpx.HTTPError:
                record['status']='network_error'
            except (ValueError,TypeError,KeyError):
                record['status']='invalid_response'
            record['latency_ms']=(perf_counter()-due)*1000
            records.append(record)
        monitor_task=asyncio.create_task(monitor())
        began=perf_counter()
        try:
            for i in range(scheduled):
                due=began+arrival_offset(i,rate,burst_size)
                await asyncio.sleep(max(0,due-perf_counter()))
                if len(pending)>=max_pending:
                    records.append({'index':i,'issued':False,'status':'client_capacity','send_lag_ms':(perf_counter()-due)*1000})
                    continue
                task=asyncio.create_task(send(i,due)); pending.add(task)
                task.add_done_callback(pending.discard)
            await asyncio.sleep(max(0,began+duration-perf_counter()))
            if pending:
                await asyncio.gather(*pending)
        finally:
            stopped.set(); await monitor_task
        elapsed=perf_counter()-began
    records.sort(key=lambda r:r['index'])
    status=dict(Counter(r['status'] for r in records))
    success=[r for r in records if r['status']=='200']
    on_time=sum(r['latency_ms']<=50 for r in success)
    return {'target_rps':rate,'duration_seconds':duration,'burst_size':burst_size,'scheduled':scheduled,
            'issued':sum(r['issued'] for r in records),'successful':len(success),'status_counts':status,
            'generator_dropped':status.get('client_capacity',0),'invalid_responses':status.get('invalid_response',0),
            'degraded_successful':sum(r['degraded'] for r in success),'deadline_ms':50,
            'on_time_successful':on_time,'on_time_success_rate':on_time/scheduled,
            'successful_qps':len(success)/elapsed,'failure_rate':1-len(success)/scheduled,
            'latency_success_ms':percentiles([r['latency_ms'] for r in success]),
            'generator_lag_ms':percentiles([r['send_lag_ms'] for r in records]),
            'server_resources':server.summary(),'generator_resources':generator.summary(),'records':records}
