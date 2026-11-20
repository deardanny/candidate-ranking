"""Versioned news ranking, bounded shadow and artifact-bound deployment control.

Reuses FairJob's Redis client and deployment/concurrency patterns, with a separate
news contract and state lifetime. Redis values are checked against a trusted local
snapshot; the local snapshot is the fallback and feature materialization source.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty,Full
from threading import Lock
from time import perf_counter
from typing import Literal

from fastapi import FastAPI,HTTPException,Header,Response
import numpy as np
from prometheus_client import CollectorRegistry,Counter,Histogram,generate_latest,CONTENT_TYPE_LATEST
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator

from ..catalog import redis_client
from ..io import sha256
from .features import FeatureState,seconds
from .bundle import Bundle


class RankRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    schema_version:Literal['ebnerd-v1']='ebnerd-v1'
    request_id:str=Field(min_length=1,max_length=128)
    user_id:int=Field(ge=0)
    request_time:str
    device_type:int=Field(ge=0,le=3)
    candidate_ids:list[int]=Field(min_length=1,max_length=128)
    top_k:int=Field(ge=1,le=128)

    @field_validator('request_time')
    @classmethod
    def valid_time(cls,value):
        seconds(value); return value

    @model_validator(mode='after')
    def valid_candidates(self):
        if len(set(self.candidate_ids))!=len(self.candidate_ids) or any(i<0 for i in self.candidate_ids): raise ValueError('Invalid candidate IDs')
        if self.top_k>len(self.candidate_ids): raise ValueError('top_k exceeds candidates')
        return self


def payload(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


class StateCache:
    def __init__(self,state,url=None):
        self.state=state; self.url=url; self.client=redis_client(url) if url else None
        self.articles={i:payload({'version':state.version,'article_id':i,**a}) for i,a in state.articles.items()}
        self.users={i:payload({'version':state.version,'user_id':i,**u}) for i,u in state.users.items()}

    def keys_values(self,ids,user):
        return [(f'ebnerd:{self.state.version}:article:{i}',self.articles[i]) for i in ids if i in self.articles]+(
            [(f'ebnerd:{self.state.version}:history:{user}',self.users[user])] if user in self.users else [])

    def seed(self):
        if self.client is None: return 0
        pairs=self.keys_values(list(self.articles),-1)+[(f'ebnerd:{self.state.version}:history:{i}',v) for i,v in self.users.items()]
        client=redis_client(self.url,timeout=2)
        try:
            for start in range(0,len(pairs),500):
                with client.pipeline(transaction=False) as pipeline:
                    for k,v in pairs[start:start+500]: pipeline.set(k,v)
                    pipeline.execute()
        finally: client.close()
        return len(pairs)

    def lookup(self,ids,user):
        if not self.client: return 0,[]
        pairs=self.keys_values(ids,user)
        try: values=self.client.mget([k for k,_ in pairs])
        except Exception: return 0,['redis_unavailable']
        if len(values)!=len(pairs): return 0,['cache_invalid']
        hits=sum(actual==expected for actual,(_,expected) in zip(values,pairs))
        reasons=[]
        if any(v is None for v in values): reasons.append('cache_miss')
        if any(v is not None and v!=expected for v,(_,expected) in zip(values,pairs)): reasons.append('cache_invalid')
        return hits,reasons

    def close(self):
        if self.client: self.client.close()


class Ranker:
    def __init__(self,bundle,state,cache=None):
        self.bundle=Bundle(bundle); self.state=state; self.cache=cache or StateCache(state)

    def rank(self,request):
        start=perf_counter(); now=seconds(request.request_time)
        # Future state is rejected even when all candidates would be skipped.
        if now<self.state.available_from or self.state.latest_event>=now: raise ValueError('Future history snapshot for request')
        ids=[i for i in request.candidate_ids if i in self.state.articles]
        skipped=[i for i in request.candidate_ids if i not in self.state.articles]
        if not ids: raise ValueError('No catalog candidates remain')
        hits,reasons=self.cache.lookup(ids,request.user_id)
        if skipped: reasons.append('missing_catalog_candidate')
        if request.user_id not in self.state.users: reasons.append('missing_user_history')
        looked=perf_counter(); x=self.state.matrix(request.user_id,request.request_time,request.device_type,ids)
        built=perf_counter(); method='lightgbm_lambdarank'; version=self.bundle.meta['model_version']
        try: scores=self.bundle.predict(x)
        except Exception:
            scores=np.array([np.log1p(self.state.pop[i]) for i in ids]); reasons.append('model_inference_error')
            method='fixed_past_click_popularity'; version='fallback-'+self.state.version[:12]
        inferred=perf_counter()
        ranked=sorted(zip(ids,scores),key=lambda z:(-float(z[1]),z[0]))[:request.top_k]
        result={'request_id':request.request_id,'schema_version':'ebnerd-v1','score_type':'ranking_score',
            'model_version':version,'attempted_model_version':self.bundle.meta['model_version'],
            'state_version':self.state.version,'model_metadata_sha256':self.bundle.digest,
            'content_snapshot_assumption':self.bundle.meta['content_snapshot_assumption'],
            'publication_stability_assumption':self.bundle.meta['publication_stability_assumption'],
            'scoring_method':method,'degraded':bool(reasons),'fallback_reason':','.join(sorted(set(reasons))) or None,
            'cache_hits':hits,'skipped_candidate_ids':skipped,
            'model_unseen_candidates':sum(i not in self.bundle.seen for i in ids),
            'rankings':[{'article_id':i,'score':float(s),'position':j+1} for j,(i,s) in enumerate(ranked)],
            'timings_ms':{'lookup':(looked-start)*1000,'features':(built-looked)*1000,
                          'inference':(inferred-built)*1000,'service':(perf_counter()-start)*1000}}
        return result


QUALITY_KEYS={'both_development_folds_positive','holdout_ndcg_lower_positive','hit1_nonnegative','cold_hit1_nonnegative','offline_online_parity'}


def check_evidence(candidate,active,state,quality,performance):
    reasons=[]
    for label,evidence in [('quality',quality),('performance',performance)]:
        if evidence.get('candidate_metadata_sha256')!=sha256(Path(candidate)/'metadata.json'): reasons.append(label+'_candidate_mismatch')
        if evidence.get('state_version')!=state.version: reasons.append(label+'_state_mismatch')
    if quality.get('champion_metadata_sha256')!=active.bundle.digest: reasons.append('champion_mismatch')
    checks=quality.get('checks',{})
    if set(checks)!=QUALITY_KEYS or any(v is not True for v in checks.values()): reasons.append('quality_gate_failed')
    rate=performance.get('on_time_success_rate',-1)
    if (not isinstance(rate,(int,float)) or not np.isfinite(rate) or rate<.999 or rate>1
        or performance.get('scheduled',0)<500 or performance.get('duration_seconds',0)<10
        or performance.get('deadline_ms')!=50 or performance.get('invalid_responses')!=0
        or performance.get('degraded_successful')!=0 or performance.get('generator_dropped')!=0): reasons.append('performance_gate_failed')
    return reasons


class Deployment:
    def __init__(self,bundle,state,cache,alias_callback=None):
        self.active=Ranker(bundle,state,cache); self.previous=None; self.lock=Lock(); self.transition=Lock(); self.events=[]
        self.alias_callback=alias_callback

    def snapshot(self):
        with self.lock: return self.active

    def promote(self,directory,quality,performance,warmup):
        with self.transition:
            active=self.snapshot(); reasons=check_evidence(directory,active,active.state,quality,performance)
            if reasons: raise ValueError(';'.join(reasons))
            staged=Ranker(directory,active.state,active.cache); response=staged.rank(warmup)
            if response['degraded']: raise ValueError('Warmup degraded')
            if self.alias_callback: self.alias_callback(staged.bundle)
            with self.lock: self.previous,self.active=self.active,staged
            event={'action':'promote','from':active.bundle.meta['model_version'],'to':staged.bundle.meta['model_version']}
            self.events.append(event); return event

    def rollback(self):
        with self.transition:
            if self.previous is None: raise ValueError('No rollback target')
            if self.alias_callback: self.alias_callback(self.previous.bundle)
            with self.lock:
                old=self.active; self.active,self.previous=self.previous,self.active
            event={'action':'rollback','from':old.bundle.meta['model_version'],'to':self.active.bundle.meta['model_version']}
            self.events.append(event); return event


def shadow_worker(bundle,state_path,incoming,outgoing,ready):
    try:
        state=FeatureState.load(state_path); ranker=Ranker(bundle,state); ready.send({'ready':True})
        while True:
            item=incoming.get()
            if item is None: return
            request,active=item
            try:
                prediction=ranker.rank(RankRequest.model_validate(request))
                if active['state_version']!=prediction['state_version']: raise ValueError('Shadow state mismatch')
                result={'status':'compared','top1_agreement':active['rankings'][0]['article_id']==prediction['rankings'][0]['article_id'],
                    'active_version':active['model_version'],'shadow_version':prediction['model_version']}
            except Exception as error: result={'status':'error','error':type(error).__name__}
            try: outgoing.put_nowait(result)
            except Full: pass
    except Exception as error: ready.send({'ready':False,'error':type(error).__name__})
    finally: ready.close()


class Shadow:
    def __init__(self,bundle,state_path,capacity=16):
        ctx=mp.get_context('spawn'); self.incoming=ctx.Queue(capacity); self.outgoing=ctx.Queue(capacity)
        receiver,sender=ctx.Pipe(duplex=False)
        self.process=ctx.Process(target=shadow_worker,args=(str(bundle),str(state_path),self.incoming,self.outgoing,sender),daemon=True)
        self.process.start(); sender.close()
        if not receiver.poll(60): self.close(); raise TimeoutError('Shadow startup timeout')
        status=receiver.recv(); receiver.close()
        if not status['ready']: self.close(); raise ValueError('Shadow startup failed')

    def submit(self,request,active):
        if not self.process.is_alive(): return False
        try: self.incoming.put_nowait((request,active)); return True
        except Full: return False

    def drain(self):
        result=[]
        while True:
            try: result.append(self.outgoing.get_nowait())
            except Empty: return result

    def close(self):
        if self.process.is_alive():
            try: self.incoming.put_nowait(None)
            except Full: pass
            self.process.join(2)
            if self.process.is_alive(): self.process.terminate(); self.process.join(5)
        for q in [self.incoming,self.outgoing]: q.cancel_join_thread(); q.close()


@dataclass
class Settings:
    bundle:Path
    state:Path
    redis_url:str|None=None
    shadow:Path|None=None
    max_inflight:int=4
    admin_token:str=''
    seed_redis:bool=False
    registry_json:Path|None=None


class Promotion(BaseModel):
    model_config=ConfigDict(extra='forbid')
    bundle:str
    quality:dict
    performance:dict
    warmup:RankRequest


def create_app(settings):
    if settings.max_inflight<1: raise ValueError('Invalid capacity')
    registry=CollectorRegistry(); requests=Counter('ebnerd_rank_requests','Ranking responses',['result'],registry=registry)
    latency=Histogram('ebnerd_rank_seconds','Handler latency',buckets=(.001,.0025,.005,.01,.025,.05,.1,1),registry=registry)
    @asynccontextmanager
    async def lifespan(app):
        state=await asyncio.to_thread(FeatureState.load,settings.state); cache=StateCache(state,settings.redis_url)
        if settings.seed_redis: await asyncio.to_thread(cache.seed)
        def alias_callback(bundle):
            if settings.registry_json is None: return
            import mlflow
            record=json.loads(settings.registry_json.read_text())
            entry=next((r for r in record['models'] if r['metadata_sha256']==bundle.digest),None)
            if entry is None: raise ValueError('Bundle is not registered')
            client=mlflow.MlflowClient(tracking_uri=record['tracking_uri'])
            version=client.get_model_version(record['registry_name'],entry['version'])
            if version.tags.get('bundle_metadata_sha256')!=bundle.digest: raise ValueError('Registry digest mismatch')
            client.set_registered_model_alias(record['registry_name'],'champion',entry['version'])
        app.state.deployment=Deployment(settings.bundle,state,cache,alias_callback)
        app.state.shadow=await asyncio.to_thread(Shadow,settings.shadow,settings.state) if settings.shadow else None
        app.state.shadow_results=[]; app.state.shadow_counts={'submitted':0,'dropped':0,'compared':0,'error':0}
        app.state.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='ebnerd-score')
        app.state.tokens=asyncio.Queue(settings.max_inflight)
        for _ in range(settings.max_inflight): app.state.tokens.put_nowait(True)
        async def collect():
            while True:
                if app.state.shadow:
                    for result in app.state.shadow.drain():
                        app.state.shadow_counts[result['status']]+=1; app.state.shadow_results.append(result)
                    del app.state.shadow_results[:-20]
                await asyncio.sleep(.05)
        collector=asyncio.create_task(collect())
        try: yield
        finally:
            collector.cancel()
            try: await collector
            except asyncio.CancelledError: pass
            app.state.pool.shutdown(wait=True)
            if app.state.shadow: await asyncio.to_thread(app.state.shadow.close)
            cache.close()
    app=FastAPI(title='EB-NeRD temporal ranking lab',lifespan=lifespan)

    @app.get('/healthz')
    def health(): return {'status':'ok'}

    @app.get('/readyz')
    def ready():
        ranker=app.state.deployment.snapshot()
        return {'ready':True,'model_version':ranker.bundle.meta['model_version'],'state_version':ranker.state.version}

    @app.get('/metrics')
    def metrics(): return Response(generate_latest(registry),headers={'Content-Type':CONTENT_TYPE_LATEST})

    @app.get('/shadow/status')
    def shadow_status(): return {**app.state.shadow_counts,'recent':app.state.shadow_results}

    @app.post('/rank')
    async def rank(request:RankRequest):
        try: app.state.tokens.get_nowait()
        except asyncio.QueueEmpty: requests.labels('overload').inc(); raise HTTPException(503,'Ranking capacity exhausted')
        started=perf_counter(); active=app.state.deployment.snapshot()
        try: future=asyncio.get_running_loop().run_in_executor(app.state.pool,active.rank,request)
        except BaseException: app.state.tokens.put_nowait(True); raise
        future.add_done_callback(lambda _:app.state.tokens.put_nowait(True))
        try: result=await asyncio.shield(future)
        except ValueError as error: requests.labels('invalid').inc(); raise HTTPException(422,str(error)) from error
        latency.observe(perf_counter()-started); requests.labels('degraded' if result['degraded'] else 'ok').inc()
        if app.state.shadow and int.from_bytes(hashlib.sha256(request.request_id.encode()).digest()[:8],'big')%5==0:
            submitted=app.state.shadow.submit(request.model_dump(),result)
            app.state.shadow_counts['submitted' if submitted else 'dropped']+=1
        return result

    def authorize(token):
        if not settings.admin_token or not hmac.compare_digest(token or '','Bearer '+settings.admin_token): raise HTTPException(403,'Unauthorized deployment control')

    @app.post('/admin/promote')
    async def promote(body:Promotion,authorization:str|None=Header(default=None)):
        authorize(authorization)
        try: return await asyncio.to_thread(app.state.deployment.promote,body.bundle,body.quality,body.performance,body.warmup)
        except Exception as error: raise HTTPException(409,str(error)) from error

    @app.post('/admin/rollback')
    async def rollback(authorization:str|None=Header(default=None)):
        authorize(authorization)
        try: return await asyncio.to_thread(app.state.deployment.rollback)
        except ValueError as error: raise HTTPException(409,str(error)) from error

    return app


def app_factory():
    return create_app(Settings(Path(os.environ['EBNERD_BUNDLE']),Path(os.environ['EBNERD_STATE']),
        os.environ.get('EBNERD_REDIS_URL'),Path(os.environ['EBNERD_SHADOW']) if os.environ.get('EBNERD_SHADOW') else None,
        int(os.environ.get('EBNERD_MAX_INFLIGHT','4')),os.environ.get('EBNERD_ADMIN_TOKEN',''),os.environ.get('EBNERD_SEED_REDIS')=='1',
        Path(os.environ['EBNERD_REGISTRY_JSON']) if os.environ.get('EBNERD_REGISTRY_JSON') else None))
