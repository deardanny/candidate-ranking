"""Observable ranking API with isolated shadow scoring and gated deployment."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict

from ..io import read_json, sha256
from ..serving import RankRequest, RankResponse
from .deployment import Deployment
from .shadow import Shadow


@dataclass
class Settings:
    active: Path
    shadow: Path | None = None
    redis_url: str | None = None
    max_inflight: int = 2
    shadow_capacity: int = 16
    shadow_sample_rate: float = .2
    admin_token: str = ''
    experiment_report: Path | None = None

    def __post_init__(self):
        if self.max_inflight < 1 or self.shadow_capacity < 1 or not 0 <= self.shadow_sample_rate <= 1:
            raise ValueError('Invalid service capacity or sample rate')

    @classmethod
    def from_env(cls):
        report = Path(os.environ.get('FAIRJOB_V2_EXPERIMENT','reports/v2/experiment/metrics.json'))
        experiment = read_json(report)
        shadow = os.environ.get('FAIRJOB_V2_SHADOW',experiment['challenger']['bundle'])
        return cls(active=Path(os.environ.get('FAIRJOB_V2_ACTIVE',experiment['champion']['bundle'])),
                   shadow=Path(shadow) if shadow else None,redis_url=os.environ.get('FAIRJOB_REDIS_URL') or None,
                   max_inflight=int(os.environ.get('FAIRJOB_MAX_INFLIGHT','2')),
                   shadow_capacity=int(os.environ.get('FAIRJOB_SHADOW_CAPACITY','16')),
                   shadow_sample_rate=float(os.environ.get('FAIRJOB_SHADOW_RATE','.2')),
                   admin_token=os.environ.get('FAIRJOB_ADMIN_TOKEN',''),experiment_report=report)


class Promotion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bundle: str
    quality_evidence: str
    performance_evidence: str
    warmup: RankRequest


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter('fairjob_http_requests','HTTP rank requests',['status'],registry=self.registry)
        self.duration = Histogram('fairjob_http_duration_seconds','Full ASGI rank response time',buckets=(.001,.0025,.005,.01,.025,.05,.1,.25,1,5),registry=self.registry)
        self.stages = Histogram('fairjob_stage_duration_seconds','Scoring stage time',['stage'],buckets=(.0001,.0005,.001,.005,.01,.05,.1),registry=self.registry)
        self.inflight = Gauge('fairjob_inflight','Admitted requests including queued CPU work',registry=self.registry)
        self.fallback = Counter('fairjob_fallback','Degraded responses',['reason'],registry=self.registry)
        self.shadow = Counter('fairjob_shadow','Shadow processing outcomes',['outcome'],registry=self.registry)
        self.shadow_pending = Gauge('fairjob_shadow_pending','Submitted minus collected shadow tasks',registry=self.registry)
        self.shadow_latency = Histogram('fairjob_shadow_duration_seconds','Shadow queue and scoring time',registry=self.registry)
        self.top1 = Counter('fairjob_shadow_top1','Top-1 agreement; not a quality metric',['agreement'],registry=self.registry)
        self.overlap = Histogram('fairjob_shadow_topk_overlap','Intersection / union of top-k IDs',buckets=(0,.25,.5,.75,1),registry=self.registry)
        self.deployments = Counter('fairjob_deployments','Deployment decisions',['outcome'],registry=self.registry)
        self.replay = Gauge('fairjob_replay_worst_on_time_ratio','Worst completed replay repeat; scheduled-time deadline, not live handler latency',
                            ['capacity','rps','burst'],registry=self.registry)


def create_app(settings=None, shadow_factory=Shadow):
    settings = settings or Settings.from_env()
    metrics = Metrics()

    def alias_callback(directory):
        if settings.experiment_report is None:
            return
        import mlflow
        experiment = read_json(settings.experiment_report)
        digest = sha256(Path(directory)/'metadata.json')
        row = next((r for r in experiment['runs'] if sha256(Path(r['bundle'])/'metadata.json')==digest),None)
        if row is None:
            raise ValueError('Bundle is not a registered experiment artifact')
        client = mlflow.MlflowClient(tracking_uri=experiment['tracking_uri'],registry_uri=experiment['tracking_uri'])
        version = client.get_model_version(experiment['registry_name'],row['registry_version'])
        if version.tags.get('bundle_metadata_sha256') != digest:
            raise ValueError('Registry artifact mismatch')
        client.set_registered_model_alias(experiment['registry_name'],'champion',row['registry_version'])

    @asynccontextmanager
    async def lifespan(app):
        app.state.metrics = metrics
        app.state.deployment = Deployment(settings.active,settings.redis_url,alias_callback)
        app.state.shadow,app.state.shadow_startup_error = None,None
        app.state.shadow_results = []
        app.state.shadow_totals = {'submitted':0,'dropped':0,'compared':0,'error':0}
        app.state.tokens = asyncio.Queue(maxsize=settings.max_inflight)
        app.state.admin_lock = asyncio.Lock()
        for _ in range(settings.max_inflight):
            app.state.tokens.put_nowait(True)
        pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='fairjob-v2')
        if settings.shadow is not None:
            try:
                app.state.shadow = await asyncio.to_thread(shadow_factory,settings.shadow,settings.shadow_capacity)
            except Exception as error:
                app.state.shadow_startup_error = type(error).__name__
                metrics.shadow.labels('startup_error').inc()
        async def collect():
            while True:
                if app.state.shadow:
                    for result in app.state.shadow.drain():
                        status = result['status']
                        app.state.shadow_totals[status] += 1
                        metrics.shadow.labels(status).inc()
                        metrics.shadow_pending.dec()
                        if status == 'compared':
                            metrics.shadow_latency.observe(result['elapsed_ms']/1000)
                            metrics.top1.labels(str(result['top1_agreement']).lower()).inc()
                            metrics.overlap.observe(result['topk_overlap'])
                        app.state.shadow_results.append(result)
                    # Only bounded diagnostics; aggregate counts are cumulative.
                    del app.state.shadow_results[:-100]
                await asyncio.sleep(.05)
        collector = asyncio.create_task(collect())
        app.state.pool = pool
        try:
            yield
        finally:
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass
            pool.shutdown(wait=True)
            if app.state.shadow:
                await asyncio.to_thread(app.state.shadow.close)
            app.state.deployment.close()

    app = FastAPI(title='FairJob V2 model deployment lab',lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid(_, error):
        return JSONResponse(status_code=422,content={'detail':[{k:item[k] for k in ['loc','msg','type']} for item in error.errors()]})

    # Pure ASGI middleware measures final body send, including serialization/queueing.
    class Measure:
        def __init__(self, app):
            self.app = app
        async def __call__(self, scope, receive, send):
            if scope['type']!='http' or scope.get('path')!='/rank':
                return await self.app(scope,receive,send)
            started,status = perf_counter(),500
            async def measured_send(message):
                nonlocal status
                if message['type']=='http.response.start':
                    status = message['status']
                await send(message)
            try:
                await self.app(scope,receive,measured_send)
            finally:
                metrics.requests.labels(str(status)).inc()
                metrics.duration.observe(perf_counter()-started)
    app.add_middleware(Measure)

    @app.get('/healthz')
    async def health():
        return {'status':'alive'}

    @app.get('/readyz')
    async def ready():
        active = app.state.deployment.snapshot()
        return {'ready':True,'model_version':active.bundle.metadata['model_version'],
                'metadata_sha256':sha256(active.bundle.directory/'metadata.json'),
                'redis_status':active.catalog.redis_status,
                'shadow_ready':bool(app.state.shadow and app.state.shadow.process.is_alive()),
                'shadow_startup_error':app.state.shadow_startup_error}

    @app.get('/metrics')
    async def expose():
        # Bounded labels (3 capacities x 3 rates x 2 arrivals), from completed phases only.
        replay_path = Path(os.environ.get('FAIRJOB_REPLAY_METRICS','reports/v2/load/metrics.json'))
        try:
            replay = read_json(replay_path)
            worst = {}
            for row in replay['runs']:
                labels = (str(row['capacity']),str(row['target_rps']),str(row['burst_size']))
                worst[labels] = min(worst.get(labels,1.),row['on_time_success_rate'])
            for labels,ratio in worst.items():
                metrics.replay.labels(*labels).set(ratio)
        except (FileNotFoundError,json.JSONDecodeError):
            pass  # A benchmark may be writing its next completed phase.
        return Response(generate_latest(metrics.registry),headers={'Content-Type':CONTENT_TYPE_LATEST})

    @app.get('/shadow/status')
    async def shadow_status():
        return {**app.state.shadow_totals,'recent':app.state.shadow_results[-10:]}

    @app.post('/rank',response_model=RankResponse)
    async def rank(request:RankRequest):
        try:
            app.state.tokens.get_nowait()
        except asyncio.QueueEmpty:
            raise HTTPException(503,'Ranking capacity exhausted')
        metrics.inflight.inc()
        started = perf_counter()
        active = app.state.deployment.snapshot()
        try:
            future = asyncio.get_running_loop().run_in_executor(app.state.pool,active.rank,request)
        except BaseException:
            app.state.tokens.put_nowait(True); metrics.inflight.dec()
            raise
        def release(_):
            app.state.tokens.put_nowait(True); metrics.inflight.dec()
        future.add_done_callback(release)
        try:
            result = await asyncio.shield(future)
        except ValueError as error:
            raise HTTPException(422,str(error)) from error
        result['timings_ms']['handler'] = (perf_counter()-started)*1000
        for stage,elapsed in result['timings_ms'].items():
            metrics.stages.labels(stage).observe(elapsed/1000)
        if result['degraded']:
            for reason in result['fallback_reason'].split(','):
                metrics.fallback.labels(reason).inc()
        sample = int.from_bytes(hashlib.sha256(request.request_id.encode()).digest()[:8],'big')/2**64
        if app.state.shadow and sample < settings.shadow_sample_rate:
            submitted = app.state.shadow.submit(request.model_dump(),result)
            outcome = 'submitted' if submitted else 'dropped'
            app.state.shadow_totals[outcome] += 1
            metrics.shadow.labels(outcome).inc()
            if submitted:
                metrics.shadow_pending.inc()
        return result

    def authorize(authorization):
        if not settings.admin_token or not hmac.compare_digest(authorization or '', 'Bearer '+settings.admin_token):
            raise HTTPException(403,'Deployment control is disabled or unauthorized')

    @app.post('/admin/promote')
    async def promote(body:Promotion,authorization:str|None=Header(default=None)):
        authorize(authorization)
        async with app.state.admin_lock:
            try:
                result = await asyncio.to_thread(app.state.deployment.promote,body.bundle,body.quality_evidence,body.performance_evidence,body.warmup)
            except Exception as error:
                metrics.deployments.labels('blocked').inc()
                raise HTTPException(409,str(error)) from error
            metrics.deployments.labels('promoted').inc()
            return result

    @app.post('/admin/rollback')
    async def rollback(authorization:str|None=Header(default=None)):
        authorize(authorization)
        async with app.state.admin_lock:
            try:
                result = await asyncio.to_thread(app.state.deployment.rollback)
            except Exception as error:
                raise HTTPException(409,str(error)) from error
            metrics.deployments.labels('rolled_back').inc()
            return result

    return app


def app_factory():
    return create_app()
