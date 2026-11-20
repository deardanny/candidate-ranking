"""Local-only Redis/HTTP stack; owns and cleans up only its child processes."""
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import httpx
from local_runtime import stop_process

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def stack(bundle,shadow=None,api_port=8001,redis_port=6381,registry=True):
    for port in [api_port,redis_port]:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); probe.bind(('127.0.0.1',port)); probe.listen(1)
    folder=ROOT/'runs/ebnerd/serving'; folder.mkdir(parents=True,exist_ok=True)
    redis_process=api_process=None; token=secrets.token_urlsafe(24)
    with (folder/'redis.log').open('a') as rl,(folder/'api.log').open('a') as al:
        try:
            redis_process=subprocess.Popen([str(ROOT/'.tools/redis-7.2.12/src/redis-server'),'--bind','127.0.0.1','--port',str(redis_port),
                '--save','','--appendonly','no','--protected-mode','yes'],stdout=rl,stderr=subprocess.STDOUT)
            redis_url=f'redis://127.0.0.1:{redis_port}/0'
            from fairjob_ranker.catalog import redis_client
            connection=redis_client(redis_url,timeout=1); deadline=time.monotonic()+15
            while True:
                try: connection.ping(); break
                except Exception:
                    if redis_process.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Redis startup failed')
                    time.sleep(.1)
            connection.close()
            env=os.environ.copy(); env.update(EBNERD_BUNDLE=str(Path(bundle).resolve()),
                EBNERD_STATE=str(ROOT/'data/processed/ebnerd-v1/validation_state'),EBNERD_REDIS_URL=redis_url,
                EBNERD_SEED_REDIS='1',EBNERD_ADMIN_TOKEN=token,MLFLOW_DISABLE_TELEMETRY='true',MLFLOW_DISABLE_AGENT_HINT='1')
            if shadow: env['EBNERD_SHADOW']=str(Path(shadow).resolve())
            if registry: env['EBNERD_REGISTRY_JSON']=str(ROOT/'models/ebnerd-v1/registry.json')
            api_process=subprocess.Popen([sys.executable,'-m','uvicorn','fairjob_ranker.ebnerd.service:app_factory','--factory',
                '--host','127.0.0.1','--port',str(api_port),'--no-access-log'],env=env,stdout=al,stderr=subprocess.STDOUT)
            api_url=f'http://127.0.0.1:{api_port}'; deadline=time.monotonic()+90
            with httpx.Client(base_url=api_url,timeout=2,trust_env=False) as client:
                while True:
                    if api_process.poll() is not None: raise RuntimeError('API startup failed; see runs/ebnerd/serving/api.log')
                    try:
                        response=client.get('/readyz')
                        if response.status_code==200 and response.json()['ready']: break
                    except httpx.HTTPError: pass
                    if time.monotonic()>deadline: raise TimeoutError('API readiness timed out')
                    time.sleep(.2)
            yield {'api_url':api_url,'redis_url':redis_url,'api_process':api_process,'redis_process':redis_process,'admin_token':token}
        finally:
            stop_process(api_process); stop_process(redis_process)
