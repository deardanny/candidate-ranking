"""Own a local Redis/API pair and stop only the child processes started here."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import httpx
from fairjob_ranker.catalog import AdCatalog, redis_client, seed_catalog
from fairjob_ranker.torch_model import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@dataclass
class LocalStack:
    redis_process: subprocess.Popen
    api_process: subprocess.Popen
    api_url: str
    redis_url: str
    seeded_ads: int


@contextmanager
def local_stack(model_dir: Path, api_port: int, redis_port: int, log_dir: Path,
                environment_overrides: dict | None = None, uvicorn_workers: int = 1,
                api_app: str = "fairjob_ranker.serving:app"):
    # Abort before writing Redis keys if either endpoint belongs to another process.
    for port in [api_port, redis_port]:
        with socket.socket() as probe:
            # A just-stopped server can leave TIME_WAIT sockets; active listeners still fail.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
            probe.listen(1)
    local = ROOT / ".tools/redis-7.2.12/src/redis-server"
    binary = str(local) if local.exists() else shutil.which("redis-server")
    if not binary:
        raise RuntimeError("Run python3 scripts/setup_local_redis.py first, or install redis-server")
    bundle = load_bundle(model_dir)
    catalog = AdCatalog(model_dir / "ad_catalog.parquet", bundle.metadata["catalog_version"])
    redis_url, api_url = f"redis://127.0.0.1:{redis_port}/0", f"http://127.0.0.1:{api_port}"
    log_dir.mkdir(parents=True, exist_ok=True)
    redis_process = api_process = None
    with (log_dir / "redis.log").open("a") as redis_log, (log_dir / "api.log").open("a") as api_log:
        try:
            redis_process = subprocess.Popen([binary, "--bind", "127.0.0.1", "--port", str(redis_port),
                                               "--save", "", "--appendonly", "no", "--protected-mode", "yes"],
                                              cwd=ROOT, stdout=redis_log, stderr=subprocess.STDOUT)
            connection = redis_client(redis_url, timeout=2)
            try:
                deadline = time.monotonic() + 15
                while True:
                    if redis_process.poll() is not None:
                        raise RuntimeError(f"Redis startup failed; see {log_dir / 'redis.log'}")
                    try:
                        connection.ping()
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Redis did not become ready")
                        time.sleep(0.1)
                count = seed_catalog(catalog, connection)
            finally:
                connection.close()
            environment = os.environ.copy()
            environment.update(FAIRJOB_MODEL_DIR=str(model_dir.resolve()), FAIRJOB_REDIS_URL=redis_url)
            environment.update(environment_overrides or {})
            api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", api_app,
                                             "--app-dir", str(ROOT / "scripts"), "--workers", str(uvicorn_workers),
                                             "--host", "127.0.0.1", "--port", str(api_port)],
                                            cwd=ROOT, env=environment, stdout=api_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            with httpx.Client(base_url=api_url, timeout=2, trust_env=False) as client:
                while True:
                    if api_process.poll() is not None:
                        raise RuntimeError(f"API startup failed; see {log_dir / 'api.log'}")
                    try:
                        response = client.get("/readyz")
                        if response.status_code == 200 and response.json()["ready"]:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"API did not become ready; see {log_dir / 'api.log'}")
                    time.sleep(0.1)
            yield LocalStack(redis_process, api_process, api_url, redis_url, count)
        finally:
            stop_process(api_process)
            stop_process(redis_process)
