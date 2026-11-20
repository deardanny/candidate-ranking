"""FastAPI ranking service: strict inputs, shared features, bounded CPU execution."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch

from .catalog import AdCatalog, redis_client
from .schema import CATEGORICAL, FEATURES, NUMERICAL, VERSION
from .torch_model import load_bundle, predict_tensors, to_tensors

logger = logging.getLogger("fairjob.serving")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False
Code = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RequestContext(StrictModel):
    cat0: Code
    cat1: Code
    numeric: dict[str, float]

    @field_validator("numeric", mode="before")
    @classmethod
    def validate_numeric(cls, values):
        if not isinstance(values, dict) or set(values) != set(NUMERICAL):
            raise ValueError("numeric must contain exactly num16 through num50")
        try:
            valid = all(type(v) in (int, float) and math.isfinite(v) for v in values.values())
        except (TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("numeric values must be finite numbers")
        return values


class RankRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    schema_version: Literal["fairjob-v1"]
    context: RequestContext
    candidate_ids: list[Code] = Field(min_length=1, max_length=64)
    top_k: int = Field(strict=True, ge=1, le=64)

    @model_validator(mode="after")
    def check_candidates(self):
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.top_k > len(self.candidate_ids):
            raise ValueError("top_k cannot exceed candidate count")
        return self


class RankedAd(StrictModel):
    product_id: int
    score: float = Field(ge=0, le=1)
    position: int = Field(ge=1)


class RankResponse(StrictModel):
    request_id: str
    schema_version: str
    model_version: str
    attempted_model_version: str
    feature_version: str
    catalog_version: str
    scoring_method: str
    degraded: bool
    fallback_reason: str | None
    skipped_candidate_ids: list[int]
    cache_hits: int
    oov_categories: int
    rankings: list[RankedAd]
    timings_ms: dict[str, float]


@dataclass
class Settings:
    model_dir: Path = Path("models/pytorch-v1")
    redis_url: str | None = "redis://127.0.0.1:6380/0"
    redis_timeout: float = 0.005
    torch_threads: int = 1
    max_workers: int = 1
    max_inflight: int = 2
    batch_candidates: bool = True

    def __post_init__(self):
        if min(self.torch_threads, self.max_workers, self.max_inflight) < 1 or self.redis_timeout <= 0:
            raise ValueError("Service limits must be positive.")

    @classmethod
    def from_env(cls):
        return cls(model_dir=Path(os.environ.get("FAIRJOB_MODEL_DIR", "models/pytorch-v1")),
                   redis_url=os.environ.get("FAIRJOB_REDIS_URL", "redis://127.0.0.1:6380/0") or None,
                   redis_timeout=float(os.environ.get("FAIRJOB_REDIS_TIMEOUT", "0.005")),
                   torch_threads=int(os.environ.get("FAIRJOB_TORCH_THREADS", "1")),
                   max_workers=int(os.environ.get("FAIRJOB_MAX_WORKERS", "1")),
                   max_inflight=int(os.environ.get("FAIRJOB_MAX_INFLIGHT", "2")),
                   batch_candidates=os.environ.get("FAIRJOB_BATCH_CANDIDATES", "1") == "1")


class RankService:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        torch.set_num_threads(settings.torch_threads)
        self.bundle = load_bundle(settings.model_dir)
        self.client = client if client is not None else (redis_client(settings.redis_url, settings.redis_timeout) if settings.redis_url else None)
        self.catalog = AdCatalog(settings.model_dir / "ad_catalog.parquet", self.bundle.metadata["catalog_version"], self.client)
        self.catalog.probe()
        self.counters, self.counter_lock = Counter(), Lock()
        # Warm up the same tensor path that requests use.
        c = torch.zeros((1, len(CATEGORICAL)), dtype=torch.int64)
        n = torch.zeros((1, len(NUMERICAL)), dtype=torch.float32)
        if not np.isfinite(predict_tensors(self.bundle.model, c, n)).all():
            raise ValueError("Model warm-up returned nonfinite scores.")

    def rank(self, request: RankRequest) -> dict:
        started = perf_counter()
        records, skipped, reasons, hits = self.catalog.lookup(request.candidate_ids)
        if not records:
            raise ValueError("No known candidates remain.")
        lookup_end = perf_counter()
        context = {"cat0": request.context.cat0, "cat1": request.context.cat1, **request.context.numeric}
        raw = pd.DataFrame([{**record, **context} for record in records])
        transformed = self.bundle.transformer.transform(raw[FEATURES])
        categorical, numerical = to_tensors(transformed)
        oov = int(transformed[CATEGORICAL].eq(0).to_numpy().sum())
        transform_end = perf_counter()
        method, model_version = "pytorch", self.bundle.metadata["model_version"]
        try:
            scores = predict_tensors(self.bundle.model, categorical, numerical,
                                     batch_size=8192 if self.settings.batch_candidates else 1)
            if scores.shape != (len(records),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError("Invalid model scores")
        except Exception as error:
            # A complete alternative ordering, never a mixture of two scoring methods.
            reasons.append("model_inference_error")
            method = "smoothed_ad_ctr"
            model_version = method + ":" + self.bundle.metadata["artifact_sha256"]["fallback.json"][:12]
            scores = np.array([self.bundle.fallback["scores"].get(str(r["product_id"]), self.bundle.fallback["train_ctr"]) for r in records])
            logger.warning("model_inference_error type=%s", type(error).__name__)
        infer_end = perf_counter()
        ranked = sorted(zip(records, scores), key=lambda item: (-float(item[1]), item[0]["product_id"]))
        rankings = [{"product_id": r["product_id"], "score": float(s), "position": i + 1}
                    for i, (r, s) in enumerate(ranked[:request.top_k])]
        timings = {"lookup": (lookup_end - started) * 1000, "transform": (transform_end - lookup_end) * 1000,
                   "inference": (infer_end - transform_end) * 1000, "sort": (perf_counter() - infer_end) * 1000,
                   "service": (perf_counter() - started) * 1000}
        reasons = sorted(set(reasons))
        response = {"request_id": request.request_id, "schema_version": VERSION,
                    "model_version": model_version, "attempted_model_version": self.bundle.metadata["model_version"],
                    "feature_version": self.bundle.metadata["feature_version"], "catalog_version": self.catalog.version,
                    "scoring_method": method, "degraded": bool(reasons), "fallback_reason": ",".join(reasons) or None,
                    "skipped_candidate_ids": sorted(skipped), "cache_hits": hits, "oov_categories": oov,
                    "rankings": rankings, "timings_ms": timings}
        with self.counter_lock:
            self.counters.update(requests=1, candidates=len(records), cache_hits=hits,
                                 oov_categories=oov, degraded_requests=int(bool(reasons)),
                                 model_fallback_requests=int(method != "pytorch"))
        logger.info(json.dumps({"request_id": request.request_id, "model_version": model_version,
                               "candidates": len(records), "degraded": bool(reasons),
                               "fallback_reason": response["fallback_reason"], "timings_ms": timings}))
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service, app.state.startup_error = None, None
        app.state.tokens = asyncio.Queue(maxsize=settings.max_inflight)
        app.state.rejected_overload = 0
        for _ in range(settings.max_inflight):
            app.state.tokens.put_nowait(True)
        pool = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="fairjob")
        app.state.pool = pool
        try:
            app.state.service = RankService(settings)
        except Exception as error:
            app.state.startup_error = type(error).__name__
            logger.error("Model startup failed: %s", type(error).__name__)
        try:
            yield
        finally:
            pool.shutdown(wait=True)
            if app.state.service is not None and app.state.service.client is not None:
                app.state.service.client.close()

    app = FastAPI(title="FairJob Ads Ranking", version="0.2.0", lifespan=lifespan,
                  description="Rank supplied candidates using anonymized FairJob click data.")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_, error):
        # Nonfinite numbers and ValueError objects cannot be serialized as JSON inputs/context.
        return JSONResponse(status_code=422, content={"detail": [
            {key: item[key] for key in ["loc", "msg", "type"]} for item in error.errors()]})

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready():
        service = app.state.service
        if service is None:
            return JSONResponse(status_code=503, content={"ready": False, "reason": "model_bundle_unavailable"})
        return {"ready": True, "model_version": service.bundle.metadata["model_version"],
                "redis_status": service.catalog.redis_status,
                "degraded": service.catalog.redis_status != "available"}

    @app.get("/metrics")
    async def metrics():
        service = app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        with service.counter_lock:
            snapshot = dict(service.counters)
        return {**snapshot, "rejected_overload": app.state.rejected_overload}

    @app.post("/rank", response_model=RankResponse)
    async def rank(request: RankRequest):
        service = app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            app.state.tokens.get_nowait()
        except asyncio.QueueEmpty:
            app.state.rejected_overload += 1
            raise HTTPException(status_code=503, detail="Ranking capacity exhausted") from None
        received = perf_counter()
        try:
            future = asyncio.get_running_loop().run_in_executor(app.state.pool, service.rank, request)
        except Exception:
            app.state.tokens.put_nowait(True)
            raise
        # Release only when CPU work finishes, even if the HTTP caller disconnects.
        future.add_done_callback(lambda _: app.state.tokens.put_nowait(True))
        try:
            response = await asyncio.shield(future)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response["timings_ms"]["handler"] = (perf_counter() - received) * 1000
        return response

    return app


app = create_app()
