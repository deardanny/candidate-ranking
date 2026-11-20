"""Versioned Redis MGET with a verified local catalog as the fallback."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from .schema import AD_FEATURES


def redis_client(url: str, timeout: float = 0.005) -> redis.Redis:
    return redis.Redis.from_url(url, decode_responses=True, protocol=2,
                                socket_connect_timeout=timeout, socket_timeout=timeout,
                                retry=Retry(NoBackoff(), 0), max_connections=8)


class AdCatalog:
    def __init__(self, path: Path, version: str, client=None):
        frame = pd.read_parquet(path)
        if set(frame.columns) != {"product_id", *AD_FEATURES} or frame.empty:
            raise ValueError("Invalid catalog schema.")
        values = frame[["product_id", *AD_FEATURES]].to_numpy()
        if frame.product_id.duplicated().any() or not np.isfinite(values).all() or (values < 0).any() or not (values == np.floor(values)).all():
            raise ValueError("Invalid catalog IDs or categorical codes.")
        self.records = {int(row.product_id): {c: int(getattr(row, c)) for c in AD_FEATURES}
                        for row in frame.itertuples(index=False)}
        self.version, self.client = version, client
        self.redis_status = "disabled" if client is None else "unchecked"

    def key(self, product_id: int) -> str:
        return f"fairjob:catalog:{self.version}:ad:{product_id}"

    def payload(self, product_id: int) -> dict:
        return {"catalog_version": self.version, "product_id": product_id, "features": self.records[product_id]}

    def probe(self) -> None:
        if self.client is not None:
            try:
                self.redis_status = "available" if self.client.ping() else "unavailable"
            except redis.RedisError:
                self.redis_status = "unavailable"

    def lookup(self, ids: list[int]) -> tuple[list[dict], list[int], list[str], int]:
        reasons, cached = [], [None] * len(ids)
        if self.client is None:
            reasons.append("redis_disabled")
        else:
            try:
                cached = self.client.mget([self.key(i) for i in ids])
                if len(cached) != len(ids):
                    raise redis.RedisError("Unexpected MGET response size")
                self.redis_status = "available"
            except redis.RedisError:
                cached = [None] * len(ids)
                reasons.append("redis_unavailable")
                self.redis_status = "unavailable"
        records, skipped, hits = [], [], 0
        for i, value in zip(ids, cached):
            if i not in self.records:
                skipped.append(i)
                continue
            expected = self.payload(i)
            if value is not None:
                try:
                    # Local snapshot is authoritative for this immutable catalog version.
                    if json.loads(value) != expected:
                        raise ValueError("Cache payload does not match catalog version")
                    hits += 1
                except (ValueError, TypeError, UnicodeError):
                    reasons.append("cache_invalid")
            elif self.redis_status == "available":
                reasons.append("cache_miss")
            records.append({"product_id": i, **expected["features"]})
        return records, skipped, sorted(set(reasons)), hits


def seed_catalog(catalog: AdCatalog, client, batch_size: int = 500) -> int:
    """Only write this catalog's namespaced keys; never flush a Redis database."""
    ids = list(catalog.records)
    for start in range(0, len(ids), batch_size):
        with client.pipeline(transaction=False) as pipe:
            for i in ids[start:start + batch_size]:
                pipe.set(catalog.key(i), json.dumps(catalog.payload(i), separators=(",", ":")))
            pipe.execute()
    return len(ids)
