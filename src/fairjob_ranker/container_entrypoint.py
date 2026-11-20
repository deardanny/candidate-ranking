"""Populate this immutable catalog namespace, then run the API as PID 1."""
import logging
import os
from pathlib import Path
import sys

import redis

from .catalog import AdCatalog, redis_client, seed_catalog
from .torch_model import load_bundle


def main():
    directory = Path(os.environ.get("FAIRJOB_MODEL_DIR", "/model"))
    bundle = load_bundle(directory)
    url = os.environ.get("FAIRJOB_REDIS_URL")
    if url:
        client = redis_client(url, timeout=2)
        try:
            catalog = AdCatalog(directory / "ad_catalog.parquet", bundle.metadata["catalog_version"])
            print(f"Seeded {seed_catalog(catalog, client)} ads", flush=True)
        except redis.RedisError:
            logging.warning("Redis unavailable at startup; local catalog fallback remains enabled")
        finally:
            client.close()
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "fairjob_ranker.serving:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
