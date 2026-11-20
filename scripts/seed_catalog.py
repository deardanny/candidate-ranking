"""Populate only the selected bundle's versioned catalog namespace in Redis."""
import argparse
from pathlib import Path

from fairjob_ranker.catalog import AdCatalog, redis_client, seed_catalog
from fairjob_ranker.torch_model import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/pytorch-v1"))
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6380/0")
    args = parser.parse_args()
    bundle = load_bundle(args.model)
    client = redis_client(args.redis_url, timeout=2.0)
    try:
        catalog = AdCatalog(args.model / "ad_catalog.parquet", bundle.metadata["catalog_version"])
        print(f"Seeded {seed_catalog(catalog, client):,} ads; catalog={catalog.version}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
