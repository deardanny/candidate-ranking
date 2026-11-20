"""Verify real HTTP/Redis and offline parity; this is not a load benchmark."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from fairjob_ranker.catalog import redis_client
from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.torch_model import load_bundle
from local_runtime import local_stack, stop_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/pytorch-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/week2/serving_verification.json"))
    parser.add_argument("--api-port", type=int, default=8001)
    parser.add_argument("--redis-port", type=int, default=6381)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use another --output")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("walkthrough", root / "examples/week2_walkthrough.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    valid = pd.read_parquet(root / "data/processed/fairjob-v1/validation.parquet")
    bundle = load_bundle(args.model)
    model_scores = bundle.predict(valid)
    scored = valid.assign(offline_score=model_scores)
    ids = [20562] + [int(i) for i in valid.impression_id.unique() if i != 20562][:99]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False,
              "model_version": bundle.metadata["model_version"], "requests": len(ids),
              "atol": 1e-6, "rtol": 1e-5, "max_abs_error": 0.0, "cache_hits": 0,
              "note": "Functional verification on validation requests, not a throughput or latency benchmark.",
              "source_code_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))}}
    with local_stack(args.model, args.api_port, args.redis_port, root / "runs/verify-week2") as stack:
        report["seeded_ads"] = stack.seeded_ads
        with httpx.Client(base_url=stack.api_url, timeout=10, trust_env=False) as client:
            for i in ids:
                frame = scored.loc[scored.impression_id.eq(i)]
                request = example.make_request(frame)
                result = client.post("/rank", json=request)
                result.raise_for_status()
                response = result.json()
                assert not response["degraded"], response
                assert response["scoring_method"] == "pytorch"
                assert response["cache_hits"] == len(frame)
                lookup = frame.set_index("product_id").offline_score
                actual = [r["score"] for r in response["rankings"]]
                expected = [lookup[r["product_id"]] for r in response["rankings"]]
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                report["max_abs_error"] = max(report["max_abs_error"], float(np.max(np.abs(np.array(actual) - expected))))
                report["cache_hits"] += response["cache_hits"]
                reverse = {**request, "candidate_ids": list(reversed(request["candidate_ids"]))}
                reordered = client.post("/rank", json=reverse)
                reordered.raise_for_status()
                assert [r["product_id"] for r in response["rankings"]] == [r["product_id"] for r in reordered.json()["rankings"]]
                if i == 20562:
                    report["example_request"], report["example_response"] = request, response
            original = report["example_response"]
            request = report["example_request"]
            connection = redis_client(stack.redis_url, timeout=2)
            try:
                report["redis_version"] = connection.info("server")["redis_version"]
            finally:
                connection.close()
            stop_process(stack.redis_process)
            response = client.post("/rank", json=request)
            response.raise_for_status()
            fallback = response.json()
            assert fallback["degraded"] and "redis_unavailable" in fallback["fallback_reason"]
            assert fallback["rankings"] == original["rankings"]
            report["redis_stopped_same_scores"] = True
            report["redis_stopped_response"] = fallback
            report["readiness_after_redis_stop"] = client.get("/readyz").json()
            assert report["readiness_after_redis_stop"]["ready"]
            bad = {**request, "candidate_ids": [request["candidate_ids"][0]] * 2, "top_k": 2}
            report["duplicate_ids_status"] = client.post("/rank", json=bad).status_code
            assert report["duplicate_ids_status"] == 422
            report["metrics"] = client.get("/metrics").json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"PASS: {len(ids)} validation impressions, offline/API max error={report['max_abs_error']:.3g}, Redis shutdown fallback verified.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
