"""Run an open-loop sweep against an owned native API/Redis pair."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.loadtest import environment, run_phase
from fairjob_ranker.replay import load_requests
from local_runtime import local_stack, stop_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/replay/validation.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/pytorch-v1"))
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[10, 50, 100])
    parser.add_argument("--warmup", type=float, default=60)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--redis-port", type=int, default=6390)
    parser.add_argument("--redis-down", action="store_true")
    parser.add_argument("--stress-64", action="store_true")
    parser.add_argument("--fault", choices=["none", "model_error", "slow_model"], default="none")
    parser.add_argument("--legacy-features", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    requests, corpus = load_requests(args.corpus)
    settings = read_json(args.settings)
    if args.legacy_features and (corpus['split'] != 'validation' or args.fault != 'none'):
        raise ValueError('Legacy feature comparison is validation-only and cannot combine faults')
    if corpus['split'] == 'test':
        from fairjob_ranker.final_evaluation import verify_frozen
        if args.frozen is None or sha256(args.frozen) != corpus['frozen_experiment_sha256']:
            raise ValueError('Test load measurement requires the matching frozen experiment')
        frozen = verify_frozen(args.frozen)
        if sha256(args.settings) != frozen['serving_settings_sha256']:
            raise ValueError('Test measurement cannot change the frozen settings')
    if args.stress_64:
        import pandas as pd
        ids = pd.read_parquet(args.model / "ad_catalog.parquet", columns=["product_id"]).product_id.tolist()[:64]
        requests = [{**r, "candidate_ids": ids, "top_k": 64} for r in requests]
    overrides = {k: str(v) for k, v in settings["environment"].items()}
    if args.fault != "none":
        overrides["FAIRJOB_FAILURE_MODE"] = args.fault
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
              "corpus": corpus, "settings": settings, "warmup_seconds": args.warmup,
              "redis_down": args.redis_down, "stress_64": args.stress_64, "fault": args.fault,
              "legacy_features": args.legacy_features,
              "model_metadata_sha256": sha256(args.model / "metadata.json"), "runs": [],
              "latency_definition": "scheduled request time to fully received HTTP response; includes generator lag and queueing",
              "source_code_sha256": {str(p): sha256(p) for p in sorted(Path('src').rglob('*.py'))}}
    args.output.mkdir(parents=True)
    write_json(args.output / "running.json", result)
    with local_stack(args.model, args.api_port, args.redis_port, args.output / "logs",
                     environment_overrides=overrides, uvicorn_workers=settings.get("uvicorn_workers", 1),
                     api_app="legacy_server:app" if args.legacy_features else ("fault_server:app" if args.fault != "none" else "fairjob_ranker.serving:app")) as stack:
        if args.redis_down:
            stop_process(stack.redis_process)
        pids = [stack.api_process.pid, stack.redis_process.pid]
        for rate in args.rates:
            print(f"rate={rate:g}: warmup {args.warmup:g}s", flush=True)
            if args.warmup:
                asyncio.run(run_phase(stack.api_url, requests, rate, args.warmup, pids=pids))
            print(f"rate={rate:g}: measuring {args.duration:g}s", flush=True)
            measured = asyncio.run(run_phase(stack.api_url, requests, rate, args.duration, pids=pids))
            records = measured.pop("records")
            with (args.output / f"requests-{rate:g}.jsonl").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            result["runs"].append(measured)
            write_json(args.output / "metrics.json", result)
            print(f"rate={rate:g}: successful_qps={measured['successful_qps']:.2f}, p95={measured['latency_success_ms']['p95']}, failures={measured['failure_rate']:.4%}", flush=True)
    (args.output / "running.json").unlink()


if __name__ == "__main__":
    main()
