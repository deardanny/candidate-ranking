"""Start a local demo; Ctrl+C stops both owned processes. No Docker required."""
import argparse
from pathlib import Path
import subprocess
from local_runtime import local_stack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/pytorch-v1"))
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--redis-port", type=int, default=6380)
    args = parser.parse_args()
    try:
        with local_stack(args.model, args.api_port, args.redis_port, Path("runs/local")) as stack:
            print(f"Redis: {stack.seeded_ads:,} ads loaded", flush=True)
            print(f"API: {stack.api_url}; Swagger: {stack.api_url}/docs", flush=True)
            print("Ctrl+C stops the API and Redis started by this command.", flush=True)
            while True:
                try:
                    code = stack.api_process.wait(timeout=1)
                    raise RuntimeError(f"API exited with status {code}")
                except subprocess.TimeoutExpired:
                    if stack.redis_process.poll() is not None:
                        print("Redis exited; the API is using its local catalog fallback.", flush=True)
                        stack.api_process.wait()
    except KeyboardInterrupt:
        print("Stopped the local demo.", flush=True)


if __name__ == "__main__":
    main()
