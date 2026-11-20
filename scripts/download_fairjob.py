"""Download the pinned official FairJob release and verify its published checksum."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REVISION = "941b0913be47e8a1600effadf8a8762b27182680"
SHA256 = "defeacb5233e26edd4679e2097d3582bc7858620ae301528675f78131a35c18c"
SIZE = 190_748_253
BASE = f"https://huggingface.co/datasets/criteo/FairJob/resolve/{REVISION}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "3", "--connect-timeout", "20", "--max-time", "900",
        "--output", str(temporary), url,
    ], check=True)
    return temporary


def main() -> None:
    path = ROOT / "data/raw/fairjob.csv.gz"
    if not path.exists():
        print(f"Downloading FairJob ({SIZE:,} bytes)...", flush=True)
        temporary = download(f"{BASE}/fairjob.csv.gz", path)
        if temporary.stat().st_size != SIZE or sha256(temporary) != SHA256:
            raise RuntimeError("Downloaded data failed size/SHA-256 verification; .part retained.")
        temporary.replace(path)
    elif path.stat().st_size != SIZE or sha256(path) != SHA256:
        raise RuntimeError("Existing data differs from the pinned release; refusing to overwrite.")

    source = ROOT / "data/source/FairJob.dataset-card.md"
    if not source.exists():
        download(f"{BASE}/README.md", source).replace(source)
    manifest = {
        "dataset": "criteo/FairJob", "revision": REVISION,
        "source_url": f"{BASE}/fairjob.csv.gz", "source_page": "https://huggingface.co/datasets/criteo/FairJob",
        "local_path": "data/raw/fairjob.csv.gz", "size_bytes": path.stat().st_size,
        "sha256": sha256(path), "expected_sha256": SHA256,
        "license": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    source.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"path": str(path), "size_bytes": SIZE, "sha256_verified": True}), flush=True)


if __name__ == "__main__":
    main()
