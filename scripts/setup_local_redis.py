"""Build a pinned Redis locally without changing system packages (requires cc/make)."""
import hashlib
from pathlib import Path
import subprocess
import tarfile

VERSION = "7.2.12"
SHA256 = "97c60478a7c777ac914ca9d87a7e88ba265926456107e758c62d8f971d0196bc"
ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / ".tools"
    target.mkdir(exist_ok=True)
    archive = target / f"redis-{VERSION}.tar.gz"
    if not archive.exists():
        subprocess.run(["curl", "--fail", "--location", "--connect-timeout", "10", "--max-time", "120",
                        f"https://download.redis.io/releases/redis-{VERSION}.tar.gz", "-o", str(archive)], check=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Redis source archive checksum mismatch")
    source = target / f"redis-{VERSION}"
    if not source.exists():
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                destination = (target / member.name).resolve()
                if target.resolve() not in destination.parents or member.issym() or member.islnk():
                    raise ValueError("Unexpected archive member")
            stream.extractall(target)
    with (target / "redis-build.log").open("w") as log:
        subprocess.run(["make", "-C", str(source), "-j", "4", "MALLOC=libc"], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(source / "src/redis-server"), "--version"], check=True)


if __name__ == "__main__":
    main()
