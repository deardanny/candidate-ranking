"""Run the project venv; on macOS reuse scikit-learn's bundled OpenMP library.

Usage: python3 scripts/run.py -m fairjob_ranker prepare
This only changes the child process environment, not system libraries.
"""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    raise SystemExit("Create .venv and install this project first; see README.md.")
environment = os.environ.copy()
if sys.platform == "darwin":
    libraries = list((root / ".venv/lib").glob("python*/site-packages/sklearn/.dylibs/libomp.dylib"))
    if len(libraries) == 1:
        old = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = str(libraries[0].parent) + (":" + old if old else "")
os.execve(str(python), [str(python), *sys.argv[1:]], environment)

