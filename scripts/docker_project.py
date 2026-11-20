"""Run Docker using this project's CLI config and optional named Colima runtime."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
config = root / ".tools/docker-config"
if config.exists():
    environment["DOCKER_CONFIG"] = str(config)
socket = Path.home() / ".colima/adsranking/docker.sock"
if "DOCKER_HOST" not in environment and socket.exists():
    environment["DOCKER_HOST"] = "unix://" + str(socket)
docker = shutil.which("docker") or "/opt/homebrew/bin/docker"
if len(sys.argv) > 1 and sys.argv[1] in ('start-runtime', 'stop-runtime'):
    colima = shutil.which('colima') or '/opt/homebrew/bin/colima'
    if sys.argv[1] == 'start-runtime':
        command = [colima, 'start', 'adsranking', '--cpus', '2', '--memory', '4', '--disk', '20',
                   '--root-disk', '10', '--vm-type', 'vz', '--runtime', 'docker',
                   '--mount', str(root)+':w', '--ssh-config=false', '--activate=false']
    else:
        command = [colima, 'stop', 'adsranking']
    raise SystemExit(subprocess.call(command, cwd=root, env=environment))
raise SystemExit(subprocess.call([docker, *sys.argv[1:]], cwd=root, env=environment))
