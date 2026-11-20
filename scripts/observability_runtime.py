"""Run owned, loopback-only Prometheus, Grafana and MLflow processes."""
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time

import httpx

from local_runtime import ROOT,stop_process


@contextmanager
def observability(log_dir):
    for port in [9095,3005,5005]:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',port));probe.listen(1)
    prometheus=shutil.which('prometheus') or '/opt/homebrew/bin/prometheus'
    grafana=shutil.which('grafana') or '/opt/homebrew/bin/grafana'
    if not Path(prometheus).exists() or not Path(grafana).exists():
        raise RuntimeError('Install prometheus and grafana; see docs/V2_GUIDE.md')
    home=Path(os.environ.get('GRAFANA_HOMEPATH','/opt/homebrew/opt/grafana/share/grafana'))
    state=(ROOT/'runs/v2/observability').resolve()
    state.mkdir(parents=True,exist_ok=True);log_dir.mkdir(parents=True,exist_ok=True)
    secret=state/'grafana-admin.secret'
    if not secret.exists():
        secret.write_text(secrets.token_urlsafe(24));secret.chmod(0o600)
    password=secret.read_text()
    env={**os.environ,'FAIRJOB_PROJECT_ROOT':str(ROOT),
         'GF_PATHS_DATA':str(state/'grafana-data'),'GF_PATHS_LOGS':str(log_dir.resolve()),
         'GF_PATHS_PLUGINS':str(state/'plugins'),'GF_PATHS_PROVISIONING':str(ROOT/'monitoring/provisioning'),
         'GF_SECURITY_ADMIN_PASSWORD':password,'GF_INSTALL_PLUGINS':'',
         'MLFLOW_DISABLE_TELEMETRY':'true','MLFLOW_DISABLE_AGENT_HINT':'1',
         'MPLCONFIGDIR':str(ROOT/'.tools/matplotlib')}
    # Grafana 13 packages Prometheus as a separately installed plugin.
    if not (state/'plugins/prometheus/plugin.json').exists():
        subprocess.run([grafana,'cli','--pluginsDir',str(state/'plugins'),'plugins','install','prometheus','13.2.7'],
                       cwd=ROOT,env=env,check=True)
    commands={
        'prometheus':[prometheus,'--config.file='+str(ROOT/'monitoring/prometheus.yml'),
                      '--storage.tsdb.path='+str(state/'prometheus'),'--web.listen-address=127.0.0.1:9095'],
        'grafana':[grafana,'server','--homepath',str(home),'--config',str(ROOT/'monitoring/grafana.ini')],
        'mlflow':[sys.executable,'-m','mlflow','server','--backend-store-uri','sqlite:///'+str(ROOT/'runs/v2/mlflow.db'),
                  '--host','127.0.0.1','--port','5005','--workers','1'],
    }
    processes,logs={},[]
    try:
        for name,command in commands.items():
            log=(log_dir/(name+'.log')).open('a');logs.append(log)
            processes[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        urls={'prometheus':'http://127.0.0.1:9095/-/ready','grafana':'http://127.0.0.1:3005/api/health','mlflow':'http://127.0.0.1:5005/health'}
        with httpx.Client(timeout=2,trust_env=False) as client:
            for name,url in urls.items():
                deadline=time.monotonic()+45
                while True:
                    if processes[name].poll() is not None:raise RuntimeError(f'{name} exited; see {log_dir}')
                    try:
                        if client.get(url).status_code==200:break
                    except httpx.HTTPError:pass
                    if time.monotonic()>deadline:raise TimeoutError(f'{name} not ready')
                    time.sleep(.2)
        yield {'processes':processes,'grafana_password':password}
    finally:
        for process in reversed(list(processes.values())):stop_process(process)
        for log in logs:log.close()
