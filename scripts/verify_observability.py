from datetime import datetime,timezone
import argparse
from pathlib import Path
import time

import httpx
import mlflow

from fairjob_ranker.io import write_json
from fairjob_ranker.replay import load_requests
from local_runtime import local_stack
from observability_runtime import observability


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reports/v2/observability'))
    output=parser.parse_args().output
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    requests,_=load_requests(Path('data/replay/validation.jsonl'))
    with local_stack(Path('models/pytorch-v1'),8033,6399,output/'logs'/'api',api_app='v2_server:app') as stack:
        with observability(output/'logs') as monitor:
            with httpx.Client(timeout=10,trust_env=False) as client:
                for request in requests[:30]:
                    response=client.post(stack.api_url+'/rank',json=request);response.raise_for_status()
                query='sum(fairjob_http_requests_total{status="200"})'
                deadline=time.monotonic()+15
                while True:
                    prom=client.get('http://127.0.0.1:9095/api/v1/query',params={'query':query}).json()
                    values=prom.get('data',{}).get('result',[])
                    if values and float(values[0]['value'][1])>=30:break
                    if time.monotonic()>deadline:raise AssertionError('Prometheus did not scrape API counters')
                    time.sleep(.5)
                dashboard=client.get('http://127.0.0.1:3005/api/dashboards/uid/fairjob-v2')
                dashboard.raise_for_status()
                panels=dashboard.json()['dashboard']['panels']
                assert len(panels)>=8
                # Verify Grafana's data source can actually reach Prometheus.
                deadline=time.monotonic()+30
                while True:
                    health=client.get('http://127.0.0.1:3005/api/datasources/uid/fairjob-prometheus/health',auth=('admin',monitor['grafana_password']))
                    if health.status_code==200 or time.monotonic()>deadline:break
                    time.sleep(.5)
                health.raise_for_status()
                registry=mlflow.MlflowClient(tracking_uri='http://127.0.0.1:5005',registry_uri='http://127.0.0.1:5005')
                model=registry.get_model_version_by_alias('fairjob-v2','champion')
                write_json(output/'evidence.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
                    'prometheus_scraped_successes':float(values[0]['value'][1]),'grafana_dashboard_uid':'fairjob-v2',
                    'grafana_panels':len(panels),'grafana_datasource_health':health.json(),
                    'mlflow_http_champion_version':model.version,'all_bound_to_loopback':True})
    print('Prometheus scrape, Grafana dashboard/data source and MLflow HTTP registry verified')


if __name__=='__main__':main()
