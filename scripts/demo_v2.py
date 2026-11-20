"""Start the V2 API and all three local UIs; stop owned processes on Ctrl+C."""
import argparse
from pathlib import Path
import time

import httpx

from fairjob_ranker.io import read_json
from fairjob_ranker.replay import load_requests
from local_runtime import local_stack
from observability_runtime import observability


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration',type=float,default=0,help='0 keeps the demo open until Ctrl+C')
    args=parser.parse_args()
    experiment=read_json(Path('reports/v2/experiment/metrics.json'))
    requests,_=load_requests(Path('data/replay/validation.jsonl'))
    print('기준 모델:',experiment['champion']['name'],flush=True)
    print('새 후보:',experiment['challenger']['name'],flush=True)
    print('품질 교체 조건:',experiment['quality_gate']['checks'],flush=True)
    env={'FAIRJOB_ADMIN_TOKEN':'local-demo-only','MLFLOW_DISABLE_AGENT_HINT':'1','MLFLOW_DISABLE_TELEMETRY':'true'}
    selected=Path('configs/v2_serving_selected.json')
    if selected.exists():
        env['FAIRJOB_MAX_INFLIGHT']=str(read_json(selected)['max_inflight'])
    with local_stack(Path('models/pytorch-v1'),8033,6399,Path('runs/v2/demo/logs/api'),environment_overrides=env,api_app='v2_server:app') as stack:
        with observability(Path('runs/v2/demo/logs')):
            with httpx.Client(base_url=stack.api_url,timeout=10,trust_env=False) as client:
                for request in requests[:30]:
                    response=client.post('/rank',json=request);response.raise_for_status()
                print('실제 API 순위:',response.json()['rankings'],flush=True)
                candidate=experiment['challenger']
                blocked=client.post('/admin/promote',headers={'Authorization':'Bearer '+env['FAIRJOB_ADMIN_TOKEN']},json={
                    'bundle':candidate['bundle'],'quality_evidence':'reports/v2/experiment/quality_gate.json',
                    'performance_evidence':'reports/v2/verification/candidate_performance.json','warmup':requests[0]})
                assert blocked.status_code==409 and 'quality_gate_failed' in blocked.text
                print('품질 미달 후보 교체 차단:',blocked.json(),flush=True)
                print('API http://127.0.0.1:8033/docs',flush=True)
                print('MLflow http://127.0.0.1:5005',flush=True)
                print('Grafana http://127.0.0.1:3005/d/fairjob-v2',flush=True)
                print('Prometheus http://127.0.0.1:9095',flush=True)
                print('Ctrl+C로 이번 명령의 서버들을 함께 종료합니다.',flush=True)
                began=time.monotonic();i=30
                try:
                    while not args.duration or time.monotonic()-began<args.duration:
                        client.post('/rank',json=requests[i%len(requests)]).raise_for_status()
                        i+=1;time.sleep(.2)
                except KeyboardInterrupt:
                    pass


if __name__=='__main__':main()
