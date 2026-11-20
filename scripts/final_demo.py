"""One command for the five-step final demonstration; owned servers stop on exit."""
from pathlib import Path

import httpx

from fairjob_ranker.final_evaluation import verify_frozen
from fairjob_ranker.io import read_json
from local_runtime import local_stack, stop_process


def main():
    frozen = verify_frozen(Path('reports/frozen_experiment.json'))
    example = read_json(Path('reports/final/demo_example.json'))
    print('1. 실제 test 노출의 컨텍스트와 후보를 사용합니다. 정답 click·과거 rank는 요청에 없습니다.')
    context = example['request']['context']
    print(f"   컨텍스트: cat0={context['cat0']}, cat1={context['cat1']}, num28={context['numeric']['num28']:.4f}, num36={context['numeric']['num36']:.4f} (숫자 피처 총 35개)")
    print('   후보:', example['request']['candidate_ids'])
    print('2. 같은 후보의 두 모델 점수를 비교합니다.')
    for row in example['rows']:
        print(f"   광고 {row['product_id']}: LightGBM={row['lightgbm_score']:.7f}, PyTorch={row['pytorch_score']:.7f}, 관측 클릭={row['click']}")
    settings = frozen['serving_settings']
    with local_stack(Path(frozen['pytorch_dir']), 8020, 6395, Path('runs/final-demo'),
                     environment_overrides=settings['environment'], uvicorn_workers=settings['uvicorn_workers']) as stack:
        with httpx.Client(base_url=stack.api_url, timeout=5, trust_env=False) as client:
            result = client.post('/rank',json=example['request'])
            result.raise_for_status()
            normal = result.json()
            print('3. API 응답:', normal['model_version'])
            print('   순서:', [r['product_id'] for r in normal['rankings']])
            print(f"   이번 요청의 handler 시간: {normal['timings_ms']['handler']:.2f}ms (부하 측정 p95와 다름)")
            stop_process(stack.redis_process)
            result = client.post('/rank',json=example['request'])
            result.raise_for_status()
            fallback = result.json()
            assert fallback['rankings'] == normal['rankings'] and fallback['degraded']
            print('4. Redis 중단 후 같은 순위 유지:', fallback['fallback_reason'])
    metrics = read_json(Path('reports/final/metrics.json'))
    print('5. 최종 test 품질과 실제 부하 측정 결과')
    for name, report in metrics['models'].items():
        print(f"   {name}: LogLoss={report['all']['click']['logloss']:.6f}, NDCG@5={report['all']['ranking']['ndcg_at_5']:.6f}")
    for row in read_json(Path('reports/week3/test-load/metrics.json'))['runs']:
        print(f"   {row['target_rps']:g} RPS: QPS={row['successful_qps']:.2f}, p50/p95/p99={row['latency_success_ms']['p50']:.2f}/{row['latency_success_ms']['p95']:.2f}/{row['latency_success_ms']['p99']:.2f}ms, 실패율={row['failure_rate']:.3%}")
    print('공개된 후보의 재정렬 데모입니다. 온라인 CTR 상승을 입증한 결과는 아닙니다.')


if __name__ == '__main__':
    main()
