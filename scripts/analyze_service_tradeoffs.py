"""Audit completed traces for deadline success and preprocessing cost; no retraining."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def summarize(records, run, deadline_ms):
    """The denominator includes rejected, timed-out and unissued requests."""
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError('Deadline must be positive and finite')
    if len(records) != run['scheduled'] or not records:
        raise ValueError('Incomplete trace')
    if sorted(r['index'] for r in records) != list(range(run['scheduled'])):
        raise ValueError('Duplicate or missing scheduled request index')
    statuses = dict(Counter(r['status'] for r in records))
    if statuses != run['status_counts']:
        raise ValueError('Trace does not match recorded status counts')
    successful = [r for r in records if r['status'] == '200' and r['issued']]
    failed = [r for r in records if r['status'] != '200']
    if len(successful) != run['successful'] or any(r['status'] == '200' and not r['issued'] for r in records):
        raise ValueError('Trace does not match recorded successful count')
    if any(not math.isfinite(r['latency_ms']) or r['latency_ms'] < 0 for r in successful):
        raise ValueError('Invalid successful response latency')
    late = [r for r in successful if r['latency_ms'] > deadline_ms]
    stages = {key: mean(r['timings_ms'][key] for r in successful) if successful else None
              for key in ['lookup', 'transform', 'inference', 'service', 'handler']}
    share = stages['inference'] / stages['service'] if stages['service'] else None
    return {'target_rps': run['target_rps'], 'scheduled': len(records),
            'successful': len(successful), 'failed': len(failed), 'late_successful': len(late),
            'on_time_successful': len(successful) - len(late),
            'on_time_success_rate': (len(successful) - len(late)) / len(records),
            'deadline_ms': deadline_ms, 'status_counts': statuses,
            'success_p95_ms': run['latency_success_ms']['p95'], 'mean_stages_ms': stages,
            'inference_share_of_mean_service': share,
            'ideal_service_speedup_if_inference_free': 1 / (1 - share) if share is not None and share < 1 else None,
            'failed_with_generator_lag_over_10ms': sum(r['send_lag_ms'] > 10 for r in failed),
            'successful_with_generator_lag_over_10ms': sum(r['send_lag_ms'] > 10 for r in successful)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/advanced/service_tradeoffs')
    parser.add_argument('--deadline-ms', type=float, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, summaries = {}, {}
    for name in ['test-load', 'validation-legacy-controlled', 'validation-optimized']:
        folder = ROOT / 'reports/week3' / name
        metrics_path = folder / 'metrics.json'
        metrics = read_json(metrics_path)
        if (folder / 'running.json').exists():
            raise ValueError(f'Load run not closed: {name}')
        sources[str(metrics_path.relative_to(ROOT))] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        summaries[name] = []
        for run in metrics['runs']:
            trace_path = folder / f"requests-{run['target_rps']:g}.jsonl"
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            sources[str(trace_path.relative_to(ROOT))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            summaries[name].append(summarize(records, run, args.deadline_ms))
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Post-hoc engineering audit of recorded traces; no fresh test evaluation or model selection.',
              'deadline_definition': 'Scheduled request time to full response, including generator delay. HTTP 200 is not a ranking quality guarantee.',
              'inference_speedup_note': 'Amdahl estimate for mean service time only, holding every other cost fixed. Not a p95 or capacity prediction.',
              'generator_lag_note': 'Association only; these co-located traces cannot identify the cause of rejections.',
              'source_sha256': sources, 'runs': summaries}
    lines = ['# 기존 부하 로그로 다시 본 지연·실패·병목', '',
             '기존 로그의 사후 분석이다. 새 모델을 평가하거나 기존 test를 새 holdout으로 사용하지 않았다.', '',
             f'기한 내 성공률 = HTTP 200이며 예정 시각부터 {args.deadline_ms:g}ms 이내 완료된 요청 / 전체 예정 요청.',
             '발행하지 못한 요청·503·timeout도 분모에 포함한다. HTTP 성공은 광고 순위의 품질을 뜻하지 않는다.', '',
             '| RPS | 성공 p95 ms | 실패 | 기한 초과 성공 | 기한 내 성공 / 예정 | 기한 내 성공률 |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in summaries['test-load']:
        lines.append(f"| {r['target_rps']:g} | {r['success_p95_ms']:.2f} | {r['failed']} | {r['late_successful']} | {r['on_time_successful']} / {r['scheduled']} | {r['on_time_success_rate']:.4%} |")
    hundred = next(r for r in summaries['test-load'] if r['target_rps'] == 100)
    lines += ['', f"100 RPS의 실패 {hundred['failed']}건 중 {hundred['failed_with_generator_lag_over_10ms']}건은 발행 지연도 10ms를 넘었다. 같은 기기의 발행기 지연과 서버 거절이 함께 관측됐지만, 원인·인과관계는 이 로그만으로 확정하지 않는다. 독립 발행기와 burst 통제 실험이 다음 단계다.", '',
              '## 추론만 빨라지면 얼마나 도움이 되나', '',
              '| Validation 설정 | 평균 변환 ms | 평균 추론 ms | 평균 service ms | 추론 비중 |',
              '|---|---:|---:|---:|---:|']
    for name in ['validation-legacy-controlled', 'validation-optimized']:
        r = summaries[name][0]
        s = r['mean_stages_ms']
        lines.append(f"| {name} | {s['transform']:.3f} | {s['inference']:.3f} | {s['service']:.3f} | {r['inference_share_of_mean_service']:.2%} |")
    optimized = summaries['validation-optimized'][0]
    lines += ['', f"현재 validation 측정에서 추론을 완전히 없애도 다른 비용이 그대로라면 평균 service 시간의 이론상 속도 향상은 약 {optimized['ideal_service_speedup_if_inference_free']:.2f}배다. ONNX/Triton이 API 전체를 몇 배 빠르게 할 것이라는 근거는 아직 없다. 이 계산은 평균 시간에 대한 것이며 p95나 최대 처리량의 상한이 아니다.",
              '위 비교는 각 설정 1회, 50 RPS, 준비 3초·측정 20초의 짧은 관측이다. 반복 측정으로 확정한 인과 효과가 아니다.',
              '원시 trace와 요약의 해시는 metrics.json에 기록했다. 기존 3주차 측정과 고정 모델은 변경하지 않았다.', '']
    args.output.mkdir(parents=True)
    (args.output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'RESULTS.md').write_text('\n'.join(lines))
    print(args.output / 'RESULTS.md')


if __name__ == '__main__':
    main()
