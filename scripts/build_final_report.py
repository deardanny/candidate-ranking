"""Combine actual evaluation, load, container and test evidence into one report."""
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from fairjob_ranker.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    final = read_json(ROOT / 'reports/final/metrics.json')
    load = read_json(ROOT / 'reports/week3/test-load/metrics.json')
    docker = read_json(ROOT / 'reports/week3/docker_verification.json')
    parity = read_json(ROOT / 'reports/week3/final_serving_verification.json')
    test_xml = ET.parse(ROOT / 'reports/week3/pytest.xml')
    suites = list(test_xml.getroot().iter('testsuite'))
    tests = sum(int(s.get('tests',0)) for s in suites)
    failures = sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    if failures:
        raise ValueError('Cannot mark tests complete when failures remain')
    if len(load['runs']) != 3 or any(r['duration_seconds'] != 300 for r in load['runs']) or load['warmup_seconds'] != 60:
        raise ValueError('Full 60s warmup + 300s sweep must finish first')
    slo = next(r for r in load['runs'] if r['target_rps'] == 50)
    target_met = slo['latency_success_ms']['p95'] < 50 and slo['failure_rate'] < .001 and slo['generator_dropped'] == 0
    lines = ['# FairJob 광고 후보 랭킹·실시간 서빙 — 완료 보고서', '',
             '공개 FairJob의 관측 광고 후보를 점수화하고 재정렬하는 학습·서빙 프로젝트다. 실제 광고 경매나 온라인 CTR 상승을 재현한 것은 아니다.', '',
             '## 구현과 검증', '',
             f'- 정제·분할, LightGBM·PyTorch, FastAPI·Redis, Docker Compose, 요청 재생·부하·장애 테스트와 최종 test 평가를 구현했다. 테스트 {tests}개 통과.',
             f'- 최종 test 요청 100개의 오프라인/API 최대 점수 차이: {parity["max_abs_error"]:.3g}. 후보 순서 불변·Redis 중단·모델 오류 때 대체 점수 확인.',
             f'- Linux ARM64 컨테이너 요청 100개의 Mac 오프라인 점수 대비 최대 차이: {docker["max_abs_error"]:.3g}. Redis 중단·재시작 검증.',
             '- 모델·서빙 설정을 먼저 고정한 뒤 test를 평가했다. Test 결과로 모델을 다시 학습하거나 설정을 조정하지 않았다.', '',
             '## 최종 test 품질', '', '| 모델 | LogLoss ↓ | AP ↑ | ROC-AUC ↑ | NDCG@5 ↑ | MRR ↑ |', '|---|---:|---:|---:|---:|---:|']
    for name, metrics in final['models'].items():
        c,r = metrics['all']['click'],metrics['all']['ranking']
        values=[c['logloss'],c['average_precision'],c['roc_auc'],r['ndcg_at_5'],r['mrr']]
        lines.append(f'| {name} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    r=final['models']['pytorch']['all']['ranking']
    random=final['models']['pytorch']['randomized']['ranking']
    ndcg_difference = final['paired_bootstrap']['pytorch_vs_lightgbm']['all']['ndcg_at_5']
    lines += ['',f'클릭 평가 {final["test_rows"]:,}행; 조건부 랭킹 평가 {r["eligible_impressions"]}개 노출 / {r["eligible_users"]}명 사용자. 무작위 표시 랭킹 하위집단은 {random["eligible_impressions"]}개 노출이다.',
              '그룹별 표본 수·coverage·신규 광고·calibration과 사용자 단위 bootstrap 95% 구간은 [최종 평가](final/RESULTS.md), [전체 지표](final/metrics.json)에 있다.',
              '서비스 데모는 미리 고정한 PyTorch 모델을 사용한다. Validation에서 더 강했던 LightGBM을 품질 기준으로 함께 보고한다.',
              f'PyTorch−LightGBM NDCG 차이는 {ndcg_difference["mean_difference"]:.6f}, 사용자 bootstrap 95% 구간은 [{ndcg_difference["ci95"][0]:.6f}, {ndcg_difference["ci95"][1]:.6f}]다. LogLoss와 랭킹 지표의 우열을 따로 해석한다.', '',
              '## 부하 측정', '',
              '각 단계는 60초 warm-up 후 300초 측정했다. 요청 도착률은 합성했으며, 컨텍스트·후보는 실제 test 노출이다. 요청 예정 시각부터 응답 완료까지 측정해 클라이언트 지연·서버 대기를 포함한다.', '',
              '| 목표 RPS | 성공 QPS | p50 ms | p95 ms | p99 ms | 실패율 | degraded 비율 | 발행 / 예정 |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in load['runs']:
        p=row['latency_success_ms']
        lines.append(f"| {row['target_rps']:g} | {row['successful_qps']:.2f} | {p['p50']:.2f} | {p['p95']:.2f} | {p['p99']:.2f} | {row['failure_rate']:.3%} | {row['degraded_rate']:.3%} | {row['issued']} / {row['scheduled']} |")
    lines += ['', f'사전에 둔 50 RPS 목표(p95 < 50ms, 실패율 < 0.1%)는 **{"충족" if target_met else "미충족"}**했다. 실패율에는 503·timeout·네트워크 오류·발행기 용량 초과를 모두 포함한다.',
              '지연 표는 성공 응답 기준이며, 전체 응답 지연·503 수·발행 지연은 [원본 측정](week3/test-load/metrics.json)에 별도로 기록했다. QPS는 마지막 요청이 끝날 때까지의 drain 시간을 포함한다.', '',
              '## 환경과 자원', '', f"- {load['environment']['platform']}; {load['environment']['machine']}; Python {load['environment']['python']}",
              f"- 물리 코어 {load['environment']['physical_cores']}, 논리 코어 {load['environment']['logical_cores']}, RAM {load['environment']['ram_gib']:.1f} GiB",
              '- 위 부하 수치는 네이티브 macOS 프로세스에서 측정했다. API·Redis와 발행기는 같은 기기를 사용했고 Colima VM은 정지했다. 컨테이너 성능 수치로 표현하지 않는다.',
              '- 서버 CPU 100%는 코어 1개 사용량이다. 메모리는 API·자식 프로세스·Redis의 RSS 합으로 공유 페이지가 중복 집계될 수 있다. 발행기 자원은 별도로 수집했다.',
              '- Docker 기능 검증: Colima VM 2 CPU / 4 GiB, API 제한 2 CPU / 2 GiB, Redis 제한 0.5 CPU / 256 MiB. 이미지에는 데이터·가중치를 넣지 않고 모델 번들을 읽기 전용으로 마운트했다.',
              f'- PyTorch 가중치 파일 {Path(final["frozen_experiment"]["pytorch_dir"]).joinpath("weights.pt").stat().st_size:,} bytes. Uvicorn worker 1개, PyTorch thread 1개, 후보 일괄 추론, 실행·대기 합계 2개. [고정 설정](../configs/serving_final.json), [Python 패키지 버전](../requirements-week3.lock).', '',
              '| 목표 RPS | 서버 CPU p95 (코어 1개=100%) | 서버 RSS 최대 MiB | 발행기 CPU p95 | 발행기 RSS 최대 MiB |',
              '|---:|---:|---:|---:|---:|']
    for row in load['runs']:
        a,b=row['server_resources'],row['generator_resources']
        lines.append(f"| {row['target_rps']:g} | {a['cpu_percent_one_core_100']['p95']:.1f} | {a['rss_mib']['max']:.1f} | {b['cpu_percent_one_core_100']['p95']:.1f} | {b['rss_mib']['max']:.1f} |")
    lines += ['', '## Validation에서 수행한 최적화 비교', '',
              '범주 사전 lookup을 모델 로딩 시 한 번 준비하고 수치형 피처를 배열로 변환했다. 이전 변환 수식과 dtype·값이 정확히 같은지 검증했다. 아래는 설정 선택용 짧은 탐색 측정이며 장시간 수치와 직접 동일시하지 않는다.', '',
              '| 실험 | 조건 | p95 ms | 실패율 |', '|---|---|---:|---:|']
    for name in ['validation-legacy-controlled','validation-optimized','validation-serial','validation-two-workers-rerun','validation-optimized-64','validation-serial-64']:
        data=read_json(ROOT/f'reports/week3/{name}/metrics.json')
        row=data['runs'][0]
        lines.append(f"| {name} | {'합성 후보 64개' if data['stress_64'] else '실제 후보'} | {row['latency_success_ms']['p95']:.2f} | {row['failure_rate']:.2%} |")
    lines += ['', '초기 baseline 두 실행은 환경 준비 및 동기 자원 수집기의 영향을 받을 수 있어 비교 표에서 제외했다. 이전 전처리를 재현한 별도 validation 앱을 동일한 최종 측정기로 측정한 legacy-controlled 결과를 사용했다. 이 비교 실행은 최종 설정 고정 후 측정 방식의 일관성을 확인하기 위해 추가했으며 모델·설정 선택에는 사용하지 않았다. 중단된 worker 실행도 제외했다.', '',
              '## 장애·과부하 측정', '', '| 시나리오 | 성공 QPS | 실패율 | degraded 비율 | 상태 코드 |', '|---|---:|---:|---:|---|']
    faults = {}
    for name in ['redis-outage','model-error','overload']:
        data=read_json(ROOT/f'reports/week3/{name}/metrics.json')
        row=data['runs'][0]
        faults[name]=row
        lines.append(f"| {name} | {row['successful_qps']:.2f} | {row['failure_rate']:.2%} | {row['degraded_rate']:.2%} | {row['status_counts']} |")
    lines += ['', 'Redis 장애는 로컬 카탈로그로 같은 점수를 반환한다. 모델 오류는 모든 후보를 train 통계 대체 모델로 처리하고 모델 버전·사유를 표시한다. 과부하는 테스트 앱에서 추론을 50ms 지연시킨 강제 조건이며, 실행·대기 용량을 넘는 요청을 503으로 거절한다.',
              '장애 주입은 별도 테스트 ASGI 앱에만 존재하며 일반 API에 제어 endpoint를 만들지 않았다.', '',
              '## 재현·데모', '',
              '- [실행 안내](../README.md), [코드와 설계](../docs/FINAL_GUIDE.md), [고정 실험](frozen_experiment.json)',
              '- 로컬 5단계 데모: `python3 scripts/run.py scripts/final_demo.py`',
              '- Docker: `docker compose up --build`; 이 Mac의 전용 Colima 설정은 `python3 scripts/docker_project.py compose up --build`',
              '- 부하 재실행: `python3 scripts/run.py scripts/benchmark_suite.py --corpus data/replay/test.jsonl --frozen reports/frozen_experiment.json --settings configs/serving_final.json --output runs/new-test-load`',
              '- 원본·가공 데이터·모델·원시 요청 trace는 Git에서 제외한다. 데이터는 FairJob의 CC-BY-NC-SA 4.0 조건을 따른다.', '',
              '## 범위와 한계', '',
              '타임스탬프·전체 경매 후보·입찰·예산·전환 데이터가 없다. 공개 후보의 재정렬과 로컬 요청 서빙을 구현했다. Retrieval, 온라인 A/B 테스트, Kafka/Kubernetes, 온라인 학습은 이 프로젝트 범위에 포함하지 않았다.', '']
    (ROOT/'reports/PROJECT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    evidence={'created_at_utc':datetime.now(timezone.utc).isoformat(),'tests_passed':tests,'test_failures':failures,
              'docker_verified':True,'test_evaluated':True,'full_load_sweep_completed':True,'target_50rps_met':target_met,
              'fault_scenarios':list(faults),'final_parity_max_error':parity['max_abs_error']}
    write_json(ROOT/'reports/completion_evidence.json',evidence)
    print('Saved reports/PROJECT_REPORT.md and completion_evidence.json')


if __name__ == '__main__':
    main()
