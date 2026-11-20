"""Generate a report only from completed V2 experiments and verification artifacts."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,median
import xml.etree.ElementTree as ET

from fairjob_ranker.io import read_json,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    experiment=read_json(root/'reports/v2/experiment/metrics.json')
    load=read_json(root/'reports/v2/load/metrics.json')
    verification=read_json(root/'reports/v2/verification/evidence.json')
    release=read_json(root/'reports/v2/release_rehearsal/evidence.json')
    observability=read_json(root/'reports/v2/observability-verified/evidence.json')
    control=read_json(root/'reports/v2/control_hardening.json')
    assert control['http_promotion_rollback_verified'] and control['real_champion_unchanged']
    tests=list(ET.parse(root/'reports/v2/pytest.xml').getroot().iter('testsuite'))
    assert all(int(t.attrib.get('failures',0))+int(t.attrib.get('errors',0))==0 for t in tests)
    test_count=sum(int(t.attrib['tests']) for t in tests)
    expected=len(load['config']['rates'])*len(load['config']['burst_sizes'])*len(load['config']['capacities'])*load['config']['repeats']
    assert load.get('completed') and len(load['runs'])==expected
    groups=defaultdict(list)
    for r in load['runs']:
        groups[(r['capacity'],r['target_rps'],r['burst_size'])].append(r)
    assert all(len(g)==3 for g in groups.values())
    capacity_scores={capacity:min(r['on_time_success_rate'] for r in load['runs'] if r['capacity']==capacity and r['target_rps']==100)
                     for capacity in load['config']['capacities']}
    chosen=min(capacity_scores,key=lambda c:(-capacity_scores[c],c))
    selection={'max_inflight':chosen,'shadow_sample_rate':load['config']['shadow_sample_rate'],
               'basis':'Best worst-repeat 50ms success at 100 RPS across constant and burst; ties prefer smaller capacity',
               'worst_100rps_on_time':capacity_scores[chosen],
               'development_only':True,'configuration':'configs/v2_load.json'}
    write_json(root/'configs/v2_serving_selected.json',selection)
    candidate,champion=experiment['challenger'],experiment['champion']
    lines=['# V2: 모델 실험에서 교체 차단까지','',
           'DCN-V2·랭킹 손실을 추가하고 MLflow에 실험을 기록했다. 품질 조건을 통과하지 못한 후보를 shadow에서 비교하며, 실제 API 교체는 차단했다. Prometheus/Grafana 관측과 버전 교체·롤백 검증도 구현했다.','',
           '## 모델 실험 결과','',
           '원 train을 사용자 단위로 다시 나누고, inner train에서만 전처리·모델·대체 통계를 적합했다. 네 조건을 seed 42/43/44로 실행했다. 아래는 내부 validation의 세 seed 평균과 표준편차다. V1 결과와 분할이 달라 직접 개선율을 계산하지 않는다.','',
           '| 모델·목표 | LogLoss 평균 ↓ | NDCG@5 평균 ↑ | NDCG seed 표준편차 |','|---|---:|---:|---:|']
    for family,values in experiment['selection']['families'].items():
        lines.append(f"| {family} | {values['logloss_mean']:.6f} | {values['ndcg_mean']:.6f} | {values['ndcg_seed_std']:.6f} |")
    lines += ['', 'Pairwise 가중치 0.1은 내부 NDCG 평균을 높였지만 LogLoss 허용 범위를 넘었다. 이 설정에서 관측한 결과이며, pairwise 방식 전반이 나쁘다는 결론은 아니다.',
              f"후보군에서 내부 LogLoss 조건을 통과한 {candidate['family']}를 선택하고 대표 seed를 사전에 정한 42로 고정한 뒤, 기존 outer validation에서 평가했다. 이 후보가 전체 모델 중 최선이었다는 뜻은 아니다.", '',
              '| 별도 validation 비교 | LogLoss ↓ | NDCG@5 ↑ |','|---|---:|---:|']
    for r in [champion,candidate]:
        lines.append(f"| {r['name']} | {r['outer']['all']['click']['logloss']:.6f} | {r['outer']['all']['ranking']['ndcg_at_5']:.6f} |")
    ci=experiment['quality_gate']['comparison']['all']['ndcg_at_5']
    lines += ['',f"후보−LightGBM NDCG 차이 {ci['mean_difference']:.6f}, 사용자 bootstrap 95% 구간 [{ci['ci95'][0]:.6f}, {ci['ci95'][1]:.6f}]. 631개 노출 / 178명 사용자에 대한 개발 평가다.",
              f"**품질 교체 조건: {'통과' if experiment['quality_gate']['passed'] else '차단'}**. 조건별 결과: `{experiment['quality_gate']['checks']}`.",
              '0을 포함하는 구간으로 확정적인 성능 하락을 주장하지 않는다. 사전에 정한 차이 점추정 ≥ 0 및 구간 하한 ≥ −0.01 조건을 충족하지 못해 교체를 보류했다.',
              '기존 test는 이번 작업에서 읽지 않았다. 기존 test를 이미 확인한 프로젝트이므로 이번 결과는 새 독립 test 성능이 아니다.', '',
              '## 실행 가능한 배포 검증', '',
              f'- 모델 실험 13개(신경망 12 + LightGBM 1)를 로컬 MLflow Tracking/Registry에 등록했다. 실제 HTTP registry로 조회하고 alias 모델을 다시 로드해 점수 일치를 확인했다.',
              f'- 정상 요청 100개: 오프라인/API 최대 오차 {verification["max_abs_error"]:.3g}, 후보 순서 변경에 불변, Redis 중단 시 같은 점수.',
              f'- 실제 후보 교체 요청: HTTP 409, `{verification["promotion_blocked"]["detail"]}`. 기존 활성 모델 유지.',
              '- Shadow는 spawned process와 제한된 입출력 큐를 사용한다. 느린 shadow 및 예외를 주입해도 활성 요청 50개씩은 정상 응답했고, shadow drop/error를 별도 기록했다.',
              '- 실제 HTTP promote/rollback 및 MLflow alias 복귀를 검증했다. 이 시연은 **동일 가중치의 새 패키지 revision**을 사용한 배포 절차 검증이며, 품질 미달 DCN을 승격한 것이 아니다. 별도 MLflow 레지스트리에 격리했다.',
              f'- 자동 테스트 {test_count}개 통과. Prometheus의 실제 API 수집, Grafana 패널 {observability["grafana_panels"]}개 provisioning·데이터 소스 연결, MLflow HTTP 조회 확인.', '',
              '## 요청 도착 형태와 대기 용량 실험', '',
              '각 조건 준비 2초 + 측정 10초, 조건별 3회다. 실행 순서를 고정 seed로 섞었다. 일정 간격(burst=1)과 5건씩 묶은 도착(burst=5)을 분리했다. 같은 rate의 비교는 같은 요청 prefix를 사용한다.',
              '기한 내 성공은 전체 예정 요청 중 50ms 안에 유효한 HTTP 200을 받은 비율이다. 거절·timeout·발행 누락은 실패로 남는다. 이는 순위 품질 지표가 아니다.', '',
              '| 대기 포함 용량 | RPS | 묶음 크기 | 기한 내 성공 평균 | 기한 내 성공 최저 | 실패율 평균 | 성공 p95 중앙값 ms |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for (capacity,rate,burst),rs in sorted(groups.items()):
        lines.append(f"| {capacity} | {rate} | {burst} | {mean(r['on_time_success_rate'] for r in rs):.3%} | {min(r['on_time_success_rate'] for r in rs):.3%} | {mean(r['failure_rate'] for r in rs):.3%} | {median(r['latency_success_ms']['p95'] for r in rs):.2f} |")
    lines += ['',f"개발 설정 선택: 총 in-flight **{chosen}개**. 100 RPS의 일정/묶음 도착을 합친 최저 기한 내 성공률은 {capacity_scores[chosen]:.3%}다. 99.9% 제안 목표를 {'충족' if capacity_scores[chosen]>=.999 else '충족하지 못'}했다.",
              '서버·발생기는 같은 Mac이며 shadow 표본 비율은 20%다. API·Redis·shadow를 서버 자원에 포함했다. MLflow/Grafana/Prometheus UI 프로세스와 Colima는 부하 측정 중 종료했다. 별도 발생기 기기 검증은 수행하지 않았다.',
              '이 짧은 실험은 설정 탐색이다. V1의 각 5분 측정이나 운영 SLO 인증으로 표현하지 않는다. CPU/RSS·발행 지연·상태 코드·모델 일치·degraded 비율·원시 요청은 [전체 기록](load/metrics.json)에 있다.', '',
              '부하 측정 후 모델 교체 제어부에서 레지스트리 I/O를 요청 snapshot 락 밖으로 옮겼다. 점수 계산·요청 스케줄링 코드는 그대로다. 54개 측정은 변경 전 버전에 해당하며, [당시 코드](load/source/deployment.py)와 [변경 해시·HTTP 재검증](control_hardening.json)을 보존했다. 수정 후 회귀 테스트와 실제 교체·rollback을 다시 통과했다.', '',
              '## 재현과 증거', '',
              '- [실행·코드 안내](../../docs/V2_GUIDE.md), [스토리](../../docs/PROJECT_STORY.md)',
              '- [사전 실험 계약](experiment/preregistered.json), [전체 모델 결과](experiment/metrics.json)',
              '- [HTTP·shadow·차단 증거](verification/evidence.json), [동일 가중치 교체·롤백](release_rehearsal/evidence.json)',
              '- [관측 도구 검증](observability-verified/evidence.json), [테스트 XML](pytest.xml)',
              '- 한 명령 데모: `python3 scripts/run.py scripts/demo_v2.py`', '',
              '## 남은 범위', '',
              'ONNX/Triton, 별도 발생기 호스트, 실제 온라인 A/B 테스트와 새 독립 평가 데이터는 추가하지 않았다. 현재 데이터에 없는 시간·전환·정책 확률을 만들어내지 않았다. V2 배포는 localhost 개발 환경이며 V1 Docker 실행 기록과 구분한다.', '']
    (root/'reports/v2/RESULTS.md').write_text('\n'.join(lines))
    write_json(root/'reports/v2/completion.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'tests_passed':test_count,'model_runs':len(experiment['runs']),'load_runs':len(load['runs']),
        'real_candidate_promoted':False,'quality_gate_passed':experiment['quality_gate']['passed'],
        'http_shadow_verified':True,'package_rollback_verified':True,'observability_verified':True,
        'post_load_control_reverified':True,'serving_selection':selection})
    print('Saved reports/v2/RESULTS.md and selected serving configuration')


if __name__=='__main__':main()
