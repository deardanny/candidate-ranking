"""Render the metadata audit from measured artifacts, without selecting a release."""
from datetime import datetime,timezone
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ebnerd-metadata-audit'
def read(name):return json.loads((OUT/(name+'.json')).read_text())
s=read('summary');v=read('verification');e=read('exposure_comparison');t=read('tie_diagnostics')
f=read('frozen_interventions');p=read('coherent_permutation');c=read('modification_cohorts');a=read('archive_page_comparison')
replay=read('exposure_scoring_replay');eligibility=read('eligibility');positive=read('deliberately_invalid_positive_control')
delay_diagnostics=read('exposure_delay_diagnostics')['results']
delay_table='\n'.join(f"| {label} | {100*delay_diagnostics[str(sec)]['candidate_fraction_without_arrived_exposure']:.2f}% | {100*delay_diagnostics[str(sec)]['clicked_candidate_fraction_without_arrived_exposure']:.2f}% | {delay_diagnostics[str(sec)]['requests_all_feature_vectors_identical']:,} |" for sec,label in [(60,'60초'),(3600,'1시간'),(86400,'24시간')])
stress_text=''
if (OUT/'exposure_stress.json').exists():
    stress=read('exposure_stress')
    stress_text=f"""
추가로 **50회/초 × 20분, {stress['processed']:,}요청**을 예정 시각에 맞춰 단일 프로세스에서 재생했다. 모든 피처·모델 점수 차이는 0이었다. 예정 시각부터 계산 완료까지 p95 **{stress['scheduled_to_complete_ms']['p95']:.3f}ms**, p99 **{stress['scheduled_to_complete_ms']['p99']:.3f}ms**, 최댓값 **{stress['scheduled_to_complete_ms']['max']:.3f}ms**였으며, 50ms 초과는 **{stress['deadline_misses']}건**이었다. 이 수치에는 스케줄러 대기가 포함된다. HTTP·저장소·실제 로그 전달을 포함하지 않고, 역사적 사건 시각과 성능 측정의 벽시계는 따로 진행했다. 따라서 60초 도착 가정이나 운영 HTTP SLO를 입증한 검사가 아니다. [지속 재생 결과](exposure_stress.json).

초과 8건은 긴 계산 2건과 그 뒤의 대기 6건이었다. GC/OS 추적을 수집하지 않아 계산이 길어진 내부 원인은 단정하지 않는다. 짧은 검사에서 보이지 않던 긴 지연을 그대로 보존했으며, 결과를 바꾸려고 재실행하지 않았다. [지연 사례](exposure_stress_outliers.json).
"""
arms={r['arm']:r for r in s['arms'] if r['fold']=='audit_holdout'}
def interval(value,scale=1):return f"{value['estimate']*scale:+.6f} [{value['lower']*scale:+.6f}, {value['upper']*scale:+.6f}]"
def contrast(a,b,metric='ndcg5'):return s['comparisons'][a+'__minus__'+b]['audit_holdout'][metric]
names={'events_only':'고정 과거 클릭 통계','events_user':'고정 과거 클릭 통계 + 반복 클릭 이력',
    'clean_events_only':'고정 과거 클릭 통계, 정제 요청 학습','clean_events_user':'고정 통계·반복 클릭 이력, 정제 요청 학습',
    'pub_user':'과거 통계·이력 + 발행시각','pub_category_no_affinity':'발행시각 + 카테고리 (개인 관심도 제외)',
    'pub_category':'발행시각 + 카테고리 (사용자 피처 제외)',
    'pub_category_user':'발행시각 + 카테고리 관심도','pub_topic_user':'발행시각 + 주제 관심도',
    'pub_entity_user':'발행시각 + 개체명 관심도','full_metadata':'기존 메타데이터 기준선','full_personalized':'기존 전체 개인화',
    'permuted_personalized':'기사 메타데이터를 섞고 재학습','exposure_global':'과거 노출 로그 60초 지연, 공통 통계',
    'exposure_personal':'과거 노출 로그 60초 지연, 사용자 정보 포함','exposure_1h':'과거 노출 로그 1시간 지연','exposure_24h':'과거 노출 로그 24시간 지연'}
ordered=list(names)
table='\n'.join(f"| {names[k]} | {'원본' if arms[k]['population']=='raw' else '정제'} | {arms[k]['mean']['ndcg5']:.6f} | {100*arms[k]['mean']['hit1']:.3f}% | {arms[k]['min']['ndcg5']:.6f}–{arms[k]['max']['ndcg5']:.6f} |" for k in ordered)
frozen='\n'.join(f"| {label} | {f[key]['metrics']['ndcg5']:.6f} | {f[key]['minus_frozen_original']['ndcg5']['estimate']:+.6f} |" for key,label in [('metadata_all_missing','카테고리·premium·관심도 9개 결측'),('category_missing','카테고리·premium 및 카테고리 관심도 결측'),('topics_missing','주제 관심도 결측'),('entities_missing','개체명 관심도 결측'),('publication_missing','발행시각 피처 결측')])
common_order=['full_personalized','exposure_global','exposure_personal','exposure_1h','exposure_24h']
common='\n'.join(f"| {names[k]} | {e['models'][k]['mean']['ndcg5']:.6f} | {100*e['models'][k]['mean']['hit1']:.3f}% | {1000*e['models'][k]['mean']['common_cold_hit1']:.2f} |" for k in common_order)
main=e['cohorts']['exposure_personal']['all_common']['minus_full_personalized']
cold=e['cohorts']['exposure_personal']['all_model_cold'];new=e['cohorts']['exposure_personal']['new_user_with_fixed_history']
dev='\n'.join(f"| {fold} | {next(r['mean']['ndcg5'] for r in s['arms'] if r['arm']=='full_personalized' and r['fold']==fold):.6f} | {next(r['mean']['ndcg5'] for r in s['arms'] if r['arm']=='exposure_personal' and r['fold']==fold):.6f} |" for fold in ['early','late','audit_holdout'])

decision={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'original_benchmark_result_reproduced':True,'original_metadata_point_in_time_certified':False,
    'fraction_of_gain_due_to_post_event_edits_identified':False,'metadata_dependency_observed':True,
    'article_independent_replay_implemented':True,'actual_exposure_arrival_timestamps_available':False,
    'new_champion_selected_or_promoted':False,'holdout_used_to_select_a_winner':False,'original_registry_aliases_changed':False,
    'conclusion':'Preserve original snapshot benchmark metrics; withhold a leakage-free deployment claim. Article-free exposure replay is promising under explicit arrival assumptions, but loses Top1 and all-cold performance versus the snapshot model and deteriorates with log delay.',
    'required_next_evidence':['Versioned article-field values and processing versions with available_at timestamps.','Actual exposure-event ingestion timestamps and complete production logging contract.','A new untouched evaluation period, followed by an authorized online evaluation if appropriate.']}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')

text=f'''# EB-NeRD 메타데이터 시점 감사 — 좋은 점수와 당시 이용 가능했던 정보

실행일: 2026-11-19. **기존 성능 개선은 재현됐지만, 그 개선이 사후 수정된 메타데이터의 영향을 받지 않았다고 인증할 수 없다.** 수정 전 버전이 없어서 누수 기여율을 식별할 수 없다는 것이 핵심 결론이다. 수정시각이 늦다는 이유만으로 모든 필드가 바뀌었거나 모든 개선이 누수라고 단정하지도 않는다.

이번에는 **{s['models_fitted']}개 모델**을 학습·재학습했다. 17개 조건 × seed 3개 × 개발 두 구간/최종 재학습 3개다. 모든 조건을 보고했고 새 champion을 고르지 않았다. 기존에 이미 본 5/25–6/1 평가 주를 다시 사용한 **사후 감사**이며 새로운 독립 검증 성과가 아니다. 원래 모델·피처·고정 지표는 보존했다. [판단 JSON](decision.json), [최초 감사 계획](protocol.json), [모든 조건·seed·구간](summary.json).

## 1. 누수가 의심되는 경로를 분리했다

| 경로 | 실제 확인 | 판단 |
|---|---|---|
| 당일 정답·미래 클릭 통계를 모델 입력에 사용 | 원래 25개 입력과 과거 이력 시간 검사 재확인 | 원래 모델은 해당 컬럼을 사용하지 않음 |
| 이력 기사의 카테고리·주제를 나중 버전으로 읽음 | 과거 클릭 시각은 정상이어도 기사 lookup은 현재 제공 snapshot | 과거 클릭만 사용했다는 사실로 이 경로를 배제할 수 없음 |
| 후보 기사의 카테고리·주제를 나중 버전으로 읽음 | 평가 후보 2,927,774행 모두 마지막 수정시각이 요청 이후 | 당시 값은 파일만으로 확인 불가 |
| 발행시각 피처 | 별도 제거·재학습·아카이브 표본 확인 | 발행시각도 전체 불변성이 입증된 필드는 아님 |
| 발행시각에 의한 요청 제외 | 원본 train 232,887 / test 244,647을 기사 파일 없이 재구축 | 피처 삭제만으로 남던 정제 의존까지 제거 |

공식 설명상 이력은 각 주 시작 전 21일의 고정 이력이며, 주제·개체 태그는 제목·요약·본문에서 생성된다. 따라서 콘텐츠가 바뀌면 파생 피처도 영향을 받을 수 있다. [공식 데이터 설명](https://recsys.eb.dk/dataset/).

마지막 수정시각은 필드별 변경 이력이 아니다. **어떤 필드가 언제 바뀌었는지, 추출 모델이 언제 실행됐는지**는 제공 snapshot에 없다. 모든 기사가 6/29 이후 수정됐다는 관측은 가용 시점의 불확실성이지, 수정의 원인이나 누수의 크기에 대한 증거가 아니다.

## 2. 피처를 나눈 재학습 결과

각 조건은 원래 LightGBM 설정을 유지하고 원래 두 개발 구간으로 조기 종료했다. 최종 학습 라운드는 두 번째 개발 구간에서 정했다. seed는 42·43·44이며, 표는 세 seed 평균이다. 정제 평가는 244,587요청, 원본 평가는 244,647요청이다. 서로 다른 모집단의 숫자를 동일 요청의 차이로 계산하지 않는다.

| 조건 | 평가 모집단 | NDCG@5 | Hit@1 | seed별 NDCG 범위 |
|---|---|---:|---:|---:|
{table}

주요 같은 모집단의 비교(95% 사용자 단위 paired bootstrap, 2,000회):

- 전체 개인화 − 메타데이터 기준선: **{interval(contrast('full_personalized','full_metadata'))}**.
- 다른 입력을 동일하게 두고 카테고리 관심도 두 피처만 추가: **{interval(contrast('pub_category_user','pub_category_no_affinity'))}**.
- 주제 관심도 추가: **{interval(contrast('pub_topic_user','pub_user'))}**.
- 개체명 관심도 추가: **{interval(contrast('pub_entity_user','pub_user'))}**.
- 기사 파일 없는 반복 클릭 이력 추가: **{interval(contrast('events_user','events_only'))}**. 구간이 0을 포함한다.

카테고리 관심도만으로도 전체 모델에 가까운 결과가 나왔다. 따라서 효과가 주제·개체명 추출에만 의존하는 것은 아니다. **카테고리도 버전이 확인되지 않았으므로 이것이 무누수 증명은 아니다.** 각 비교는 다른 모델을 재학습한 결과라, 차이를 더하거나 나눠서 전체 개선의 원인별 기여율을 만들지 않는다.

## 3. 같은 모델에 다른 정보를 넣는 대조

기존 seed 42 전체 모델의 NDCG@5는 0.583994다. 아래는 재학습 없이 입력을 결측으로 만든 결과다.

| 조작 | NDCG@5 | 원래 점수 대비 차이 |
|---|---:|---:|
{frozen}
| 같은 발행일 안에서 기사 메타데이터를 일관되게 교환 | {p['frozen_model_metrics']['ndcg5']:.6f} | {p['minus_original']['ndcg5']['estimate']:+.6f} |

교환 때 후보와 이력 양쪽 lookup을 함께 바꿨다. 기사 ID·발행시각은 그대로 두고 카테고리·premium·주제·개체명을 묶어서 이동했다. 행동·사용자·발행시각 16피처는 **모든 학습·평가 행에서 원본과 동일**했다. 교환한 피처로 다시 학습한 모델도 모두 보고했다.

이 결과는 모델이 해당 정보를 사용한다는 증거다. 결측·무작위 교환은 분포를 바꾸며 **실제 수정 전 값**이 아니므로 감소량을 누수량으로 해석할 수 없다. 개체명 결측의 작은 상승을 보고 그 피처를 제거한 모델을 승격하지 않았다. [고정 모델 조작](frozen_interventions.json), [일관된 교환](coherent_permutation.json).

## 4. 기존 이력 교체 검사가 증명하지 못했던 것

일부러 잘못된 점수를 하나 만들었다. 같은 사용자의 평가 주 전체 클릭 횟수에 **현재 정답까지 포함**해 후보 점수로 썼다. 이 점수의 NDCG는 {positive['future_user_item_counts']['ndcg5']:.6f}였고, 사용자 연결을 섞으면 {positive['permuted_future_user_item_counts']['ndcg5']:.6f}로 떨어졌다.

**명백히 누수된 점수도 사용자 이력을 섞으면 나빠졌다.** 따라서 이전의 0.5840 → 0.5282 이력 교체 실험은 사용자와 입력의 연결이 중요하다는 증거이며, 누수가 없다는 인증은 아니다. 이 의도적 누수 대조는 모델로 학습·등록하지 않았다. [명시적으로 잘못된 양성 대조](deliberately_invalid_positive_control.json).

## 5. 정제와 과거 버전도 따로 확인했다

기사 파일을 전혀 읽지 않는 파이프라인으로 원본 요청 전체를 다시 만들었다. 원래 정제 집합과 겹치는 모든 후보에서 13개 행동·사용자 피처, 후보 ID, 라벨, 그룹이 정확히 같았다. 제외된 요청은 train 641개, test 60개다. 같은 article-free 모델을 원본 요청 전체와 공통 요청에 적용했을 때 NDCG 차이는 약 0.000024였다. 같은 요청에 대해 정제 train과 원본 train으로 학습한 모델의 평균 차이도 0.0002 미만이었다. **이 검사 범위에서는 행 제외가 기존 +0.037 수준 개선의 주된 설명이 아니다.** 전체 콘텐츠 모델의 모든 선택 편향을 없앴다는 뜻은 아니다. [정제 경로 대조](eligibility.json).

공식 전체 기사 bundle 125,541개도 별도 다운로드했다. small과 겹치는 20,738개 기사의 20개 필드가 전부 동일했다. 큰 bundle이 과거 버전을 제공하는 대조군은 아니었다. [전체 파일 비교](reference_comparison.json), [독립 재계산](reference_verification.json).

수정시각이 6/29인 후보만 있는 179,550요청에서도 원래 개인화 이득은 +0.037839였고, 6/30 이후 수정 후보가 하나라도 있는 65,037요청에서도 +0.035258이었다. 후기 수정 기사에만 이득이 집중된 현상은 아니다. **6/29도 평가 뒤이므로 어느 쪽도 무누수 대조군은 아니다.** [수정시각별 전체 후보 평가](modification_cohorts.json).

공개 URL 12개를 정답과 무관한 규칙으로 골라 Internet Archive도 조회했다. 9개는 접근 가능한 가까운 snapshot이 있었고, 3개는 해당 기사의 첫 평가 노출보다 이른 페이지였다. 이 3개 페이지의 제목과 발행시각은 제공 데이터와 같았다. 다만 이는 오래된 페이지 표본이며, 아카이브 시점과 요청 사이 변경·카테고리 ID·독점 태그 생성 버전을 확인하지 못한다. 전체 후보의 과거 버전을 복원한 것이 아니다. [조회 계획](archive_probe_protocol.json), [페이지 비교](archive_page_comparison.json), [Archive API 설명](https://archive.org/help/wayback_api.php).

## 6. 메타데이터를 버리자 왜 성능이 내려갔나

고정 과거 이력만 쓰는 입력에서는 **{t['all_candidates_identical_inputs']:,}요청, {100*t['fraction_all_identical']:.2f}%**의 후보 입력이 모두 같았다. 이때 모델은 같은 요청의 후보를 구분할 수 없다. 실제 모델 점수도 동점 기대값과 일치했다.

동일한 입력끼리는 같은 점수를 줘야 한다는 제약 아래 **정답을 미리 아는 비현실적 oracle**까지 계산했는데, 전체 NDCG 상한은 {t['metrics']['label_oracle_constrained_to_identical_feature_ties']['ndcg5']:.6f}였다. 실제 고정 이력 모델은 약 0.404다. 이 oracle은 서비스 모델이 아니며, 이 피처 표현·고정 이력 일정·동점 평가에 한정한 낙관적 상한이다. 더 큰 모델만으로 기존 0.584를 회복할 수 있다고 기대할 근거가 없었다. [동일 입력의 한계](tie_diagnostics.json).

## 7. 추가 가설: 기사 파일 대신 이미 발생한 노출 로그를 쓰자

카탈로그와 발행시각을 읽지 않고 과거 노출에서 기사별 누적/최근 1시간/최근 24시간 노출 수, 최초/최근 노출 경과시간, 사용자의 과거 노출 횟수를 계산했다. 클릭 라벨은 이 신규 집계에 사용하지 않았다. 현재 요청을 점수화한 후 그 노출을 큐에 넣고, **노출시각 + 지연 < 현재 요청시각**인 기록만 반영한다. 같은 시각의 요청 순서로 정보가 유입되지 않도록 했다.

60초, 1시간, 24시간 도착 지연 각각의 학습·평가 피처를 다시 만들고 재학습했다. 조건별 원본 244,647요청 평가와 별도로, 아래는 **원래 모델과 똑같은 244,587요청·후보**만 비교한 표다. 마지막 열은 모든 모델에 같은 학습 미관측 기사 집합을 적용했다.

| 모델 | NDCG@5 | Hit@1 | 새 기사 클릭 선택 / 전체 1,000요청 |
|---|---:|---:|---:|
{common}

60초 도착 + 사용자 정보 모델을 기존 전체 모델과 비교하면:

- NDCG 차이: **{interval(main['ndcg5'])}**.
- Hit@1 차이: **{interval(main['hit1'],100)} %p**. NDCG와 반대로 낮아졌다.
- 후보가 모두 모델 학습에 없던 {cold['requests']:,}요청: NDCG 차이 **{interval(cold['minus_full_personalized']['ndcg5'])}**.
- 원본 train에 없는 사용자, 고정 과거 이력은 있는 {new['requests']:,}요청: **{interval(new['minus_full_personalized']['ndcg5'])}**.

같은 후보로 비교한 평가 주의 7일 모두 NDCG 차이는 양수였지만, Hit@1 차이는 5일에서 음수였다. 하루 평균만 보고 지표 간 손익을 숨기지 않았다. 같은 평가 주의 날짜별 진단이므로 새로운 외부 기간의 재현으로 세지 않는다.

개발 기간의 방향도 모두 같지는 않았다. 아래 숫자는 각 모델 고유 모집단 평균이며, raw/clean 차이가 있으므로 정확한 paired 차이 표는 아니다.

| 시간 구간 | 기존 전체 모델 | 60초 도착 + 사용자 정보 |
|---|---:|---:|
{dev}

따라서 **기사 메타데이터 없이도 쓸 만한 순위 신호를 만들 수 있었다**는 관측을 얻었지만, 기존 모델을 모든 목표에서 대체했다는 결론은 내리지 않는다. 첫 개발 구간·Top1·모든 후보가 새 기사인 집단에서는 약점이 남았다. 로그 도착 지연을 1시간으로 바꾸면 0.535, 24시간이면 0.425 수준으로 떨어졌다.

지연이 길어질 때 입력에서 실제로 무엇이 사라지는지도 원본 244,647요청에서 계산했다.

| 로그 도착 가정 | 과거 노출이 아직 없는 후보 행 | 클릭 후보 중 과거 노출이 아직 없는 비율 | 모든 후보 입력이 같은 요청 |
|---|---:|---:|---:|
{delay_table}

24시간 지연에서는 클릭 후보의 92% 이상에 도착한 과거 노출이 없었다. 단순히 모델 점수가 흔들린 것이 아니라 새 후보를 구별할 정보가 줄었다. 지연이 길어져도 과거 노출 횟수가 늘어나는 행이 없다는 단조성도 전체 행에서 확인했다. 이 집단들은 지연에 따라 구성이 달라지므로 표를 성능 감소의 인과적 기여율로 해석하지 않는다. [도착 지연의 정보 손실 진단](exposure_delay_diagnostics.json).

실제 ingestion timestamp는 데이터에 없다. 60초는 재생 가정이며 입증된 운영 사실이 아니다. 공개 데이터의 사용자·요청 선택도 완전한 운영 노출 로그와 같다고 가정할 수 없다. 이 대안은 **기사 snapshot 의존 경로를 제거한 것**이지 모든 종류의 데이터 편향·가용 시점 불확실성을 제거한 것이 아니다. [60초 사전 계획](exposure_protocol.json), [지연 민감도 계획](exposure_latency_protocol.json), [같은 후보 비교](exposure_comparison.json).

![같은 후보에서의 성능과 로그 지연 민감도](audit_results.png)

## 8. 검증한 범위

- 신규 {v['models_verified']}개 모델을 모두 다시 로드해 **{v['prediction_rows_recomputed']:,}개 후보 점수**를 재계산했다. 저장 점수와 최대 차이 0.
- sklearn과 직접 계산으로 {v['independent_metric_queries']:,}요청의 NDCG/AUC/Top1을 독립 대조했다. 허용 오차 1e-12 안에서 일치했다.
- 기존 네 조건의 seed 42 예측을 개발 두 구간·최종 구간에서 완전히 재현했다. 원래 설정·피처·최종 지표·고정 artifact 해시도 유지됐다.
- 세 도착 지연 각각에서 train+test **5,514,689후보 행의 7개 신규 피처**를 timestamp 정렬/검색 방식으로 독립 재계산해 전부 일치했다.
- 원본 과거 노출로 상태를 재생한 뒤 마지막 {replay['requests_scored']:,}요청의 메모리 피처·모델 점수가 offline cache와 정확히 같았다. 이 내부 계산의 p95는 {replay['in_memory_feature_score_observe_ms']['p95']:.3f}ms였다. HTTP·네트워크·저장소·실제 수집 지연을 포함한 부하 SLO가 아니다.
- 신규 관련 단위 테스트 7개를 통과했다. 원래 FairJob/EB-NeRD registry와 서비스 모델은 교체하지 않았다.

큰 기사 파일 재검산 도중, null을 정규화한 뒤 pandas가 다시 NaN으로 변환해 동일 null을 다르다고 세는 검증 스크립트 오류를 발견했다. canonical JSON 비교로 수정했고 실제 모든 필드가 동일함을 재확인했다. 모델 결과는 바뀌지 않았다.

[모델·지표 재검산](verification.json) · [60초 피처 전체 검산](exposure_feature_verification.json) · [1시간 검산](exposure_d3600_feature_verification.json) · [24시간 검산](exposure_d86400_feature_verification.json) · [메모리 점수 재생](exposure_scoring_replay.json).

{stress_text}

## 9. 프로젝트의 결론과 다음 작업

프로젝트 스토리는 “콘텐츠를 넣어 성능이 올랐다”에서 끝나지 않는다. **좋은 점수가 나온 뒤, 그 입력이 요청 당시에도 있었는지 다시 물었다. 기존 대조 검사만으로는 답할 수 없어서 성공 주장의 범위를 줄였고, 기사 버전에 기대지 않는 과거 노출 방식으로 대안을 만들었다. 전체 순위는 회복됐지만 Top1·새 기사·로그 지연에서 다른 실패를 확인했다.** 실제 운영 사고를 겪었다는 이야기가 아니라 이번에 수행한 오프라인 감사와 재생 실험의 기록이다.

현재 판단은 다음과 같다.

1. 기존 0.5468 → 0.5840 결과는 **제공 snapshot 조건의 benchmark**로 보존한다. 누수 없는 운영 개선으로 소개하지 않는다.
2. 누수량을 알아내려면 기사 필드별 버전, 추출 모델 버전, `available_at` 기록이 필요하다. 현재 자료로 수치 하나를 만들어 단정하지 않는다.
3. 기사 파일 없는 노출 기반 방식은 별도 연구 후보로 남긴다. 원래 champion을 자동 교체하지 않는다.
4. 다음 검증에는 실제 노출 수집 도착시각과 새로운 미사용 기간이 필요하다. 그때 Top1, 새 아이템 성과, 데이터 도착 지연을 함께 판단해야 한다.

사용자 bootstrap 구간은 이번 주·세 seed에 조건부인 기술 통계다. 공유 기사/기간 변동, 장기 일반화, 온라인 CTR·매출의 인과효과를 포괄하지 않는다. 같은 holdout에서 여러 대조를 확인했으므로 모든 양의 구간을 새로운 확증 성공으로 세지 않는다.

재현 명령과 파일 위치는 [감사 실행 가이드](../../docs/EBNERD_METADATA_AUDIT.md)에 정리했다. 원본 기사·사용자별 이력·예측·아카이브 HTML은 공개 보고서에 포함하지 않고 로컬 ignored data에 보관했다.

후속 데이터 계약은 [값과 가용 시각의 구분](../../docs/FEATURE_AVAILABILITY.md)에 있다. 모델 alias는 [읽기 전용 registry 점검](registry_check.json)에서 FairJob 13, EB-NeRD 5로 유지됨을 확인했다.
'''
(OUT/'RESULTS.md').write_text(text)

labels=['Snapshot + interests','Category interests','Fixed past clicks','Exposure: 60 seconds','Exposure: 1 hour','Exposure: 24 hours']
snapshot=arms['full_personalized']['mean'];cat=arms['pub_category_user']['mean']
event=eligibility['results']['events_user']['row_selection_effect']
measures=[snapshot,cat,{'ndcg5':event['ndcg5']['same_raw_model_common_clean'],'hit1':event['hit1']['same_raw_model_common_clean']},
          *[e['models'][k]['mean'] for k in ['exposure_personal','exposure_1h','exposure_24h']]]
colors=['#c98b37','#c98b37','#688399','#296b9a','#296b9a','#296b9a']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1.4,1]})
positions=np.arange(len(labels));ax.barh(positions,[m['ndcg5'] for m in measures],color=colors,height=.62)
for i,m in enumerate(measures):ax.text(m['ndcg5']+.007,i,f"{m['ndcg5']:.4f}",va='center',fontsize=10)
ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlim(0,.67);ax.set_xlabel('NDCG@5 (same 244,587 requests)');ax.set_title('Metadata independence changes the trade-off',loc='left',weight='bold',pad=15)
for i,m in enumerate(measures):
    bx.scatter(100*m['hit1'],m['ndcg5'],color=colors[i],s=65)
    short=['Snapshot','Category','Fixed history','60 sec','1 hour','24 hours'][i]
    bx.annotate(short,(100*m['hit1'],m['ndcg5']),xytext=(6,7 if i!=1 else -14),textcoords='offset points',fontsize=9)
bx.set_xlim(13,35);bx.set_ylim(.38,.62);bx.set_xlabel('Hit@1 (%)');bx.set_ylabel('NDCG@5');bx.set_title('Better NDCG does not imply better Top1',loc='left',weight='bold',pad=15)
bx.grid(alpha=.16)
fig.suptitle('EB-NeRD: post-hoc metadata availability audit',fontsize=16,weight='bold',x=.04,ha='left')
fig.text(.04,.035,'3 seeds; complete candidate sets. Amber: article snapshot not time-verified. Dark blue: assumed exposure delay.\nExposure models use earlier exposure events, not current/future click labels. No model was promoted.',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.12,1,.93]);fig.savefig(OUT/'audit_results.png',dpi=180);fig.savefig(OUT/'audit_results.pdf');plt.close(fig)
print('Rendered audit report and figure; decision remains unverified metadata availability.')
