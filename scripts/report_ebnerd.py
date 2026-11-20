"""Build a result table and figure from frozen research and serving artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'reports/ebnerd-experiment'
r=json.loads((out/'final_metrics.json').read_text()); v=json.loads((out/'verification.json').read_text())
s=json.loads((ROOT/'reports/ebnerd-serving/verification.json').read_text()); selection=json.loads((out/'selection.json').read_text())
assert r['quality_passed'] and v['passed'] and s['passed']
names={'random':'무작위 동점 기대값','historical_popularity':'과거 클릭 인기',
       'behavior':'과거 통계·요청 정보','behavior_history':'과거 통계 + 반복 클릭 이력',
       'freshness':'과거 통계 + 기사 최신성','metadata':'최신성 + 기사 메타데이터',
       'personalized':'최신성 + 메타데이터 + 사용자 이력',
       'personalized_no_freshness':'개인화에서 최신성 제거','shuffled_history':'개인화 모델에 타인 이력 입력'}
table='\n'.join(f"| {names[k]} | {m['ndcg5']:.6f} | {100*m['hit1']:.2f}% | {1000*m['cold_hit1']:.2f} |" for k,m in r['metrics'].items())
primary=r['comparisons']['personalized - metadata']; control=r['comparisons']['personalized - shuffled_history']
daily='\n'.join(f"| {k.removeprefix('day_')} | {q['requests']:,} | {q['ndcg_difference']['estimate']:+.4f} | [{q['ndcg_difference']['lower']:+.4f}, {q['ndcg_difference']['upper']:+.4f}] |" for k,q in r['cohorts'].items() if k.startswith('day_'))
load='\n'.join(f"| {q['target_rps']} | {q['duration_seconds']}초 | {q['scheduled']:,} | {q['on_time_success_rate']:.2%} | {q['latency_success_ms']['p95']:.2f}ms | {q['latency_success_ms']['p99']:.2f}ms |" for q in s['load_reports'])
text=f'''# EB-NeRD end-to-end 결과 — 새 기사를 고르는 데 필요한 정보

> 후속 시점 감사: 아래 수치는 제공된 기사 snapshot을 가정한 원래 실험 기록이다. 153개 모델과 원본 요청·메타데이터 대조를 추가 검증했지만, 수정 전 버전이 없어 누수 없는 운영 개선으로 인증하지 않는다. 이력 교체만으로 누수를 배제할 수 없다는 반례도 확인했다. 최신 판단은 [메타데이터 시점 감사](../ebnerd-metadata-audit/RESULTS.md)에 있다.

실행일: 2026-11-19 America/New_York. **데이터 정제 → 2개 시간 구간 대조 → 설정 고정 → 최종 재학습·평가 → HTTP/Redis → 실제 부하·교체·rollback 검증을 완료했다.** 학습 모델 18개(개발 12개, 최종 재학습 6개), 무작위·과거 인기 기준선, 사용자 이력 교체 대조를 실행했다. 본 보고서는 뉴스 추천의 관측 후보·클릭에 관한 결과다.

## 1. 무엇이 실제로 개선됐나

정제 후 원래 train 232,246요청, local test **244,587요청 / 15,342명**을 사용했다. 개발은 5/21~23과 5/23~25 두 구간, 최종 평가는 5/25~6/1이다. 요청 시각의 timezone을 임의 변환하지 않았다. 6개 피처 조합과 주요 비교·bootstrap 규칙을 [protocol](protocol.json)에 고정했고, 선택한 모델과 재학습 라운드를 고정한 다음 최종 평가를 실행했다. [선택 기록](selection.json), [평가 전 모델 해시](frozen_release.json), [전체 지표](final_metrics.json).

| 모델·대조 | NDCG@5 ↑ | 1위 클릭 선택률 ↑ | 새 기사 클릭 선택 / 전체 1,000요청 ↑ |
|---|---:|---:|---:|
{table}

마지막 열은 **전체 후보를 경쟁시킨 뒤 학습에 없던 클릭 기사를 1위로 고른 횟수**다. 새 기사만 남긴 후보 집합에서 평가하거나, 기존 기사 클릭을 새 기사 성과로 세지 않았다. 모든 원래 요청에 클릭이 하나 이상 있으므로 표의 선택률은 서비스 전체 CTR이 아니다.

주 비교인 개인화−메타데이터 모델의 사용자 단위 paired bootstrap 95% 구간(2,000회):

- NDCG@5: **{primary['ndcg5']['estimate']:+.6f}**, [{primary['ndcg5']['lower']:+.6f}, {primary['ndcg5']['upper']:+.6f}]. 상대 약 {primary['ndcg5']['estimate']/r['metrics']['metadata']['ndcg5']*100:.2f}% 개선.
- 1위 선택률: **{100*primary['hit1']['estimate']:+.2f}%p**, [{100*primary['hit1']['lower']:+.2f}, {100*primary['hit1']['upper']:+.2f}]%p.
- 새 기사 클릭 선택: **1,000요청당 {1000*primary['cold_hit1']['estimate']:+.2f}건**, [{1000*primary['cold_hit1']['lower']:+.2f}, {1000*primary['cold_hit1']['upper']:+.2f}]. 실제 증분 클릭·매출 추정값은 아니다.

![모델 대조와 날짜별 재현](results.png)

## 2. 실패와 대조가 만든 스토리

**첫 가설: 과거에 많이 클릭된 기사를 우선하면 될까?** 고정된 과거 클릭 인기는 무작위보다 낮았다. 이는 오래된 기사에 쌓인 인기를 이번 주 새 후보에 그대로 적용하는 기준선이다. 현재 트래픽의 최신 인기 추천까지 열등하다는 뜻은 아니다.

**두 번째 가설: 사용자 이력을 붙이면 바로 개인화가 될까?** 기사 ID의 반복 클릭 횟수·마지막 클릭 시각만 추가한 `behavior_history`는 `behavior`를 이기지 못했다. 최종 NDCG 차이는 −0.000380이다. 학습에 없던 기사에는 그 사용자와의 반복 클릭 기록이 없는 문제가 남는다.

**세 번째 가설: 기사 최신성과, 사용자가 읽던 주제의 관계를 함께 넣으면 어떨까?** 발행 후 경과시간을 넣자 큰 변화가 생겼다. 그 위에 과거 이력의 카테고리 비중, 최근 20개 이력의 카테고리 비중, 주제·개체 겹침을 추가했을 때 두 개발 기간과 최종 평가에서 추가 개선이 나타났다. 클릭 정답·다음 체류시간·기사 전체 기간 누적 통계는 피처에서 제외했다.

**반증 대조: 이력이 아무 사용자 것이어도 좋아지는가?** 같은 개인화 모델에 같은 snapshot의 다른 사용자 이력을 고정 seed로 치환했다. 모델 재학습이나 test를 보고 선택하는 과정은 없었다. NDCG@5가 {r['metrics']['personalized']['ndcg5']:.4f}에서 {r['metrics']['shuffled_history']['ndcg5']:.4f}로 내려갔다. 실제 이력−치환 이력 차이의 95% 구간은 [{control['ndcg5']['lower']:.4f}, {control['ndcg5']['upper']:.4f}]다. 모델이 입력 이력의 사용자 정합성을 활용한다는 증거다. 특정 한 피처의 인과효과를 분리한 실험은 아니다.

이 결과는 복잡한 신경망 추가가 아니라 **새 아이템에 전달되는 정보가 무엇인지 바꾼 실험**에서 나왔다. FairJob과 EB-NeRD의 서로 다른 지표를 직접 비교해 개선율을 계산하지 않는다.

## 3. 재현 범위와 새 아이템

개인화 모델의 NDCG@5는 첫 개발 구간 0.599953, 두 번째 구간 0.609264였다. 각각 메타데이터 모델보다 높았다. 최종 기간의 날짜별 비교도 모두 양수였다.

| 원본 시계의 07시부터 하루 | 요청 | 개인화−메타데이터 NDCG | 사용자 bootstrap 95% 구간 |
|---|---:|---:|---|
{daily}

정제된 학습 후보·학습 이력 어디에도 없던 기사 1,399개 중 861개에 실제 클릭이 있었다. 기사별 클릭 선택 비율을 동일 가중한 macro 값도 **{v['diagnostics']['metadata']['cold_item_macro_capture']:.4f} → {v['diagnostics']['personalized']['cold_item_macro_capture']:.4f}**였다. 기사 571개에서 상승, 201개에서 하락, 89개는 동률이다. 전체 요청의 사용자 bootstrap과 이 기사 macro 진단은 다른 가중 기준이며 하나의 신뢰구간으로 섞지 않는다.

모든 후보가 학습에 없던 84,879요청에서도 NDCG 차이는 +0.0378이었다. 정제된 train에 없는 사용자 3,688명의 42,213요청에서도 +0.0292였다. 이 사용자들에게는 제공된 과거 이력이 있으므로 이력 없는 사용자에 대한 개인화 성공으로 설명하지 않는다. '학습에 없는 기사'는 모델 fit 기준이며 플랫폼에 첫 등장한 순간만을 뜻하지 않는다.

## 4. 서빙에서 검증한 것

기존 FairJob의 Redis client, bounded scoring·shadow·배포 검증 구조, 예약 부하 측정 코드를 활용했다. 고정 광고 48피처 계약에 끼워 넣지 않고 `fairjob_ranker.ebnerd`에 별도 계약을 구현했다.

- 모델 bundle과 catalog/history snapshot을 분리했다. 학습에 없던 ID도 catalog에 있으면 점수화한다. 128후보 계약 테스트를 통과했다.
- HTTP 요청 201개의 모든 반환 점수·순위가 offline 예측과 일치했다. 최대 점수 차이 **{s['maximum_offline_http_score_error']:.1e}**. 후보 배열을 뒤집어도 결과가 같다.
- Redis MGET 값을 검증된 로컬 snapshot과 대조한다. 피처는 메모리의 같은 snapshot에서 계산하며 cache 손상·중단 시 이를 fallback으로 쓴다. Redis가 모든 온라인 원천 로그를 실시간 수집하는 시스템은 아니다.
- 별도 shadow 프로세스와 bounded queue, 모델·state에 묶인 품질/성능 근거 검사, warmup 후 atomic 교체, 이전 모델 rollback을 구현했다.
- 잘못된 snapshot의 교체 거절, 인증 없는 관리 요청 거절, 실제 promote → rollback → promote, Redis 손상·중단·복구, shadow 강제 종료 후 본 요청 유지까지 확인했다.
- 새로운 MLflow `ebnerd-v1`의 champion은 **version {s['registry_champion_version']}**. FairJob `fairjob-v2@champion`은 version 13으로 유지했다.

| 요청/초 | 측정 길이 | 예약 요청 | 50ms 이내 성공 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
{load}

한 Mac에서 실행한 짧은 로컬 부하 검증이다. 대규모 배포의 최대 처리량·장시간 안정성을 증명한 수치는 아니다. 지연은 예약 시각부터 응답까지 측정했으며 실패·누락을 성공 지연 통계에서 숨기지 않았다. 두 측정 모두 실패·degraded·generator drop 0건이다. [실제 HTTP/장애/부하 결과](../ebnerd-serving/verification.json).

shadow 종료 검증의 첫 실행에서는 이미 종료된 손자 프로세스를 기다리는 검증 스크립트가 timeout으로 끝났다. 이를 강제 종료 후 부모가 회수하도록 수정했다. 애플리케이션 코드는 바꾸지 않았고 이미 완료된 부하 측정은 재사용했으며, HTTP 일치·교체·복구·shadow 격리는 다시 확인했다. 이 재사용 여부를 verification에 기록했다.

## 5. 결과를 해석할 때의 경계

**이 개선은 제공된 기사 메타데이터 snapshot을 사용한 오프라인 결과다.** 모든 기사의 수정시각이 평가 기간 이후여서 category/topics/entities가 당시와 같았는지 이 파일만으로 입증할 수 없다. 기사 본문·임베딩을 추가로 학습하지 않았지만, 메타데이터도 이 한계에서 자유롭지 않다. 시간에 맞춘 이력 처리만으로 전체 피처의 완전한 point-in-time 무누수를 주장하지 않는다.

행동 정보만 사용하는 대조, publication_time 안정성을 가정하는 최신성 대조, 메타데이터 snapshot을 추가로 가정하는 개인화 모델을 모두 공개했다. 최신성 모델에도 발행시각 불변성 가정이 남는다. 현재 결과를 엄격한 운영 검증으로 확대하려면 기사 metadata의 버전 로그와 feature arrival time이 필요하다.

이 최초 실험의 평가 기간은 7일이고 학습 seed는 42 하나다. 2개 시간 구간 재현과 사용자 bootstrap은 장기 계절성·모든 학습 변동성을 포괄하지 않는다. 기존 노출 정책으로 관측된 후보이며 클릭 없는 요청이 없으므로 온라인 CTR·정책 인과효과·광고 수익을 주장하지 않는다. 제공 snapshot 조건의 점수 차이는 확인했지만, 후속 감사에서도 누수 없는 운영 개선으로 인증하지 못했다.

## 6. 산출물과 검증

전용 테스트 **14 passed**, 연구 독립 검증 **{len(v['checks'])}개**, 실제 서빙 검증 **{len(s['checks'])}개** 통과. 연구 검증은 train 500요청·test 1,000요청 피처 재생성, 6개 최종 모델의 전체 예측 재로딩 일치, sklearn과 요청별 NDCG/AUC 대조, 독립 Top1/cold 집계, 원본 FairJob champion 유지를 포함한다. [검증 원문](verification.json).

최종 모델: `models/ebnerd-v1/final-personalized/`. 서비스는 검증 뒤 종료했다. [실행·재현 가이드](../../docs/EBNERD_GUIDE.md)에서 바로 시작할 수 있다.
'''
(out/'RESULTS.md').write_text(text)
keys=['historical_popularity','behavior','freshness','metadata','personalized','shuffled_history']
labels=['Fixed past popularity','Behavior only','+ Freshness','+ Article metadata','+ User history','Shuffled user history']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
values=[r['metrics'][k]['ndcg5'] for k in keys]
axes[0].barh(labels,values,color=['#9ca3af','#64748b','#60a5fa','#2563eb','#059669','#f59e0b']); axes[0].invert_yaxis()
axes[0].set(xlabel='NDCG@5',title='Frozen chronological holdout',xlim=(0,.68))
for i,value in enumerate(values): axes[0].text(value+.006,i,f'{value:.3f}',va='center')
days=[(k,q['ndcg_difference']) for k,q in r['cohorts'].items() if k.startswith('day_')]
points=[q['estimate'] for _,q in days]; lower=[q['estimate']-q['lower'] for _,q in days]; upper=[q['upper']-q['estimate'] for _,q in days]
axes[1].errorbar(range(len(days)),points,yerr=[lower,upper],fmt='o',color='#059669',capsize=4)
axes[1].axhline(0,color='#64748b',lw=1); axes[1].set_xticks(range(len(days)),[k[-5:] for k,_ in days],rotation=30)
axes[1].set(ylabel='NDCG@5 difference',title='User history vs metadata, by day')
fig.suptitle('EB-NeRD: freshness + aligned interests\nReleased metadata snapshot; observed-click ranking, not online CTR',fontsize=12)
fig.savefig(out/'results.png',dpi=180); plt.close(fig)
print('Wrote',out/'RESULTS.md')
