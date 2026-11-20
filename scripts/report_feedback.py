"""Render the fixed experiment's actual outcome, including failed conditions."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-top1-feedback'
r=json.loads((OUT/'final_results.json').read_text());v=json.loads((OUT/'verification.json').read_text())
s=json.loads((OUT/'selection.json').read_text()); chosen=r['selection'];passed=r['also_beats_exposure_ndcg_overall_and_cold']
fmt=lambda x:'—' if x is None else f'{100*x:.2f}%'
metrics=r['metrics']; chosen_key='d3600_'+chosen
def rate(key,seg):return fmt(metrics[key][seg]['hit1'])
lines=['# 과거 클릭 반응과 1위 선택: 새 사용자 검증','',
    '**판정: '+('1시간 피드백 성숙을 가정한 재현에서는 사전 기준을 통과했다.' if passed else '사전에 정한 전체·cold 동시 개선 기준을 통과하지 못했다.')+'**',
    '실제 피드백 수신 시각은 없어 운영 가용성을 인증한 결과는 아니다. 기사 메타데이터는 사용하지 않았다. 완전 무이력 아이템에 대한 성과와 반응이 쌓인 아이템의 성과를 분리한다.','',
    f"기존 small의 모든 사용자와 겹치지 않는 **{r['cohort']['users']:,}명**, **{r['cohort']['requests']:,}개 요청**, **{r['cohort']['candidates']:,}개 후보 행**. 2023-05-25~06-01의 같은 주간이므로 새 기간 검증은 아니다. 예측과 코드 해시를 고정한 후 정답을 한 번 평가했다.",'',
    f"**전체 cold 개선과 무이력 아이템 개선은 다르다.** 무이력 정답 선택은 {rate('d3600_exposure_ndcg','zero_signal_target')} → {rate(chosen_key,'zero_signal_target')}로 악화했다. 이 조건의 균등 선택은 {rate('uniform','zero_signal_target')}다. 현재 모델은 반응이 확인된 후보에 치우친다. 알려지지 않은 후보 집합에 기회를 더 주는 정책은 검토할 수 있지만, 집합 안에서 입력이 같은 개별 후보를 구별할 정보는 없다.",'',
    '## 1시간 지연 조건의 실제 결과','',
    '| 모델 | 전체 1위 선택률 | 학습에 없던 정답 아이템을 1위로 선택 |',
    '|---|---:|---:|']
names={'exposure_ndcg':'노출 통계 + 과거 사용자 이력','exposure_top1':'같은 피처, Top1 중심 학습',
       'ctr_1h':'직전 1시간 평활 클릭률 규칙','ctr_24h':'직전 24시간 평활 클릭률 규칙',chosen:'선택된 클릭 반응 모델'}
for arm in ['exposure_ndcg','exposure_top1','ctr_1h','ctr_24h',chosen]:
    key='d3600_'+arm;lines.append(f'| {names[arm]} | {rate(key,"overall")} | {rate(key,"cold_target")} |')
lines+=['','1위 선택률은 관측된 클릭을 맞히는 비율이다. CTR·매출 증가율이 아니다. Cold 열은 cold 정답이 있는 요청에서 **그 cold 정답**을 선택해야 성공이며, 기존 후보도 모두 경쟁한다.','',
        '## 비교와 불확실성','', '| 선택 모델 대비 기준 | 전체 차이 (95% CI, %p) | Cold 정답 차이 (95% CI, %p) |','|---|---:|---:|']
for arm in ['exposure_ndcg','exposure_top1','ctr_1h','ctr_24h']:
    c=r['paired_user_bootstrap'][f'{chosen_key}-minus-d3600_{arm}']
    cells=[]
    for seg in ['overall','cold_target']:
        a=c[seg];cells.append(f"{100*a['estimate']:+.2f} [{100*a['lower']:+.2f}, {100*a['upper']:+.2f}]")
    lines.append(f'| {names[arm]} | {cells[0]} | {cells[1]} |')
lines+=['','사용자 단위 paired bootstrap 2,000회. 같은 주간·참조 스트림에 조건부인 구간이며, 날짜나 전체 서비스로의 일반화를 인증하지 않는다.','',
        '## 새 아이템과 정보의 한계','', '| 조건 | 요청 수 | 선택 모델 | 노출 기반 기준 | 균등 선택 |','|---|---:|---:|---:|---:|']
segments={'cold_target':'학습에 없던 아이템 정답','all_cold':'모든 후보가 학습에 없던 아이템',
          'adapted_cold_target':'이전 참조 노출이 있는 cold 정답','zero_feedback_target':'이전 참조 노출이 없는 cold 정답',
          'zero_signal_target':'참조 노출·전역 및 사용자 클릭 이력에도 없는 정답',
          'all_candidates_zero_signal':'모든 후보에 위 신호가 없음',
          'cold_1to4_exposures':'Cold 정답: 이전 노출 1–4회','cold_5to19_exposures':'Cold 정답: 이전 노출 5–19회',
          'cold_20plus_exposures':'Cold 정답: 이전 노출 20회 이상'}
for seg,label in segments.items():
    lines.append(f'| {label} | {metrics[chosen_key][seg]["requests"]:,} | {rate(chosen_key,seg)} | {rate("d3600_exposure_ndcg",seg)} | {rate("uniform",seg)} |')
z=r['zero_signal_indistinguishability']
lines+=['',f"신호가 전혀 없는 후보가 여럿 존재한 **{z['queries_with_multiple_identical_zero_signal_candidates']:,}개 요청**에서는 그 후보들의 38개 입력이 정확히 같았다. 같은 입력으로 후보를 구별할 수 없다. 해당 표현과 균등 동점 처리에 한정한 zero-signal 정답 선택의 oracle 상한은 {fmt(z['zero_signal_target_hit1_oracle_upper_bound'])}다. 실제 모델 성과나 모든 모델의 상한이 아니다.",'',
        '## 피드백이 늦거나 멈추면','', '| 같은 선택 모델의 입력 조건 | 전체 | Cold 정답 |','|---|---:|---:|']
for delay,label in [(3600,'1시간 후 참조 피드백 사용'),(86400,'온라인 참조 피드백만 24시간 지연'),('frozen','평가 전 통계 고정, 이후 참조 피드백 없음')]:
    key=f'd{delay}_{chosen}';lines.append(f'| {label} | {rate(key,"overall")} | {rate(key,"cold_target")} |')
lines+=['','24시간·고정 조건은 같은 모델의 입력을 늦추거나 멈춘 민감도 실험이다. 해당 조건으로 별도 최적화한 모델이 아니다. Cold 부분집합은 1시간 기준으로 고정해 같은 요청을 비교했다. 고정 조건에는 평가 주간의 참조 클릭이 한 건도 들어가지 않는다.','',
        '## 실제 프로젝트 스토리','',
        '1. 기사 스냅샷을 넣으면 점수는 높았지만, 그 값이 노출 당시 존재했는지 입증할 수 없었다. 기사 파일을 제거했다.',
        '2. 노출 빈도만으로는 어떤 후보가 관심을 받았는지 부족했다. Top1에 학습을 집중하면 해결될 것이라는 가설도 비교했다.',
        f"3. 개발 Hit1은 노출 기반 {fmt(s['mean_development_hit1']['exposure_ndcg'])}, Top1 중심 {fmt(s['mean_development_hit1']['exposure_top1'])}, 클릭 반응 추가 {fmt(s['mean_development_hit1']['feedback_ndcg'])}, 둘을 함께 적용 {fmt(s['mean_development_hit1']['feedback_top1'])}였다. 이 설정에서는 정보 추가가 효과적이었고, Top1 중심 설정만 바꾸는 것은 실패했다.",
        '4. 반복해서 본 기존 테스트 대신 새 사용자에서 전체 후보·cold 정답·피드백 지연을 함께 평가했다. 반응이 쌓인 아이템의 적응을 첫 노출부터의 콘텐츠 일반화라고 포장하지 않는다.','',
        '## 다음 행동에 대한 판단','',
        ('- EB-NeRD는 **메타데이터 없는 반응 적응** 연구에 계속 쓸 근거가 생겼다. 더 큰 모델이나 다른 데이터 다운로드가 첫 해결책은 아니다.' if passed else '- 이 설정으로는 전체·cold 동시 개선을 인증하지 못했다. 같은 평가 정답으로 더 튜닝하지 않는다.'),
        '- **완전 신규 아이템을 첫 노출부터 구별하는 목표**에는 요청 당시 후보별 콘텐츠 등 추가 정보가 필요하다. 현재의 동일한 입력을 모델 크기만으로 해결할 수 없다.',
        '- 무작정 MIND·KuaiSAR 등으로 바꾸면 후보군 또는 스냅샷 시각 문제가 다시 생길 수 있다. 다음 데이터는 서빙 시점 피처 스냅샷·버전과 실제 수신 시각을 통과 조건으로 한다.',
        '- 그 수집 계약을 `schemas/ranking-request-v1.json`, `schemas/ranking-feedback-v1.json`과 검증 코드로 구현했다. EB-NeRD에 없는 수신 시각을 만들어 넣지는 않는다.',
        '- 기존 서빙의 요청 검증·동시성 제한·버전 고정·롤백 구조는 재사용한다. 기사 카탈로그 필수 조인과 미등록 후보 삭제는 새 경로에서 제거해야 한다. 실제 이벤트 수신 시각으로 상태를 갱신하는 어댑터와 별도 HTTP 부하 검증 전에는 기존 champion으로 승격하지 않는다.','',
        '## 검증과 남은 조건','',
        f"- 독립적인 배열 검색으로 반응 피처 {v['response_feature_values_checked']:,}개 값을 재계산: 최대 오차 {v['max_absolute_error']}.",
        f"- 요청별 지표 {v['independent_query_metrics_checked']:,}개를 직접 계산하고, 재로딩한 단일 스레드 예측 {v['reloaded_single_thread_predictions_exact']:,}개를 대조했다.",
        '- 관련 테스트 17개: 성숙 시간 경계, 미래/현재 정답 불변성, 후보 순서, 중복 클릭, cold 정답의 전체 후보 경쟁, 실제 수신 시각·스냅샷 검증.',
        '- 24개 개발 학습 + 12개 최종 학습. 정답 경계·동점 처리 수정 전 개발 산출물은 제외하고 별도 보관했다.',
        '- 실제 피드백 완료/수신 시각 및 고정 이력 생성 시각은 미제공이다. 따라서 이 결과의 정확한 이름은 **사후 기사 메타데이터를 배제하고 시각 경계를 검증한 조건부 재현 실험**이다.',
        '- 모델을 운영 registry에 승격하지 않았고, 새 시간대나 온라인 실험은 수행하지 않았다.','',
        '자세한 가정·데이터 대안·재현 명령: [가이드](../../docs/EBNERD_TOP1_FEEDBACK.md). 프로토콜, 원본 해시, 예측 동결, 지표와 검증 결과는 이 폴더의 JSON 파일에 있다.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
policy_dir=ROOT/'reports/ebnerd-cold-policy'
if (policy_dir/'continuous_feasibility.json').exists():
    policy=json.loads((policy_dir/'development.json').read_text())
    bounds=json.loads((policy_dir/'continuous_feasibility.json').read_text())
    extra=['','## 후속 반증: 무이력 후보에 선택 기회를 주면 해결되는가','',
        '첫 새 사용자 검증에서 무이력 정답 성능이 악화한 것을 보고 후속 가설을 세웠다. 그 평가 정답으로 비율을 고르지 않고 기존 두 개발 구간으로 돌아가, 무이력 후보가 있는 요청에서 일정 확률로 그 후보 중 하나를 균등 선택하는 정책을 시험했다. 입력으로 현재 정답을 사용하지 않는다.',
        f"사전에 정한 8개 확률(0, 2, 5, 10, 20, 30, 40, 50%) 중 두 구간 모두에서 전체·cold·무이력 정답 지표를 각 기준보다 **0.25%p 이상** 높인 설정은 {'있었다' if policy['gate_passed'] else '없었다'}.",
        f"이 정책의 기대 성능은 탐색 확률에 선형이다. 연속 확률도 검산하니, 세 조건을 충족하려면 최소 {100*bounds['minimum_alpha']:.2f}%이면서 최대 {100*bounds['maximum_alpha']:.2f}%여야 해 교집합이 없었다. **이 고정 균등 탐색 정책과 0.25%p 기준에 한정한 실패**이며 모든 routing·calibration 방법이 불가능하다는 뜻은 아니다.",
        '개발 기준에서 탈락했으므로 두 번째 새 사용자 집합의 정답은 평가하지 않았다. 기존 첫 평가 결과를 다시 튜닝해 성공으로 만들지 않았다. [프로토콜과 개발 수치](../ebnerd-cold-policy/development.json), [연속 확률 검산](../ebnerd-cold-policy/continuous_feasibility.json).']
    with (OUT/'RESULTS.md').open('a') as f:f.write('\n'.join(extra)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13,4.8))
for ax,(segment,title) in zip(axes,[('overall','Overall Top1'),('cold_target','Clicked cold item at Top1'),('zero_signal_target','Zero-signal clicked item at Top1')]):
    arms=['exposure_ndcg','ctr_24h',chosen];values=[metrics['d3600_'+a][segment]['hit1']*100 for a in arms]
    bars=ax.bar(['Exposure\nmodel','Smoothed\nCTR 24h','Response\nmodel'],values,color=['#8b9db3','#d9a155','#227d76'])
    for b,value in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,value+.3,f'{value:.2f}%',ha='center')
    ax.set_ylim(0,max(values)*1.24+1);ax.set_title(title);ax.set_ylabel('Observed-hit rate (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Fresh users, same week — no article metadata',fontweight='bold')
fig.text(.5,.01,'1-hour feedback maturity assumed. Full candidate slates. Cold is model-unseen, not necessarily newly published. Not causal CTR lift.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(OUT/'results.png',dpi=170);plt.close(fig)
decision={'generated_utc':datetime.now(timezone.utc).isoformat(),'bounded_ebnerd_response_experiment_passed':passed,
    'article_snapshot_used':False,'new_temporal_holdout':False,'new_users':r['cohort']['users'],
    'true_zero_shot_content_problem_solved':False,'production_availability_certified':False,'registry_promoted':False,
    'next_direction':'Use verified arrival-time response features for adaptation; require at-request/versioned candidate content for zero-history items. No blind public-dataset substitution.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision),flush=True)
