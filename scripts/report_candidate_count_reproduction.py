"""Report discovery decomposition and failed candidate-count replication honestly."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fairjob_ranker.io import read_json


ROOT = Path('reports/candidate-count-reproduction')


def ci(value):
    def fmt(x):
        return '0.00' if round(x, 2) == 0 else f'{x:+.2f}'
    return f"{fmt(value['estimate'])} [{fmt(value['ci95'][0])}, {fmt(value['ci95'][1])}]"


def effect(report, cohort='all_large', comparison='full-uniform-top1'):
    return report['cohorts'][cohort]['comparisons'][comparison]


def main():
    discovery, metrics = read_json(ROOT/'discovery.json'), read_json(ROOT/'metrics.json')
    matched = read_json(ROOT/'matched_models_discovery.json')
    pooled = metrics['pooled_oof']
    decomposition, influence = discovery['decomposition'], discovery['concentration']
    thin = discovery['label_blind_thinning']
    assert not metrics['large_effect_replication_supported']
    lines = ['# 후보 11개 이상에서의 이득: 이유 분석과 사용자 분할 재현', '',
             '**결론: 처음 관측한 큰 이득은 다른 사용자 분할에서 재현되지 않았다.** '
             '원래 train 사용자만 이용해 세 번의 외부 평가 분할과 내부 조기 종료 분할을 만들고, '
             '전체 피처·광고 피처만 모델을 각 seed 3개씩 총 18개 새로 학습했다.', '',
             f"새 사용자들의 OOF 결과에서 후보 11개 이상 전체 모델의 무작위 선택 대비 이득은 **{ci(effect(pooled))}건 / 1,000요청**이었다. "
             f"2–10개 후보 집단보다 이득이 더 큰지 직접 비교한 차이는 **{ci(pooled['large_minus_small_interaction']['all'])}건**이었다. "
             '두 구간 모두 0을 포함한다. 후보 수 11을 서비스 분기나 모델 승격 조건으로 채택하지 않았다.', '',
             '## 검증 질문과 해석 범위', '',
             '지표는 클릭 없는 요청까지 포함해, 과거 로그에서 클릭된 광고를 1위로 선택한 횟수 / 전체 다중 후보 요청 × 1,000이다. '
             '동점은 균등 선택의 기대값으로 계산하고, 세 학습 seed의 요청별 결과를 평균한다. 온라인 CTR·매출 또는 후보 추가의 인과 효과가 아니다.', '',
             '주 재현 기준은 모든 외부 fold에서 이득의 점추정이 양수이고, 합친 OOF 이득의 사용자 bootstrap 95% 하한이 0보다 큰 것이다. '
             '큰 후보 집단에서 이득이 더 크다는 주장은 별도의 large-minus-small 차이 구간으로 판단했다. '
             '한 집단의 유의성 여부와 다른 집단의 유의성 여부만 비교하지 않았다. 무작위 표시 집단과 광고 피처만 대조군은 보조 분석이다.', '',
             '[실행 전 계약](protocol.json) · [설정](../../configs/candidate_count_reproduction.json) · [전체 수치](metrics.json)', '',
             '## 1. 처음 큰 이득이 관측된 구조', '',
             '기존 발견 표본에서 요청당 이득은 정확히 다음 두 값의 곱이다.', '',
             '`이득 = 클릭이 있는 요청의 비율 × 클릭이 있는 요청 안에서 (모델 1위 포착률 − 무작위 1위 포착률)`', '',
             '| 기존 발견 집단 | 요청 | 평균 후보 수 | 클릭 있는 요청 비율 | 클릭 있는 요청 중 모델 1위 포착 | 무작위 포착 | 조건부 차이 | 전체 요청당 이득 / 1,000 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in [('all_small', '후보 2–10개'), ('all_large', '후보 11개 이상')]:
        c = discovery['cohorts'][name]
        conditional = c['conditional_on_any_click']
        lines.append(f"| {label} | {c['requests']:,} | {c['mean_candidates']:.2f} | {100*c['click_request_prevalence']:.2f}% | "
                     f"{100*conditional['full_top1']:.2f}% | {100*conditional['uniform_top1']:.2f}% | "
                     f"{100*conditional['gap_top1']:.2f}%p | {effect(discovery, name)['estimate']:+.2f} |")
    share = 100*decomposition['prevalence_component_per1000']/decomposition['total_large_minus_small_per1000']
    lines += ['', '여기서는 두 현상이 함께 있었다. 큰 후보 집단은 클릭이 있는 요청이 더 많았고, '
              '클릭 있는 요청 안에서 무작위 선택 대비 차이도 컸다. 다만 모델 자체의 조건부 1위 포착률은 큰 집단에서 더 낮았다. '
              '후보가 많을 때 무작위 선택의 성공 확률도 더 낮아진다는 점을 함께 봐야 한다.', '',
              f"대칭적인 산술 분해에서 집단 간 이득 차이 {decomposition['total_large_minus_small_per1000']:.2f}건은 "
              f"클릭 있는 요청의 비율 차이 항 {decomposition['prevalence_component_per1000']:.2f}건({share:.1f}%)과 "
              f"조건부 선택 차이 항 {decomposition['conditional_gap_component_per1000']:.2f}건으로 나뉘었다. "
              '**이 비율은 정해진 분해식의 결과이며 원인 기여율이 아니다.** 사용자·노출 구성 차이가 두 항에 함께 영향을 줄 수 있다.', '',
              '관측 라벨을 미리 안다는 상한에서 무작위 기준을 뺀 값을 개선 여지로 정의한 뒤, 실제 이득을 그 값으로 나누는 점검도 했다. '
              f"그 비율은 작은 집단 {100*discovery['cohorts']['all_small']['fraction_of_label_oracle_headroom_used']['estimate']:.2f}%, "
              f"큰 집단 {100*discovery['cohorts']['all_large']['fraction_of_label_oracle_headroom_used']['estimate']:.2f}%였다. "
              '이 상한은 라벨을 사용한 진단용 기준이며 배포 가능한 정책이 아니다.', '',
              '## 2. 특정 사용자와 후보 구성에 민감했는가?', '',
              f"큰 후보 집단의 양의 사용자별 순기여 합 중 상위 5명의 몫은 **{100*influence['top5_share_of_positive_user_net_contributions']:.1f}%**였다. "
              f"이 5명을 제외한 민감도 분석에서는 이득이 +5.20에서 **{influence['excluding_five_most_positive_users_effect_per1000']:+.2f}건 / 1,000요청**으로 줄었다. "
              f"사용자 한 명씩만 제외했을 때는 {influence['leave_one_user_out_effect_per1000_min']:+.2f}~{influence['leave_one_user_out_effect_per1000_max']:+.2f}건이었다. "
              '한 사용자만으로 전체 현상이 생겼다는 뜻은 아니지만, 소수 사용자들의 영향이 컸다. 기여도가 높은 사용자를 결과를 보고 골라 제거했으므로 별도의 검증 표본으로 취급하지 않는다.', '',
              f"선택된 관측 클릭 기여의 상위 5개 광고 비중은 {100*influence['top5_product_share_of_selected_click_credit']:.1f}%였다. "
              f"원래 표시 위치 1에서 발생한 몫은 선택된 클릭 기여 중 {100*influence['share_selected_click_credit_at_original_rank1']:.2f}%, "
              f"이 집단의 전체 관측 클릭 중 {100*influence['share_all_observed_clicks_at_original_rank1']:.2f}%였다. "
              '따라서 단순히 원래 1위에서 클릭된 광고만 골랐다고 설명할 수는 없다. 이것이 다른 위치 편향까지 제거됐다는 증거는 아니다.', '',
              '같은 큰 후보 요청에서 라벨과 점수를 보지 않고 후보 5개를 무작위로 남기는 점검도 요청당 50회 실행했다. '
              f"1,000요청당 이득은 **+5.20 → {thin['delta_per1000']:+.2f}건**, 클릭 있는 요청 비율은 "
              f"**9.30% → {100*thin['click_request_prevalence']:.2f}%**가 됐다. "
              f"반면 클릭 있는 요청에서 무작위 대비 조건부 차이는 **5.59%p → {100*(thin['conditional_full_top1']-thin['conditional_uniform_top1']):.2f}%p**였다. "
              '요청당 이득의 크기는 관측 클릭이 포함된 기회의 수와 기준 정책에도 민감하다. 과거 라벨을 고정한 thinning 진단이며, '
              '실제 서비스에서 광고를 줄이거나 늘렸을 때의 사용자 반응을 실험한 것은 아니다.', '',
              '기존 발견 표본에서도 무작위 표시 요청만 보면 large-minus-small 차이는 '
              f"{ci(discovery['large_minus_small_interaction']['randomized'])}건이었다. "
              '큰 후보 집단이 더 유리하다는 차이를 이 집단에서 확인하지 못했다.', '',
              '## 3. 다른 사용자 분할에서 모델을 실제로 다시 학습', '',
              '기존 발견 validation 사용자는 새 실험의 학습·조기 종료·평가에서 모두 제외했다. '
              f"원래 train의 **{metrics['replication_pool']['users']:,}명, {metrics['replication_pool']['rows']:,}행**을 "
              '사용자 ID 해시로 3개 외부 fold로 나눴다. 각 fold를 한 번씩 평가용으로 남기고, 나머지 사용자들을 80/20으로 다시 나눠 학습과 조기 종료에 사용했다.', '',
              '학습 피처 변환은 해당 학습 사용자로만 적합했다. LightGBM 설정은 기존 기준 모델과 같고, 체크포인트는 내부 LogLoss로만 선택했다. '
              'fold마다 전체 피처·광고 피처만 모델 × seed 3개를 고정한 뒤 외부 평가를 진행했다. 컨텍스트만 모델의 순위는 모두 동점이므로 '
              '무작위 선택 기대값을 직접 비교 기준으로 사용했다.', '',
              '| 외부 fold | 학습 사용자 | 조기 종료 사용자 | 평가 사용자 | 학습 행 |',
              '|---|---:|---:|---:|---:|']
    for audit in metrics['split_audits']:
        lines.append(f"| {audit['fold']+1} | {audit['fit_users']:,} | {audit['inner_users']:,} | {audit['held_users']:,} | {audit['fit_rows']:,} |")
    lines += ['', '모든 원래 train 사용자는 자신이 학습·조기 종료에 포함되지 않은 모델로 한 번씩 평가됐다. '
              '이전에도 사용한 공개 개발 데이터의 재분할이므로 새로운 외부 데이터라고 부르지 않는다. '
              '원래 test는 읽지 않았다. fold 모델들의 학습 사용자는 서로 겹치므로 fold 결과를 독립적인 실험 반복으로 간주하지 않는다. '
              'bootstrap 구간은 고정된 학습 모델에 조건부이며 전체 학습 변동성을 추정하지 않는다.', '',
              '## 4. 재현 결과', '',
              '| 평가 | 후보 11개 이상 요청 | 전체 모델 − 무작위 / 1,000요청 [95% 구간] |',
              '|---|---:|---:|',
              f"| 기존 발견 표본, 기존 모델 | {discovery['cohorts']['all_large']['requests']:,} | {ci(effect(discovery))} |"]
    for fold, report in metrics['folds'].items():
        lines.append(f"| 새 사용자 fold {int(fold)+1} | {report['cohorts']['all_large']['requests']:,} | {ci(effect(report))} |")
    lines += [f"| 새 사용자 OOF 합계 | {pooled['cohorts']['all_large']['requests']:,} | **{ci(effect(pooled))}** |", '',
              '표의 기존 발견 구간은 이번과 같은 bootstrap 2,000회로 다시 계산했다. 이전 보고서는 4,000회였으므로 구간 끝값은 조금 다를 수 있다. '
              '점추정은 같다. 재현 기준인 모든 fold 양의 이득과 OOF 구간 하한 양수를 모두 충족하지 못했다.', '',
              '모든 새 사용자를 합친 비교:', '',
              '| 평가 집단 | 요청 | 전체 모델 − 무작위 top-1 [95% 구간] | 전체 모델 − 광고 피처만 top-1 [95% 구간] |',
              '|---|---:|---:|---:|']
    for name, label in [('all_large', '전체, 후보 11개 이상'), ('all_small', '전체, 후보 2–10개'),
                        ('randomized_large', '무작위 표시, 후보 11개 이상'), ('randomized_small', '무작위 표시, 후보 2–10개')]:
        lines.append(f"| {label} | {pooled['cohorts'][name]['requests']:,} | {ci(effect(pooled, name))} | {ci(effect(pooled, name, 'full-ad_only-top1'))} |")
    lines += ['', f"전체 OOF의 large-minus-small 차이: **{ci(pooled['large_minus_small_interaction']['all'])}건**. "
              f"무작위 표시 OOF의 같은 차이: **{ci(pooled['large_minus_small_interaction']['randomized'])}건**.", '',
              '무작위 표시·큰 후보 집단에서 전체 모델이 광고 피처만 모델보다 높은 보조 구간을 보였지만, '
              '무작위 선택 대비 주 재현이나 후보 수에 따른 차이를 입증한 것으로 대체하지 않는다. 작은 후보 집단의 양의 보조 결과도 마찬가지다. '
              '이런 보조 비교에는 다중 비교 보정을 하지 않았다.', '',
              '## 5. 학습량 감소만으로 재현 실패를 설명할 수 있는가?', '',
              '새 fold 모델의 학습 행은 약 44만~45만으로, 기존 발견 모델의 약 67만 행보다 적다. '
              '따라서 모델과 평가 집단이 동시에 달라졌다는 한계가 있다. 이를 점검하려고 주 결과를 확인한 뒤, '
              '**동일한 새 모델들을 기존 발견 사용자에게도 적용하는 진단**을 추가했다. 추가 재학습이나 체크포인트 변경은 없었다.', '',
              '| 동일한 새 모델 묶음 | 새 외부 사용자에서의 이득 | 기존 발견 사용자에서의 이득 |',
              '|---|---:|---:|']
    for fold in range(3):
        fresh = effect(metrics['folds'][str(fold)])
        previous = effect(matched['fold_models_on_discovery'][str(fold)])
        lines.append(f"| fold {fold+1}의 모델 3개 | {ci(fresh)} | {ci(previous)} |")
    matched_pooled = effect(matched['nine_model_outcome_mean_on_discovery'])
    lines += ['', f"새 모델 9개의 요청별 결과를 평균해 기존 발견 사용자에서 평가하면 **{ci(matched_pooled)}건**이었다. "
              '같은 발견 표본에서도 학습 사용자 구성에 따라 결과가 달라졌고, 같은 새 모델도 평가 사용자 집단에 따라 점추정이 달라졌다. '
              '**학습량 감소만 또는 평가 사용자 구성만이 원인이라고 단정할 수 없다.** 후보 11개 이상이라는 조건에서 안정적으로 우수하다는 설명이 '
              '현재 증거에 맞지 않는다는 결론은 유지한다. [사후 진단 계약](matched_models_protocol.json), [사후 진단 수치](matched_models_discovery.json).', '',
              '## 프로젝트 판단과 스토리', '',
              '> 후보가 많은 요청에서 좋아 보이는 결과를 발견했지만, 곧바로 그 집단에 모델을 적용하는 규칙을 만들지 않았습니다. '
              '클릭 있는 요청의 비율과 조건부 선택 능력을 분해했고, 일부 사용자의 영향도 확인했습니다. '
              '이후 발견에 쓰지 않은 사용자들로 세 번의 분할과 재학습을 수행했지만, 큰 후보 집단의 이득은 재현되지 않았습니다. '
              '후보 수에 따른 배포 규칙을 보류하고, 모델·사용자 구성에 민감한 발견으로 기록했습니다.', '',
              '원래 목표인 광고 선택 품질을 판단하는 과정에서, 하나의 validation 하위집단 결과를 일반적인 서비스 규칙으로 만들 근거가 부족함을 확인했다. '
              '새 데이터를 확보하기 전까지 11개라는 경계를 최적화하거나 다른 경계를 탐색해 성공 결과로 바꾸지 않는다. '
              '이 검증은 공개 로그의 위치 영향·후보 누락·샘플링 문제를 해결하지 않으며, 실제 CTR·매출 효과도 추정하지 않는다.', '',
              '## 재현·검증', '',
              '```bash',
              'python3 scripts/run.py scripts/candidate_count_reproduction.py \\',
              '  --output /tmp/fairjob-count-replay --models /tmp/fairjob-count-models --tracking-uri ""',
              'python3 scripts/run.py scripts/verify_candidate_count_reproduction.py',
              'python3 scripts/run.py scripts/report_candidate_count_reproduction.py',
              'python3 scripts/run.py -m pytest tests/test_candidate_count_reproduction.py tests/test_business_validation.py -q',
              '```', '',
              '첫 명령의 두 출력 경로는 아직 존재하지 않아야 한다. 검증·보고서 명령은 기본 보고서 경로를 읽는다. '
              '추가 사후 진단 코드는 [check_replication_cohort_sensitivity.py](../../scripts/check_replication_cohort_sensitivity.py)에 있다. '
              '사후 진단도 기존 출력 파일을 덮어쓰지 않는다.', '',
              '신규 테스트 2개와 관련 기존 테스트 4개가 통과했다. 세 fold의 학습 전용 피처 변환을 독립적으로 다시 적합해 확인하고, '
              '18개 저장 모델을 재로딩해 각 fold의 완전한 요청 100개에서 점수 및 별도 구현의 top-k 계산을 비교했다. '
              'MLflow에 18개 실행을 완료 상태로 기록했으며 기존 champion 13은 유지했다. 원래 test 접근과 기존 모델 변경은 없다. '
              '[검증 기록](verification.json). 기존 전체 테스트 스위트는 이번에 재실행하지 않았다.', '',
              '![후보 수 효과 재현 결과](replication.png)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

    entries = [('Discovery users / original models', effect(discovery), '#888888')]
    entries += [(f'New users: fold {fold+1}', effect(metrics['folds'][str(fold)]), '#1565c0') for fold in range(3)]
    entries += [('New users: pooled OOF', effect(pooled), '#003e7e'),
                ('Discovery users / new models (post hoc)', matched_pooled, '#c47120')]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for y, (label, value, color) in enumerate(entries):
        estimate, (lo, hi) = value['estimate'], value['ci95']
        ax.errorbar(estimate, y, xerr=[[estimate-lo], [hi-estimate]], fmt='o', color=color, capsize=4, markersize=7)
        ax.text(hi+.15, y, f'{estimate:+.2f}', va='center', fontsize=10, color=color)
    ax.set_yticks(range(len(entries)), [r[0] for r in entries])
    ax.set_ylim(len(entries)-.5, -.5)
    ax.set_xlim(-2.3, max(v['ci95'][1] for _, v, _ in entries)+1.4)
    ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=.18)
    ax.set_xlabel('Extra historically clicked ads selected at #1 per 1,000 requests\nFull model minus uniform; candidate count >= 11')
    ax.set_title('The large-candidate advantage did not replicate', fontsize=14, fontweight='bold', pad=16)
    fig.text(.5, .025, 'Fixed-model user-cluster 95% intervals. Existing public development data; not online CTR lift.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(ROOT/'replication.png', dpi=170)
    plt.close(fig)
    print(ROOT/'RESULTS.md')


if __name__ == '__main__':
    main()
