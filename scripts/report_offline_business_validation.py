"""Render business-facing selection evidence without converting it into online lift."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fairjob_ranker.io import read_json


ROOT = Path('reports/offline-business-validation')
LABELS = {'uniform': '무작위 선택 / 컨텍스트 모델의 동점 평균', 'ad_only': '광고 피처만',
          'full': '전체 피처 LightGBM', 'ad_residual': '컨텍스트 + 광고 보정'}
COHORTS = {'randomized_multi': '무작위 표시 집단 (주 분석)', 'all_multi': '전체 다중 후보 집단 (보조)',
           'randomized_observed_rank_contiguous': '무작위 표시 + 관측 위치 1..n',
           'all_multi_2_to_5': '후보 2–5개', 'all_multi_6_to_10': '후보 6–10개',
           'all_multi_11_plus': '후보 11개 이상'}


def fmt_ci(value):
    def number(v):
        return '0.00' if round(v, 2) == 0 else f'{v:+.2f}'
    return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}]"


def main():
    m = read_json(ROOT/'metrics.json')
    primary = m['primary_result']
    randomized = m['cohorts']['randomized_multi']
    lines = [
        '# 오프라인 비즈니스 검증: 더 좋은 광고를 실제로 고르는가?', '',
        '**결론: 전체 피처 모델의 점추정은 무작위 선택보다 높았지만, 사전에 정한 주 분석의 개선 기준을 충족하지 못했다.**' if not m['primary_proxy_success'] else
        '**결론: 사전에 정한 주 분석에서 관측 클릭 포착의 개선 기준을 충족했다. 온라인 정책 효과까지 검증한 것은 아니다.**', '',
        f"무작위 표시 집단의 요청 1,000건당 1위에서 포착한 관측 클릭 광고 차이는 **{fmt_ci(primary)}건**이었다. "
        '구간은 사용자 단위 paired bootstrap 95% 구간이다. 성공 기준은 하한이 0보다 큰 것이었다. '
        '실제 서비스 CTR·전환·매출 상승을 추정하거나 모델을 자동 교체하지 않았다.', '',
        '## 사업상 질문을 측정 가능한 지표로 바꾸기', '',
        '한 요청에 상위 광고 자리 하나를 배정한다고 할 때, 모델이 고른 광고가 과거 로그에서 클릭된 광고였는지를 센다. '
        '한 번의 선택 기회를 분모에 넣어, 요청 1,000건당 관측 클릭 광고를 몇 건 포착했는지 표시한다. '
        '상위 3개 자리도 같은 방식으로 별도 평가한다. 후보가 3개보다 적으면 관측 후보 수만큼 선택한다.', '',
        '| 지표 | 사업상 연결 | 이 실험에서의 의미 |',
        '|---|---|---|',
        '| 요청 1,000건당 1위 관측 클릭 포착 | 한 자리 배분의 선택 품질 | 1위에 고른 광고의 기존 click 라벨 합 / 전체 평가 요청 × 1,000 |',
        '| 요청 1,000건당 상위 3개 관측 클릭 포착 | 여러 자리를 채울 때의 선택 품질 | 상위 3개에 들어간 기존 클릭 광고 수 / 전체 평가 요청 × 1,000 |',
        '| 관측 클릭 포착률 | 로그에 있는 클릭 광고를 얼마나 상위로 모았는가 | 선택된 관측 클릭 수 / 해당 집단의 전체 관측 클릭 수 |',
        '| 실제 서비스 CTR·지원 전환·매출 | 최종 사업 효과 | 이번 공개 로그만으로 변화량을 식별하지 못함 |', '',
        '**앞의 지표는 온라인 CTR이 아니다.** 기존 위치에서 발생한 클릭 라벨로 다른 정렬을 평가한 관측값이다. '
        '상위로 옮겼을 때의 클릭 변화, 광고 간 경쟁 효과, 로그에서 빠진 후보, 비균일 샘플링을 복원하지 않는다. '
        '금액을 곱해 예상 매출로 표시하지 않았다.', '',
        '## 결과를 보기 전에 고정한 비교', '',
        '- 기존 전체 피처 모델 대 무작위 선택의 무작위 표시 집단 top-1 결과를 주 비교로 고정했다.',
        '- 기존 seed 42/43/44의 모델·보정 강도를 그대로 사용했다. 재학습·최적 seed 선택·original test 재사용은 없다.',
        '- 각 seed의 요청별 결과를 평균했다. 예측 점수 평균으로 새 앙상블을 만든 것이 아니다.',
        '- 동점에서는 동점 광고 사이의 균등한 선택을 정확히 평균했다. 컨텍스트 모델은 요청 안에서 모두 동점이라 무작위 선택 기준과 같다.',
        '- 클릭 없는 요청과 모든 후보가 클릭된 요청도 포함한다. 후보 하나뿐인 요청은 광고 간 선택이 없으므로 제외한다.',
        '- 같은 사용자의 모든 요청을 함께 복원 추출하는 paired bootstrap 4,000회로 구간을 계산했다. 요청 수가 다른 사용자는 원래 요청 수를 유지한다.',
        '- 광고 피처만 모델, 광고 보정 모델, top-3, 후보 수·관측 위치 하위집단은 보조 비교다. 다중 비교 보정은 하지 않았다.', '',
        '이 분석은 이미 사용한 development validation의 새로운 평가 항목이다. 새 독립 holdout 실험으로 취급하지 않는다. '
        '[실행 전 계약](protocol.json), [설정](../../configs/offline_business_validation.json), [전체 수치](metrics.json).', '',
        '## 평가 대상', '',
        '| 집단 | 요청 | 사용자 | 관측 클릭 | 클릭 없는 요청 |',
        '|---|---:|---:|---:|---:|']
    for name in ['randomized_multi', 'all_multi', 'randomized_observed_rank_contiguous']:
        c = m['cohorts'][name]
        lines.append(f"| {COHORTS[name]} | {c['requests']:,} | {c['users']:,} | {c['observed_clicks']:,} | {c['zero_click_requests']:,} |")
    lines += ['', f"원래 validation은 {m['original_validation_requests']:,}개 요청이며, "
              f"그중 단일 후보 요청 {m['excluded_single_candidate_requests']:,}개를 제외했다. "
              '이전 NDCG의 클릭·비클릭 혼합 요청만 사용하는 분모와 다르다. '
              '무작위 표시 집단에서도 클릭이 발생한 요청은 69개다. 관측 위치가 1..n인 경우에도 공개 후보가 완전하다고 보장하지 않는다.', '']
    for name in ['randomized_multi', 'all_multi']:
        c = m['cohorts'][name]
        lines += [f'## {COHORTS[name]}', '',
                  '| 정책 | 1위 관측 클릭 / 1,000요청 | 상위 3개 관측 클릭 / 1,000요청 | 관측 클릭 중 1위 포착률 |',
                  '|---|---:|---:|---:|']
        for policy in m['config']['policies']:
            p = c['policies'][policy]
            lines.append(f"| {LABELS[policy]} | {p['1']['observed_clicked_ads_per_1000_requests']['estimate']:.2f} | "
                         f"{p['3']['observed_clicked_ads_per_1000_requests']['estimate']:.2f} | "
                         f"{p['1']['fraction_of_observed_clicks_captured']['estimate']*100:.2f}% |")
        lines += ['', '정책 간 차이와 사용자 bootstrap 95% 구간. 단위는 관측 클릭 광고 / 1,000요청이다.', '',
                  '| 비교 | 1위 차이 [95% 구간] | 상위 3개 차이 [95% 구간] |', '|---|---:|---:|']
        for candidate, reference in m['config']['comparisons']:
            d = c['comparisons'][candidate+' - '+reference]
            lines.append(f"| {LABELS[candidate]} − {LABELS[reference]} | "
                         f"{fmt_ci(d['1']['extra_observed_clicked_ads_per_1000_requests'])} | "
                         f"{fmt_ci(d['3']['extra_observed_clicked_ads_per_1000_requests'])} |")
        lines += ['']
    lines += ['## 해석과 모델 선택에 미치는 영향', '',
              '1. **1위 선택의 우위를 확정하지 못했다.** 전체 모델이 두 주요 집단에서 더 많은 관측 클릭 광고를 1위로 올렸지만, '
              '무작위 선택 대비 top-1과 top-3 차이의 구간은 모두 0을 포함했다. 기존 NDCG@5의 개선을 top-1 선택의 확정적 개선으로 넓힐 수 없다.',
              '2. **개인화 우위도 충분히 확인되지 않았다.** 전체 모델과 광고 피처만 모델의 top-1 차이도 0을 포함한다. '
              '광고 자체의 특성을 이용하는 능력과 사용자별로 더 나은 광고를 고르는 능력을 구분한다.',
              '3. **확률 보정은 새 선택 성과를 보장하지 않았다.** 광고 보정 모델의 top-1 점추정은 두 주요 집단에서 전체 모델보다 낮았다. '
              '차이의 구간은 0을 포함하므로 열등하다고 확정하지도 않는다. 전체 모델의 순위를 그대로 가져오는 보정은 top-k 포착 결과도 그대로임을 검증했다.',
              '4. **현재 결론은 추가 검증이 필요하다는 것이다.** 이 분석에서 생산 CTR 개선이나 보정 모델 승격 근거를 얻었다고 판단하지 않는다. '
              '기존 서비스 모델과 배포 조건은 유지했다.', '',
              '## 후보 수에 따른 민감도: 보조 가설', '',
              '| 사전에 정한 집단 | 요청 | 전체 모델 − 무작위, 1위 차이 [95% 구간] |',
              '|---|---:|---:|']
    for name in ['all_multi_2_to_5', 'all_multi_6_to_10', 'all_multi_11_plus', 'randomized_observed_rank_contiguous']:
        c = m['cohorts'][name]
        d = c['comparisons']['full - uniform']['1']['extra_observed_clicked_ads_per_1000_requests']
        lines.append(f"| {COHORTS[name]} | {c['requests']:,} | {fmt_ci(d)} |")
    lines += ['', '후보 11개 이상에서 양의 구간을 관측했다. 다중 비교 보정이 없는 보조 집단이고 전체 로그의 위치 편향도 남아 있으므로, '
              '이를 주 분석의 실패를 대신할 성공 판정이나 바로 배포할 분기 규칙으로 사용하지 않는다. '
              '“후보가 많은 요청에서 이득이 더 큰가?”를 다음 독립 표본에서 검증할 가설로 남긴다.', '',
              '## 실제 tie 규칙과 학습 seed 점검', '',
              '세 seed 평균을 실제 단일 모델의 결과로 오해하지 않도록 seed 42와 광고 ID로 동점을 푸는 규칙도 표시한다.', '',
              '| 집단 | 정책 | seed 42 동점 평균 / 1,000요청 | seed 42 광고 ID 동점 처리 / 1,000요청 |',
              '|---|---|---:|---:|']
    for name in ['randomized_multi', 'all_multi']:
        c = m['cohorts'][name]
        for policy in m['config']['policies']:
            r = c['representative_seed_tie_sensitivity'][policy]['1']
            lines.append(f"| {COHORTS[name]} | {LABELS[policy]} | {r['tie_expected_per_1000']:.2f} | {r['product_id_ties_per_1000']:.2f} |")
    lines += ['', 'ID 동점 규칙은 서비스의 결정 규칙이며 학습된 광고 구별 능력이 아니다. 특히 컨텍스트 모델에서 이 규칙의 결과와 '
              '무작위 선택 기대값이 다르다. 전체 모델의 세 seed top-1 값은 두 주요 집단에서 모두 무작위 선택보다 높았지만, '
              '그 자체로 사용자 표본의 불확실성을 제거하지 않는다. seed별 정밀 수치는 JSON에 있다.', '',
              '## 왜 온라인 CTR이나 매출로 바로 환산하지 않는가?', '',
              '[FairJob §3.4](https://arxiv.org/html/2407.03059v2#S3.SS4)는 위치 영향 때문에 무작위 표시 집단에서 utility를 평가하는 접근을 설명한다. '
              '논문은 공개 데이터가 사업 지표를 숨기기 위해 비균일하게 샘플링됐다고도 명시한다. 따라서 무작위 표시 플래그만으로 '
              '원래 트래픽의 클릭률이나 모든 후보·위치에 대한 행동 확률을 복원하지 않았다.', '',
              '[Li et al., WSDM 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/published-3.pdf)의 '
              '오프라인 replay는 무작위 로깅과 일치하는 행동의 보상을 이용하는 평가다. 이번 데이터의 불완전한 후보 집합과 다중 광고 위치에 '
              '동일한 보장을 적용할 수 있다고 가정하지 않는다. `1 / 관측 후보 수`를 과거 노출 정책의 propensity로 만들어 쓰지 않았다. '
              '이번 uniform 가중치는 우리가 정의한 비교 정책의 동점 평균에만 사용한다.', '',
              '더 직접적인 사업 효과를 오프라인에서 추정하려면 요청 시점의 전체 적격 후보, 실제 선택·위치, 그 선택의 로깅 확률, '
              '정책·모델 버전, 클릭·전환 관측 기간과 샘플링 규칙이 필요하다. 금액 효과에는 입찰·실제 비용 또는 전환 가치도 필요하다. '
              '슬레이트 전체 효과와 위치별 효과에 따라 필요한 확률과 가정을 구분한 뒤 IPS/DR 또는 replay 가능성을 판단해야 한다.', '',
              '## 사실에 맞는 프로젝트 스토리', '',
              '> 한정된 광고 자리를 더 잘 배분하는지 확인하려고, 클릭이 없던 요청까지 포함해 요청 1,000건당 클릭 광고를 1위로 고른 횟수를 측정했습니다. '
              '무작위 표시 집단에서 전체 모델은 10.60건, 무작위 선택은 7.20건이었지만 차이의 구간은 0을 포함했습니다. '
              'NDCG 개선을 바로 사업 성과로 받아들이지 않고, 1위 선택에 대한 근거가 부족하다는 결과를 남겼습니다. '
              '후보가 많은 요청에서 나타난 이득은 다음 검증 가설로 분리했고, 확률 보정 모델을 자동 승격하지 않았습니다.', '',
              '## 재현과 검증', '',
              '```bash',
              'python3 scripts/run.py scripts/offline_business_validation.py \\',
              '  --output /tmp/fairjob-business-validation-replay',
              'python3 scripts/run.py scripts/report_offline_business_validation.py',
              'python3 scripts/run.py -m pytest tests/test_business_validation.py -q',
              '```', '',
              '첫 명령의 출력 경로는 아직 존재하지 않는 경로여야 한다. 보고서 생성기는 기본 보고서 경로의 동결된 결과를 읽는다. '
              '기존 서비스·학습 소스와 모델 파일은 바꾸지 않았다. 신규 테스트 4개는 동점의 전체 순열, 클릭 없는 요청의 분모, '
              '사용자 묶음 재표집, 모든 광고가 클릭된 요청을 검증한다. 기존 전체 pytest 스위트는 이번 작업에서 재실행하지 않았다.', '',
              '계산 코드: [business_validation.py](../../src/fairjob_ranker/diagnostics/business_validation.py). '
              '요청별 결과는 로컬 `request_outcomes.parquet`에 보존한다. [별도 수치 검산](verification.json).', '',
              '![관측 클릭 포착 차이와 불확실성](top1_capture.png)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    display = [('full', 'Full features', '#1565c0'), ('ad_only', 'Ad features only', '#777777'),
               ('ad_residual', 'Context + ad correction', '#008577')]
    for ax, cohort, title in zip(axes, ['randomized_multi', 'all_multi'], ['Randomized display (primary)', 'All observed displays (secondary)']):
        c = m['cohorts'][cohort]
        for y, (policy, label, color) in enumerate(display):
            # The ad-only contrast is descriptive and was not a primary hypothesis.
            if policy == 'ad_only':
                # Avoid implying a paired CI that was not preregistered/computed.
                estimate = c['policies'][policy]['1']['observed_clicked_ads_per_1000_requests']['estimate']-c['policies']['uniform']['1']['observed_clicked_ads_per_1000_requests']['estimate']
                ax.scatter(estimate, y, color=color, marker='x', s=55)
            else:
                value = c['comparisons'][policy+' - uniform']['1']['extra_observed_clicked_ads_per_1000_requests']
                estimate, (lo, hi) = value['estimate'], value['ci95']
                ax.errorbar(estimate, y, xerr=[[estimate-lo], [hi-estimate]], fmt='o', color=color, capsize=4, markersize=7)
                ax.text(hi+.12, y, f'{estimate:+.2f}', va='center', fontsize=9, color=color)
        ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
        ax.set_title(f'{title}\n{c["requests"]:,} requests, {c["observed_clicks"]} logged clicks', fontsize=11)
        ax.set_xlabel('Extra logged-click ads selected at #1\nper 1,000 requests vs uniform')
        ax.grid(axis='x', alpha=.18)
        ax.set_xlim(-2.5, 9)
        ax.set_yticks(range(3), [v[1] for v in display])
        ax.set_ylim(2.6, -.6)
    fig.suptitle('Better ad selection? Evidence remains uncertain', fontsize=15, fontweight='bold')
    fig.text(.5, .02, 'Fixed 3-seed mean; user-cluster 95% intervals. Gray x: point only. Not an online CTR or revenue estimate.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .92])
    fig.savefig(ROOT/'top1_capture.png', dpi=170)
    plt.close(fig)
    print(ROOT/'RESULTS.md')


if __name__ == '__main__':
    main()
