"""Render response-feature evidence and leakage/cold-start definitions."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fairjob_ranker.io import read_json


def ci(item):
    return f"{item['estimate']:+.3f} [{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"


def main():
    root = Path('reports/response-statistics')
    m, verified = read_json(root/'metrics.json'), read_json(root/'verification.json')
    pooled = m['pooled_oof']
    names = {'baseline': '기존 BCE', 'ad_stats': '광고 반응 통계 추가', 'ad_context_stats': '광고 통계 + 컨텍스트 조합 (주 모델)'}
    lines = ['# 광고 반응 통계 × 사용자 컨텍스트: 누수 방지와 새 광고 평가', '',
        '**결론: 같은 사용자 분할에서 기존 모델보다 좋아지지 않았다. 처음 보는 광고에서도 개선 근거를 확보하지 못했다.** '
        '광고 반응 통계만 추가한 모델과 컨텍스트 조합까지 추가한 모델을 seed 3개 × 사용자 분할 3개로 총 18개 학습했다. '
        '수치가 유리한 하위집단이나 다른 seed로 성공 기준을 바꾸지 않았고 기존 champion 13을 유지했다.', '',
        '## 가설과 실제 구현', '',
        '이전 LambdaRank 실험에서는 학습 목표를 바꿔도 다른 사용자에게 광고 선택 이득이 이어지지 않았다. '
        '이번에는 다른 학습 사용자의 반응을 광고와 익명 광고 속성의 통계로 전달하면 선택 능력이 나아지는지 검증했다.', '',
        '| 방식 | 입력 피처 | 모델 학습·선택 |', '|---|---:|---|',
        '| 기존 모델 | 기존 48개 | 이전에 고정한 BCE + 내부 LogLoss, 9개 재사용 |',
        '| 광고 통계 | 기존 48 + 통계 24 = 72개 | BCE + 내부 LogLoss |',
        '| 광고 통계 × 컨텍스트 | 기존 48 + 통계 76 = 124개 | BCE + 내부 LogLoss |', '',
        '광고 통계 24개는 `product_id`와 광고 속성 `cat2..cat12` 각각의 보정 반응률·`log1p(관측 행 수)`다. '
        '확장 모델은 여기에 컨텍스트 `cat0/cat1` 통계와 광고·광고 속성 × 컨텍스트 조합 통계를 더했다. '
        '기존 숫자 컨텍스트 35개도 그대로 들어가므로 트리에서 통계 피처와 상호작용할 수 있다. '
        '명시적 집계 조합은 cat0/cat1로 한정했다. 익명 피처에 직무·성별 등의 의미를 임의로 붙이지 않았다.', '',
        '다른 설정은 기존 모델과 동일하다: LightGBM BCE, 31 leaves, min_data_in_leaf 200, learning rate .05, '
        'L2 5, 최대 500회, 내부 LogLoss patience 30, seed 42/43/44. 새로운 hyperparameter 탐색은 하지 않았다. '
        '모든 18개 checkpoint와 통계 저장소를 고정한 뒤 외부 점수를 계산했다.', '',
        '### 보정 통계와 새 광고의 대체값', '',
        '- 전체 prior: `(학습 원천 클릭 합 + 0.5) / (학습 원천 행 수 + 1)`.',
        '- 광고 속성·컨텍스트별 반응률: `(클릭 합 + 100 × 전체 prior) / (행 수 + 100)`.',
        '- 광고별 반응률: `(광고 클릭 합 + 20 × 해당 광고의 11개 속성 반응률 평균) / (광고 행 수 + 20)`.',
        '- 광고 속성 × 컨텍스트: 부모 확률을 `sigmoid(logit(속성 확률)+logit(컨텍스트 확률)−logit(전체 prior))`로 만들고 강도 100으로 보정한다.',
        '- 광고 × 컨텍스트: 11개 속성 × 컨텍스트 확률의 평균을 부모로 사용하고 강도 20으로 보정한다.',
        '- 처음 보는 광고 ID는 행 수가 0이어서 속성 통계로 정확히 대체된다. 속성까지 처음이면 전체/컨텍스트 prior로 내려간다. 관측 건수 피처는 0이다.', '',
        '보정 강도 20/100은 실험 전에 고정한 값이다. 이 값이 최적이라는 주장은 아니다. '
        '반응률은 샘플링된 공개 로그의 비율이며 실제 서비스 모집단의 CTR·표시 확률로 해석하지 않는다.', '',
        '## 정답 누수 방지', '',
        '각 외부 사용자 분할에서 fit과 내부 validation을 먼저 분리했다. 그다음 **fit 사용자만 다시 5개 그룹으로 나눴다.** '
        '학습 행의 모든 통계는 해당 사용자가 속한 그룹 전체를 제외한 나머지 4개 그룹에서 계산했다. '
        '광고 통계뿐 아니라 전체 prior와 모든 부모 통계도 같은 원천에서 만든다.', '',
        '내부 validation과 외부 사용자에는 fit 전체로 만든 통계 저장소만 적용한다. 내부·외부 클릭 라벨은 집계에 들어가지 않는다. '
        '`transform()`은 라벨 없는 lookup 컬럼만 허용하고, click이 포함된 입력은 거부한다. '
        'product_id는 통계 조회 키로만 쓰며 원시 ID·user_id·과거 rank·displayrandom·보호 속성은 새 모델 피처에 넣지 않았다.', '',
        '학습용 target encoding을 cross-fitting으로 만드는 이유는 [scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html#sklearn.preprocessing.TargetEncoder.fit_transform)에도 설명돼 있다. '
        '이번 코드는 행 단위 기본 분할을 사용하지 않고 사용자 단위 분할과 계층적 대체값을 별도로 구현했다.', '',
        '검증한 사항:', '',
        '- 한 사용자의 클릭 라벨을 전부 뒤집어도 그 사용자의 학습 피처가 바뀌지 않는다. 같은 encoding 그룹의 다른 사용자 피처도 그대로다.',
        '- 3개 외부 분할 × 5개 encoding 분할의 사용자·요청 겹침이 0이고, 저장한 학습 통계 전체를 재계산했을 때 정확히 일치한다.',
        '- 114개 full-fit 집계 테이블의 건수·클릭 합을 원천 fit 데이터에서 별도로 검산했다.',
        '- 주 모델 fold0/seed42를 저장된 OOF 피처로 한 번 다시 학습해 내부 전체 예측이 원래 모델과 정확히 일치했다. 검증용 재실행이며 추가 모델 선택 실험이 아니다.',
        '- 저장 모델 18개를 재로딩한 모든 외부 행의 점수가 저장 예측과 일치한다. 새 광고 클릭 기여도도 별도 요청별 계산으로 검산했다.', '',
        '**시간 누수까지 검증한 것은 아니다.** 타임스탬프가 없어 “조회 시점보다 과거의 반응만 사용했다”는 조건은 확인할 수 없다. '
        '이번 결과는 사용자 분리 개발 실험이다. 실서비스용 과거 집계 검증과 구별한다.', '',
        '## 같은 사용자 분할에서의 결과', '',
        '원래 train 835,874행 / 24,174명에 이전과 동일한 3개 외부 사용자 분할을 적용했다. '
        '각 사용자는 자신이 학습·모델 선택에 들어가지 않은 모델로 한 번씩 평가된다. '
        '원래 validation/test는 읽지 않았다. seed별 요청 성과 평균이며 예측 앙상블은 아니다.', '',
        '아래는 **요청 1,000건당 관측 클릭 광고를 1위로 선택한 횟수**다. 클릭 없는 요청도 분모에 포함하고 동점은 평균한다.', '',
        '| 방식 | 전체 128,254요청 | 무작위 표시 12,519요청 | 무작위 표시에서 기존 대비 [95% 구간] |', '|---|---:|---:|---|']
    for name, label in names.items():
        delta = '기준' if name == 'baseline' else ci(pooled['randomized_multi']['comparisons'][name+'-baseline-top1'])
        lines.append(f"| {label} | {pooled['all_multi']['top1_per1000'][name]:.3f} | {pooled['randomized_multi']['top1_per1000'][name]:.3f} | {delta} |")
    lines += ['', '| 외부 사용자 분할 | 광고 통계 − 기존: 무작위 표시 | 컨텍스트 조합 − 기존: 무작위 표시 |', '|---|---|---|']
    for fold in range(3):
        comparisons = m['folds'][str(fold)]['randomized_multi']['comparisons']
        lines.append(f"| {fold} | {ci(comparisons['ad_stats-baseline-top1'])} | {ci(comparisons['ad_context_stats-baseline-top1'])} |")
    lines += ['', '주 모델의 차이는 **−0.708, −1.655, 0.000건**으로 모든 분할에서 개선되지 않았다. '
        '합친 무작위 표시 차이는 −0.785건이며 구간 상한은 수치 오차 수준의 0이다. '
        '이를 엄격히 음수인 구간으로 과장하지 않는다. 전체 표시에서도 '+ci(pooled['all_multi']['comparisons']['ad_context_stats-baseline-top1'])+'건이었다.', '',
        '**주 가설 실패.** 사전 기준은 주 모델의 무작위 표시 차이가 세 fold 모두 양수, 합친 95% 하한 >0, 전체 표시에서도 하한 >0이다. '
        '주 모델을 광고 통계만 쓰는 보조군으로 사후 교체하지 않았다.', '',
        '## 처음 보는 광고에서도 효과가 있었는가', '',
        '두 정의를 구분했다. `fit_unseen`은 모델 학습·통계 원천에 없던 광고다. 주 cold 정의인 `strict_unseen`은 **fit뿐 아니라 내부 모델 선택 데이터에도 없던 광고**다. '
        '광고 ID의 포함 여부로만 판정하며 평가 정답을 사용하지 않는다. 제품 ID의 신규성이지 시간상 신규 출시를 뜻하지 않는다.', '',
        '엄격한 새 광고는 외부 데이터 54,853행에 나타났고, 단일 후보 요청까지 합친 관측 클릭은 408개다. '
        '순위 평가용 다중 후보 요청에서는 새 광고 클릭이 234개이며, 무작위 표시에서는 16개다.', '',
        '**혼합 후보를 그대로 유지했다.** 새 광고가 포함된 요청에서 기존 광고를 없애지 않고 전체 후보를 정렬한 뒤, '
        '1위로 고른 광고가 실제 클릭된 새 광고인 경우에만 cold 성과로 계산했다. 기존 광고 클릭을 새 광고 개선으로 세지 않았다.', '',
        '| 평가 집단 | 요청 수 | 새 광고 클릭 수 | 기존의 새 광고 클릭 포착 / 1,000 | 주 모델 | 차이 [95% 구간] |', '|---|---:|---:|---:|---:|---|']
    for label, key in [('새 광고 포함 전체', 'strict_has_cold'), ('새 광고 포함 무작위 표시', 'randomized_strict_has_cold'),
                       ('후보 전부 새 광고', 'strict_all_cold')]:
        c = pooled[key]
        lines.append(f"| {label} | {c['requests']:,} | {c['cold_clicks']} | {c['cold_top1_per1000']['baseline']:.3f} | "
                     f"{c['cold_top1_per1000']['ad_context_stats']:.3f} | {ci(c['comparisons']['ad_context_stats-baseline-cold_top1'])} |")
    lines += ['', '새 광고 포함 전체에서 주 모델의 cold 포착 차이는 fold별 '+', '.join(
        f"{m['folds'][str(f)]['strict_has_cold']['comparisons']['ad_context_stats-baseline-cold_top1']['estimate']:+.3f}" for f in range(3))+
        '건이었다. 전체 1위 중 새 광고를 선택한 비율은 25.79%에서 26.14%로 조금 늘었으나, 클릭된 새 광고를 고른 횟수는 늘지 않았다.', '',
        '**새 광고 개선도 확인하지 못했다.** 특히 후보 전부가 새 광고인 무작위 표시 집단은 52요청 / 48명, 관측 클릭 0개였다. '
        '계산상 모든 모델의 포착이 0이고 bootstrap 구간도 [0,0]이지만, 이는 동등성이나 무효과를 정밀하게 증명한 결과가 아니다. '
        '비교할 클릭 신호가 없는 표본이다. 전체 all-cold 집단도 533요청 / 클릭 9개라 불확실성이 크다.', '',
        '### 확률 예측 지표와 광고 선택은 따로 읽는다', '',
        '| 평가 행 | 방식 | LogLoss ↓ | AUC ↑ |', '|---|---|---:|---:|']
    for group_key, group_label in [('all_rows', '전체'), ('strict_unseen_rows', '엄격한 새 광고')]:
        for name, label in names.items():
            p = m['probability_metrics'][group_key]['models'][name]
            lines.append(f"| {group_label} | {label} | {p['logloss']:.6f} | {p['auc']:.6f} |")
    lines += ['', '전체 AUC와 엄격한 새 광고 LogLoss의 작은 점추정 개선은 있지만, 이 표에는 불확실성 구간을 계산하지 않았다. '
        '이를 유의한 개선으로 부르거나 요청 내 광고 선택 개선으로 대신하지 않는다. '
        '특히 새 광고 행의 확률 예측과 기존 광고와 경쟁할 때 선택되는 결과는 서로 다른 평가다.', '',
        '## 판단과 실제 스토리', '',
        '> 순위 손실만 바꿔서는 다른 사용자에게 성능이 이어지지 않아, 광고 반응을 통계 피처로 전달해 봤습니다. '
        '사용자별 cross-fitting으로 자기 정답이 통계에 들어가는 것을 막았고, 처음 보는 광고에는 속성별 통계를 대체값으로 사용했습니다. '
        '18개 모델을 같은 사용자 분할에서 비교하자 AUC의 점추정은 좋아졌지만 실제 요청의 1위 선택 성과는 좋아지지 않았습니다. '
        '새 광고 성과도 기존 광고의 클릭과 분리해서 검증했고, 후보 전부가 새 광고인 무작위 표시 집단에는 클릭이 없어 우열을 판단할 수 없다는 한계까지 남겼습니다.', '',
        '이번 설정에서 **통계 피처 추가가 광고 선택을 개선한다는 가설을 채택하지 않았다.** '
        '보정 강도·상호작용 표현·과거 표시 정책에 따른 관측 편향·OOF와 full-fit 집계량 차이 등은 여전히 가능한 제약이다. '
        '실험하지 않은 원인을 실패의 확정 원인으로 쓰지 않는다. 이미 본 개발 평가에서 성공할 때까지 설정을 바꾸는 대신, '
        '다음 확증에는 시점이 있는 반응 기록과 비교 가능한 새 광고 클릭 신호가 필요하다. 기존 champion 13은 그대로다.', '',
        '## 검증과 재현', '',
        f"- 검증 상태 `{verified['status']}`. 사용자 24,174명·835,874행 외부 평가 1회, 기존 모델/소스 해시 보존.",
        '- 관련 테스트 14개 통과(신규 5개). 사용자 전체 라벨 변조, smoothing 수치, 새 ID/새 속성 대체값, 저장·로드, cold 클릭 기여 분리를 검사했다. 전체 서비스 테스트 스위트는 재실행하지 않았다.',
        '- OOF 통계 전체 재계산, 통계 테이블 114개 검산, 모델 18개 전체 외부 예측 재로딩, 주 모델 1개 정확한 재학습 검증 완료.',
        '- MLflow 완료 run 18개. 모델과 통계 저장소·feature 순서를 보존했다. 이 실험의 새 피처는 기존 API에 자동 배포하지 않았다.',
        '- 사용자 단위 paired bootstrap 4,000회는 고정 모델에 조건부다. fold 학습 집합이 겹치므로 독립 실험 3개로 해석하지 않는다. 보조 집단 구간에는 다중 비교 보정을 적용하지 않았다.',
        '- 알려진 개발 데이터의 재분할이며 새로운 독립 test가 아니다. 온라인 CTR·추가 클릭·매출을 추정하지 않는다.', '',
        '```bash', 'python3 scripts/run.py scripts/response_statistics.py',
        '# 기존 실험 디렉터리를 덮어쓰지 않는다. 재학습은 새 --output/--models 경로로 실행한다.',
        'python3 scripts/run.py scripts/verify_response_statistics.py',
        'MPLCONFIGDIR=.tools/matplotlib python3 scripts/run.py -m pytest tests/test_response_statistics.py tests/test_ranking_objective_reproduction.py tests/test_candidate_count_reproduction.py tests/test_business_validation.py -q',
        'python3 scripts/run.py scripts/report_response_statistics.py', '```', '',
        '근거: [학습 전 프로토콜](protocol.json), [전체 지표](metrics.json), [누수·모델 검증](verification.json), '
        '[설정](../../configs/response_statistics.json), [통계 피처 구현](../../src/fairjob_ranker/diagnostics/response_statistics.py), '
        '[실험 구현](../../src/fairjob_ranker/diagnostics/response_statistics_experiment.py).', '',
        '![동일 사용자 분할과 새 광고 성과](response_statistics.png)', '']
    (root/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    panels = [
        (axes[0], ['User fold0', 'User fold1', 'User fold2', 'Pooled OOF'],
         [m['folds'][str(f)]['randomized_multi']['comparisons']['ad_context_stats-baseline-top1'] for f in range(3)]+
         [pooled['randomized_multi']['comparisons']['ad_context_stats-baseline-top1']],
         'Overall ad choice: randomized display', 'Captured clicks / 1,000 requests'),
        (axes[1], ['Contains unseen ads\n(all display)', 'Contains unseen ads\n(randomized display)', 'All candidates unseen\n(all display)'],
         [pooled[key]['comparisons']['ad_context_stats-baseline-cold_top1'] for key in ['strict_has_cold', 'randomized_strict_has_cold', 'strict_all_cold']],
         'Strictly unseen ad clicks: full candidate competition', 'Captured unseen-ad clicks / 1,000 eligible requests')]
    for ax, labels, items, title, xlabel in panels:
        for i, item in enumerate(items):
            val, bounds = item['estimate'], item['ci95']
            ax.errorbar(val, i, xerr=[[val-bounds[0]], [bounds[1]-val]], fmt='o', color='#44657a', capsize=4, lw=2)
        ax.axvline(0, linestyle='--', color='#87919b', lw=1)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontsize=10, pad=15)
        ax.set_xlabel('Change vs baseline\n'+xlabel, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Response statistics + context did not improve ad selection', fontsize=15, y=1.04)
    fig.text(.5, -.07, 'User-disjoint encoding; 3 user folds × 3 seeds. Paired user bootstrap 95% intervals.\n'
             'Randomized all-unseen cohort: 52 requests, zero observed clicks — insufficient ranking evidence.\n'
             'Historical observed-click proxy on reused development data; not online CTR lift.', ha='center', fontsize=9, color='#505b66')
    fig.tight_layout(w_pad=3)
    fig.savefig(root/'response_statistics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Wrote response-statistics report and figure.')


if __name__ == '__main__':
    main()
