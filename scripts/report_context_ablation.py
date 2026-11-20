"""Build tables and a chart from the completed diagnostic experiment."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fairjob_ranker.io import read_json, write_json


LABELS = {'context_only': '컨텍스트만', 'ad_only': '광고 피처만', 'full': '전체 피처'}


def main(output):
    results = read_json(output/'metrics.json')
    config = results['config']
    groups = {v: [r for r in results['runs'] if r['variant'] == v] for v in config['variants']}
    assert all(len(g) == len(config['seeds']) for g in groups.values())
    summary = {}
    for split in ['inner', 'outer']:
        summary[split] = {}
        for cohort in ['all', 'randomized']:
            summary[split][cohort] = {}
            for variant, runs in groups.items():
                values = {}
                for section, key in [('click', 'logloss'), ('click', 'roc_auc'), ('click', 'average_precision'),
                                     ('ranking', 'ndcg_tie_averaged'), ('ranking', 'pair_accuracy'),
                                     ('ranking', 'legacy_ndcg_id_ties'), ('ranking', 'top1_click_tie_averaged'),
                                     ('score_discrimination', 'all_scores_tied_fraction')]:
                    vs = [r[split][cohort][section][key] for r in runs]
                    values[key] = {'mean': float(np.mean(vs)), 'seed_std': float(np.std(vs, ddof=1)) if len(vs)>1 else 0.}
                summary[split][cohort][variant] = values
    write_json(output/'summary.json', summary)
    audit = results['feature_audit']
    lines = ['# 클릭 성향과 광고 구별 능력은 같은가?', '',
             '현재 활성 모델인 LightGBM에서 피처 집합만 바꿔 비교한 후속 개발 실험이다. 모델 3종 × seed 3개를 새로 학습했다. 기존 서비스 모델·레지스트리 alias는 교체하지 않았다.', '',
             '## 이번 실행의 핵심 수치', '',
             f"컨텍스트만 모델의 outer AUC/LogLoss는 {summary['outer']['all']['context_only']['roc_auc']['mean']:.6f}/{summary['outer']['all']['context_only']['logloss']['mean']:.6f}, 전체 모델은 {summary['outer']['all']['full']['roc_auc']['mean']:.6f}/{summary['outer']['all']['full']['logloss']['mean']:.6f}다. 컨텍스트만 모델의 노출 내 pair accuracy는 {summary['outer']['all']['context_only']['pair_accuracy']['mean']:.6f}, 전체 모델은 {summary['outer']['all']['full']['pair_accuracy']['mean']:.6f}다.", '',
             '전체 행의 클릭 예측과 같은 노출 안의 광고 선택을 별도로 판단해야 한다. 점수가 광고마다 다르다는 것, 클릭 광고를 상위에 놓는다는 것, 사용자 컨텍스트를 추가해 광고 순위가 개선된다는 것은 서로 다른 주장이다. 아래의 전체−컨텍스트 및 전체−광고 피처만 비교와 불확실성을 함께 읽는다.', '',
             '## 질문과 통제', '',
             '- 컨텍스트만: `cat0`, `cat1`, `num16`–`num50`의 37개 피처. 같은 노출 안에서는 값이 같다.',
             '- 광고 피처만: `cat2`–`cat12`의 11개 피처. 같은 광고 ID에서는 값이 같다. 사용자 컨텍스트 없이도 광고를 구별하는 비교군이다.',
             '- 전체: 두 집합을 합한 48개 피처. 광고 피처만 모델과 비교해야 컨텍스트를 추가한 순위 이득을 판단할 수 있다.',
             '- ID·기존 표시 위치·보호 속성·senior·무작위 표시 플래그·클릭 라벨은 입력에서 제외했다.',
             '- 피처 역할은 실제 값의 불변성으로 정한 구현상 구분이다. 공식 카드의 컬럼 설명과 실제 파일의 차이를 반영했다. 익명 피처의 의미나 개인의 성향을 복원한 것은 아니다.', '',
             '모든 조건에 V2와 같은 사용자 단위 내부 분할, LightGBM 설정, 최대 500라운드·early stopping 30라운드를 적용했다. 각 조건의 checkpoint는 내부 validation LogLoss로 선택했으며, seed 42/43/44를 모두 보고한다. 모델을 전부 고정한 뒤 기존 outer validation을 읽었다.', '',
             '같은 학습 절차를 적용했지만 early stopping 시점과 실제 트리 구조는 모델마다 다르다. 따라서 관측 차이를 특정 피처의 순수한 인과 효과로 해석하거나, 모든 하이퍼파라미터에서 같은 결론이 나온다고 주장하지 않는다.', '',
             f"- 내부 train: {audit['inner_train']['rows']:,}행 / {audit['inner_train']['users']:,}명",
             f"- 내부 validation: {audit['inner_validation']['rows']:,}행 / {audit['inner_validation']['users']:,}명",
             f"- 기존 outer validation: {audit['outer_validation']['rows']:,}행 / {audit['outer_validation']['users']:,}명",
             f"- 기존 V2 전체 피처 seed 42와 재학습 모델의 outer 예측 최대 차이: {results.get('existing_v2_replication', {}).get('max_abs_prediction_error', '별도 fixture')}.", '',
             '원 test 파일은 열지 않았다. 이 비교는 이미 사용한 개발 데이터에 대한 후속 분석이며 새 독립 test·온라인 CTR 향상·인과 효과가 아니다.', '',
             '## 동점에서 생기는 착시를 분리하는 평가', '',
             '컨텍스트만 쓰면 노출 안의 모든 광고가 동점이다. 기존 서비스 평가는 동점을 광고 ID 오름차순으로 풀었다. 그 결과의 NDCG를 모델이 학습한 광고 구별 능력으로 해석하면 안 된다.', '',
             '이번 주 지표는 동점 그룹 내부의 모든 순서에 대한 **기대 NDCG@5**다. 동점을 무시하거나 임의로 한 순서를 정하지 않는다. 별도로 클릭 광고와 비클릭 광고의 점수를 노출 안에서 비교하고, 이기면 1·동점이면 0.5·지면 0을 준 뒤 노출별 평균을 낸다. 따라서 모든 광고가 동점인 모델의 pair accuracy는 0.5다.', '',
             'NDCG와 pair accuracy는 후보 2개 이상이며 클릭·비클릭이 모두 있는 노출에 한정한다. 클릭 없는 노출도 LogLoss/AUC와 점수 변화율에는 포함한다. 이 조건부 지표를 전체 CTR로 표현하지 않는다.', '',
             '동점 기대값 구현은 작은 예제의 모든 순서 열거와 [scikit-learn NDCG 동점 처리](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) 양쪽에 대조했다. 기존 ID 동점 지표도 함께 남겼다.']
    for split, title in [('inner', '내부 validation'), ('outer', '기존 outer validation')]:
        lines += ['', '## '+title, '', '아래는 세 seed의 평균이다. `summary.json`에는 seed 표준편차도 있다.']
        for cohort, cohort_label in [('all', '전체 관측 로그'), ('randomized', '무작위 표시 하위집단')]:
            ref = groups['full'][0][split][cohort]
            lines += ['', f"**{cohort_label}**: 클릭 지표 {ref['click']['rows']:,}행, 순위 지표 {ref['ranking']['eligible_impressions']:,}노출 / {ref['ranking']['eligible_users']:,}명.", '',
                      '| 피처 | LogLoss ↓ | 전체 행 AUC ↑ | 동점 평균 NDCG@5 ↑ | 노출 내 pair accuracy ↑ | 모든 점수가 같은 노출 |',
                      '|---|---:|---:|---:|---:|---:|']
            for variant in config['variants']:
                s = summary[split][cohort][variant]
                lines.append(f"| {LABELS[variant]} | {s['logloss']['mean']:.6f} | {s['roc_auc']['mean']:.6f} | {s['ndcg_tie_averaged']['mean']:.6f} | {s['pair_accuracy']['mean']:.6f} | {s['all_scores_tied_fraction']['mean']:.2%} |")
            baseline = results['diagnostic_controls'][split]['constant'][cohort]
            lines.append(f"| 학습 CTR 상수 | {baseline['click']['logloss']:.6f} | {baseline['click']['roc_auc']:.6f} | {baseline['ranking']['ndcg_tie_averaged']:.6f} | {baseline['ranking']['pair_accuracy']:.6f} | 100% |")
            lines += ['', '마지막 열은 클릭 여부와 무관하게 후보가 2개 이상인 모든 노출을 사용한다. 무작위 표시 집단도 공개된 후보에 한정된다.']
    lines += ['', '## paired 사용자 bootstrap', '',
              '아래는 outer validation에서 세 seed의 paired 차이를 먼저 평균한 뒤, 사용자를 복원 추출한 95% 구간이다. 같은 사용자의 행·노출을 함께 유지하며 seed를 독립 표본으로 부풀리지 않는다. 이 구간은 고정된 학습 모델들에 조건부이며 학습 데이터 재표본화 불확실성은 포함하지 않는다. 여러 비교를 동시에 수행한 데 대한 보정은 하지 않았다.', '',
              '| 후보 − 비교군 | 집단 | LogLoss 차이 [95% CI] ↓ | NDCG 차이 [95% CI] ↑ | Pair 차이 [95% CI] ↑ |',
              '|---|---|---|---|---|']
    for comparison in ['full - context_only', 'ad_only - context_only', 'full - ad_only', 'full - full_group_mean', 'context_only - constant']:
        for cohort in ['all', 'randomized']:
            c = results['comparisons']['outer'][comparison]['seed_mean'][cohort]
            def interval(metric):
                v = c[metric]
                return f"{v['mean_difference']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" if v else 'N/A'
            lines.append(f"| {comparison} | {cohort} | {interval('logloss')} | {interval('ndcg_tie_averaged')} | {interval('pair_accuracy')} |")
    lines += ['', '`full_group_mean`은 전체 모델의 점수를 각 노출 안에서 평균해 모든 광고에 같은 값을 준 진단용 대조다. 노출의 평균 점수는 보존하지만 광고 순서 정보는 지운다. 별도로 학습한 컨텍스트 모델이나 배포 가능한 모델로 취급하지 않는다.', '',
              '## 기존 ID 동점 순서와 비교', '',
              '| Outer 전체 로그, seed 평균 | ID로 동점 해소한 NDCG | 동점 평균 NDCG |', '|---|---:|---:|']
    for variant in config['variants']:
        s = summary['outer']['all'][variant]
        lines.append(f"| {LABELS[variant]} | {s['legacy_ndcg_id_ties']['mean']:.6f} | {s['ndcg_tie_averaged']['mean']:.6f} |")
    lines += ['', '두 열의 차이는 평가 규칙의 차이다. 기존 서비스 지표가 계산 오류였다는 뜻이 아니다. 실제 ID 정렬 동작을 측정하는 목적과 모델이 배운 구별 능력을 측정하는 목적을 구분한다.', '',
              '## 피처와 점수의 충돌', '',
              f"Outer에서 한 사용자의 여러 노출 사이에 컨텍스트가 바뀌는 사용자는 {audit['outer_validation']['users_with_context_changing_across_impressions']:,}명이다. 따라서 컨텍스트를 고정된 개인 특성으로 단정하지 않는다.", '',
              '| 후보가 여러 개인 outer 노출 | 광고 피처 벡터가 모두 같은 비율 | 랭킹 평가 가능 노출에서 같은 비율 |', '|---|---:|---:|']
    for label, values in audit['outer_validation']['candidate_feature_collisions'].items():
        lines.append(f"| {label} | {values['multi_impressions_all_ad_vectors_identical_fraction']:.2%} | {values['eligible_impressions_all_ad_vectors_identical_fraction']:.2%} |")
    lines += ['', '동일 광고 피처 벡터에는 이 피처만 받는 결정적 모델이 서로 다른 점수를 줄 수 없다. 피처가 달라도 같은 트리 leaf 조합을 지나면 동점일 수 있다.', '',
              '## 실제 노출 예시', '', '다음은 설명을 위해 사전에 정한 규칙으로 선택한 최대 이득·최대 손실 예시다. 평균적인 사례나 개선 증거로 사용하지 않는다.']
    for example in read_json(output/'examples.json'):
        lines += ['', f"**{example['selection']} — 노출 {example['impression_id']}**, 무작위 표시={example['displayrandom']}, 동점 평균 NDCG {example['context_ndcg']:.4f} → {example['full_ndcg']:.4f}.", '',
                  '| 광고 ID | 관측 클릭 | 컨텍스트 점수 | 광고 피처만 점수 | 전체 점수 |', '|---:|---:|---:|---:|---:|']
        for row in example['rows']:
            lines.append(f"| {row['product_id']} | {row['click']} | {row['context_only_score']:.6f} | {row['ad_only_score']:.6f} | {row['full_score']:.6f} |")
    lines += ['', '## 재현', '',
              '```bash', 'python3 scripts/run.py scripts/context_ablation.py',
              'python3 scripts/run.py scripts/report_context_ablation.py',
              'python3 scripts/run.py -m pytest tests/test_context_ablation.py -q', '```', '',
              '기본 결과가 이미 있으면 덮어쓰기를 거부한다. 재실행은 새 `--models`, `--output` 경로를 지정하고 보고서에도 같은 `--output`을 전달한다.', '',
              '- [사전 실험 계약](../../configs/context_ablation.json), [실행 전 기록](preregistered.json), [모델 고정 기록](frozen_models.json)',
              '- [전체 수치·bootstrap](metrics.json), [seed 평균·분산](summary.json), [피처 점검](feature_audit.json)',
              '- 로컬 행별 예측·노출별 지표: `inner_predictions.parquet`, `outer_predictions.parquet`, `inner_ranking.parquet`, `outer_ranking.parquet` (Git 제외)',
              '- MLflow experiment: `fairjob-context-ablation-v1`, 모델 9개. 기존 champion alias를 바꾸지 않는다.',
              '- [저장 예측의 독립 NDCG 대조·artifact·registry 검증](verification.json), [전체 테스트 결과](pytest.xml). 이 검증은 기본 경로에서 `python3 scripts/run.py scripts/verify_context_ablation.py`로 다시 실행할 수 있다.',
              '- [실험 코드](../../src/fairjob_ranker/diagnostics/context_ablation.py), [검증 코드](../../tests/test_context_ablation.py)', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    variants = config['variants']
    display = ['Context only', 'Ad only', 'Full features']
    colors = ['#94a3b8', '#38bdf8', '#0f766e']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, metric, title in zip(axes, ['roc_auc', 'pair_accuracy'],
                                ['Click discrimination across all rows', 'Clicked vs nonclicked ad within an impression']):
        means = [summary['outer']['all'][v][metric]['mean'] for v in variants]
        sd = [summary['outer']['all'][v][metric]['seed_std'] for v in variants]
        if metric == 'pair_accuracy':
            # Context pair accuracy is identically 0.5 in every impression;
            # paired differences vs context therefore give absolute CIs here.
            bounds = [[.5, .5] if v == 'context_only' else
                      [x+.5 for x in results['comparisons']['outer'][v+' - context_only']['seed_mean']['all']['pair_accuracy']['ci95']]
                      for v in variants]
            errors = np.array([[m-b[0] for m, b in zip(means, bounds)],
                               [b[1]-m for m, b in zip(means, bounds)]])
            note = '95% whole-user bootstrap intervals'
        else:
            errors = np.array([sd, sd])
            note = 'Error bars: seed standard deviation'
        ax.bar(display, means, color=colors, width=.65, yerr=errors, capsize=4)
        ax.axhline(.5, color='#64748b', linestyle='--', linewidth=1)
        ax.set_ylim(0, max(.85, max(means)+.12))
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('ROC-AUC' if metric == 'roc_auc' else 'Pair accuracy (ties = 0.5)')
        for i, value in enumerate(means):
            ax.text(i, value+errors[1, i]+.025, f'{value:.3f}', ha='center', fontweight='bold')
        ax.set_xlabel(note, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Can the model distinguish ads, or only contexts?', fontsize=15, fontweight='bold')
    fig.supxlabel('Existing outer validation · averages across 3 seeds · logged clicks, not online lift', fontsize=9)
    fig.savefig(output/'context_vs_ads.png', dpi=180)
    plt.close(fig)
    print('Saved', output/'RESULTS.md', 'and context_vs_ads.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/context-ablation'))
    main(parser.parse_args().output)
