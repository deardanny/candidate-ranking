"""Summarize frozen-context correction experiments without tuning on outer data."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import numpy as np

from fairjob_ranker.diagnostics.context_ablation import expected_ndcg, logloss_rows
from fairjob_ranker.io import read_json, write_json


LABELS = {'context': '컨텍스트 기준', 'full': '기존 전체 피처',
          'ad_offset': '광고 보정값 더하기', 'ad_mean_preserved': '광고 보정 + 평균 확률 유지',
          'full_rank_graft': '기존 전체 순위 + 평균 확률 유지'}


def main(output):
    result = read_json(output/'metrics.json')
    config = result['config']
    names = ['context', 'full']+config['policies']
    summary = {}
    for split in ['inner', 'outer']:
        summary[split] = {}
        for cohort in ['all', 'randomized']:
            summary[split][cohort] = {}
            for name in names:
                metrics = {}
                for section, key in [('click', 'logloss'), ('click', 'roc_auc'), ('ranking', 'ndcg_tie_averaged'),
                                     ('ranking', 'pair_accuracy'), ('score_discrimination', 'all_scores_tied_fraction')]:
                    values = [result['metrics'][split][f'{name}-s{s}'][cohort][section][key] for s in config['seeds']]
                    metrics[key] = {'mean': float(np.mean(values)), 'seed_std': float(np.std(values, ddof=1)) if len(values)>1 else 0.}
                summary[split][cohort][name] = metrics
    point_checks = {}
    for name in config['policies']:
        candidate, base = summary['outer']['all'][name], summary['outer']['all']['context']
        checks = {'logloss_lower': candidate['logloss']['mean'] < base['logloss']['mean'],
                  'auc_higher': candidate['roc_auc']['mean'] > base['roc_auc']['mean'],
                  'ndcg_higher': candidate['ndcg_tie_averaged']['mean'] > base['ndcg_tie_averaged']['mean']}
        point_checks[name] = {**checks, 'all_three': all(checks.values()),
                              'scope': 'Point estimates only, no joint significance claim'}
    write_json(output/'summary.json', {'metrics': summary, 'point_checks_vs_context': point_checks})
    primary = config['primary_policy']
    pm, baseline = summary['outer']['all'][primary], summary['outer']['all']['context']
    pc = result['comparisons']['outer'][primary+' - context']['all']
    lines = ['# 컨텍스트 확률을 고정하고 광고 점수만 보정하면?', '',
             '사용자가 제안한 방식을 실제로 학습·평가했다. 기존 컨텍스트·전체 모델 3개씩은 고정했다. 학습용 기준 점수를 만드는 사용자 OOF 모델 9개와 광고 보정 모델 3개를 새로 학습했다. 기존 서비스 모델은 교체하지 않았다.', '',
             f"사전 주 분석인 **{LABELS[primary]}**의 outer LogLoss는 {pm['logloss']['mean']:.6f}(컨텍스트 기준 {baseline['logloss']['mean']:.6f}), NDCG는 {pm['ndcg_tie_averaged']['mean']:.6f}(기준 {baseline['ndcg_tie_averaged']['mean']:.6f})다. 컨텍스트 대비 차이의 95% 구간은 LogLoss [{pc['logloss']['ci95'][0]:+.6f}, {pc['logloss']['ci95'][1]:+.6f}], NDCG [{pc['ndcg_tie_averaged']['ci95'][0]:+.6f}, {pc['ndcg_tie_averaged']['ci95'][1]:+.6f}]다. 점추정과 구간을 함께 읽는다.", '',
             '## 세 가지 결합 방식', '',
             '컨텍스트 확률을 `p0`, 그 raw score를 `z0 = log(p0 / (1-p0))`, 광고 보정 모델 출력을 `r(ad)`라고 한다.', '',
             '1. **광고 보정값 더하기**: `p = sigmoid(z0 + α × r(ad))`. 컨텍스트 모델은 고정하고 보정 모델만 학습한다. 확률에 직접 더하지 않으므로 출력은 0~1 범위다.',
             '2. **광고 보정 + 평균 확률 유지**: 위 보정의 상대 순서는 유지하면서 노출별 공통 절편을 조정해 `mean(p_ads) = p0`가 되도록 한다. 실험의 사전 주 분석 방식이다.',
             '3. **기존 전체 순위 + 평균 확률 유지**: 새 광고 보정 모델 대신 기존 전체 모델의 raw score를 순위 신호로 사용한다. 기존 모델의 순서와 컨텍스트 확률을 결합하는 대조군이며, 새로운 랭킹 모델을 학습한 결과가 아니다.', '',
             '2·3번은 로그 점수의 평균만 빼는 방법이 아니다. sigmoid 이후의 확률 평균을 맞추도록 60회 이분 탐색한다. α가 양수이면 공통 절편·양의 배율은 노출 안의 donor 순서를 바꾸지 않는다. α=0이면 컨텍스트 기준으로 정확히 돌아가 모든 광고가 동점이 된다.', '',
             '평균 확률을 유지해도 그것이 실제 CTR이라는 보장이나 LogLoss 개선 보장은 없다. 2·3번은 후보 전체가 있어야 계산되며 top-k를 자르기 전에 적용해야 한다. 후보 집합이 달라지면 점수도 달라질 수 있다.', '',
             '## 학습과 선택의 통제', '',
             f"- 기존과 같은 inner train {read_json(output/'preregistered.json')['inner_train_rows']:,}행, inner validation {read_json(output/'preregistered.json')['inner_validation_rows']:,}행을 사용했다.",
             '- 사용자를 3개 fold로 나눴다. 각 사용자의 학습용 컨텍스트 점수는 그 사용자를 전혀 보지 않은 fold 모델로 만들었다. fold 전처리도 해당 학습 사용자만 사용했다.',
             '- fold 모델의 트리 수는 이전 컨텍스트 모델의 inner-validation 선택값으로 고정했다. 제외된 fold의 클릭 라벨은 fold 모델 적합·early stopping에 사용하지 않았다.',
             '- 보정 모델은 광고 피처 11개만 사용하고, OOF 컨텍스트 raw score를 LightGBM `init_score`로 넣어 binary LogLoss를 학습했다. 사용자·광고 ID, 기존 위치, 보호 속성은 입력에 넣지 않았다.',
             '- 보정 모델의 checkpoint는 실제 배포용으로 고정한 컨텍스트 모델과 합친 inner LogLoss로 선택했다. 모델 저장·재로딩 후에는 컨텍스트 raw score를 명시적으로 다시 더하며, 이 예측이 학습 중 validation 손실과 같은지 검증했다.',
             '- α 후보는 0/0.05/0.1/0.25/0.5/1.0이다. 방식마다 세 seed의 평균 inner LogLoss가 가장 낮은 α 하나를 선택했다. 세 seed별로 유리한 α를 고르지 않았다.',
             '- 모든 선택을 저장한 뒤 기존 outer validation을 읽었다. 원 test 파일은 읽지 않았으며 outer 결과를 보고 재학습하거나 강도를 바꾸지 않았다.', '',
             'OOF 모델은 inner train의 2/3로 학습하지만 평가 기준 모델은 전체 inner train으로 학습했다. 이 차이에 따른 기준 점수 분포 변화는 남아 있다. 광고 피처만 보는 보정 모델은 사용자별로 광고 A/B의 선호 순서를 바꾸는 상호작용을 학습하지 않는다.', '',
             '[LightGBM Dataset API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Dataset.html)의 `init_score`를 사용했다. 실제 저장·추론 동작은 프로젝트 테스트로 대조했다.', '',
             '## 내부 validation에서 고정한 보정 강도', '',
             '| 방식 | α | 선택된 평균 inner LogLoss |', '|---|---:|---:|']
    for name in config['policies']:
        item = result['selection'][name]
        lines.append(f"| {LABELS[name]} | {item['alpha']} | {item['mean_inner_logloss']:.8f} |")
    lines += ['', '전체 α 탐색값은 [inner_grid.json](inner_grid.json)에 보존했다. α=0이면 보정 포기를 선택한 것이며 개선으로 계산하지 않는다.']
    for split, title in [('inner', '내부 validation'), ('outer', '기존 outer validation')]:
        lines += ['', '## '+title, '', '세 seed의 평균이다. seed 표준편차는 `summary.json`에 있다.']
        for cohort, label in [('all', '전체 로그'), ('randomized', '무작위 표시 하위집단')]:
            r = result['metrics'][split][f'context-s{config["representative_seed"]}'][cohort]
            lines += ['', f"**{label}**: {r['click']['rows']:,}행, 랭킹 평가 {r['ranking']['eligible_impressions']}노출 / {r['ranking']['eligible_users']}명.", '',
                      '| 방식 | LogLoss ↓ | AUC ↑ | 동점 평균 NDCG@5 ↑ | 노출 내 pair accuracy ↑ |', '|---|---:|---:|---:|---:|']
            for name in names:
                s = summary[split][cohort][name]
                lines.append(f"| {LABELS[name]} | {s['logloss']['mean']:.6f} | {s['roc_auc']['mean']:.6f} | {s['ndcg_tie_averaged']['mean']:.6f} | {s['pair_accuracy']['mean']:.6f} |")
    lines += ['', '## 같이 좋아졌는가: outer 점추정', '',
              '기준은 컨텍스트 모델이다. 아래는 수치의 방향만 검사하며 세 지표의 동시 통계적 우위를 뜻하지 않는다. AUC의 bootstrap 구간은 이번 분석에 추가하지 않았다.', '',
              '| 방식 | LogLoss 감소 | AUC 증가 | NDCG 증가 | 세 조건 모두 |', '|---|---|---|---|---|']
    for name in config['policies']:
        c = point_checks[name]
        values = ['예' if c[k] else '아니오' for k in ['logloss_lower', 'auc_higher', 'ndcg_higher', 'all_three']]
        lines.append(f"| {LABELS[name]} | "+' | '.join(values)+' |')
    lines += ['', '## 사용자 단위 paired bootstrap', '',
              '같은 사용자의 모든 행·노출을 함께 복원 추출했다. 세 seed의 paired 차이를 먼저 평균하며 예측을 ensemble한 값은 아니다. 2,000회 bootstrap의 각 비교별 95% 구간으로, 다중 비교 보정이나 학습 데이터 재표본화 불확실성은 포함하지 않는다.', '',
              '| 후보 − 기준 | 집단 | LogLoss 차이 [95% CI] ↓ | NDCG 차이 [95% CI] ↑ | Pair 차이 [95% CI] ↑ |',
              '|---|---|---|---|---|']
    for name in config['policies']:
        for reference in ['context', 'full']:
            for cohort in ['all', 'randomized']:
                c = result['comparisons']['outer'][name+' - '+reference][cohort]
                values = []
                for metric in ['logloss', 'ndcg_tie_averaged', 'pair_accuracy']:
                    d = c[metric]
                    values.append(f"{d['mean_difference']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]" if d else 'N/A')
                lines.append(f"| {name} − {reference} | {cohort} | "+' | '.join(values)+' |')
    lines += ['', '## 평균 확률 보존 검사', '', '| 방식 | Outer 노출별 평균 확률 변화의 최댓값 (3 seed 중 최대) |', '|---|---:|']
    for name in config['policies']:
        maximum = max(result['mean_drift']['outer'][f'{name}-s{s}']['max_absolute_impression_mean_change'] for s in config['seeds'])
        lines.append(f'| {LABELS[name]} | {maximum:.3g} |')
    example = read_json(output/'example.json')
    lines += ['', f"## 앞서 논의한 실제 노출 {example['impression_id']}", '',
              '실험 전에 고정한 설명용 사례이며 대표 성능을 뜻하지 않는다. 아래는 seed 42의 클릭 확률형 출력이다.', '',
              '| 광고 | 관측 클릭 | 컨텍스트 | 기존 전체 | 광고 보정 | 광고 보정·평균 유지 | 전체 순위·평균 유지 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for row in example['rows']:
        lines.append(f"| {row['product_id']} | {row['click']} | "+' | '.join(f"{row[n]:.4%}" for n in names)+' |')
    y = np.array([r['click'] for r in example['rows']])
    if len(y):
        lines += ['', '| 이 노출만의 지표 | 평균 LogLoss ↓ | 동점 평균 NDCG@5 ↑ |', '|---|---:|---:|']
        for name in names:
            p = np.array([r[name] for r in example['rows']])
            lines.append(f'| {LABELS[name]} | {logloss_rows(y,p).mean():.6f} | {expected_ndcg(y,p):.6f} |')
    lines += ['', '## 재현과 해석 범위', '',
              '```bash', 'python3 scripts/run.py scripts/context_residual.py',
              'python3 scripts/run.py scripts/report_context_residual.py',
              'python3 scripts/run.py -m pytest tests/test_context_residual.py -q', '```', '',
              '학습 스크립트는 기존 출력 덮어쓰기를 거부한다. 다시 실행할 때는 새 `--models`와 `--output`을 지정하고 보고서에도 같은 `--output`을 전달한다.', '',
              '- [사전 설정](../../configs/context_residual.json), [실행 전 해시·분할](preregistered.json), [outer 확인 전 선택](selection.json)',
              '- [전체 지표·OOF 감사 기록](metrics.json), [평균·seed 분산·점추정 판정](summary.json), [예시](example.json)',
              '- [실험 코드](../../src/fairjob_ranker/diagnostics/context_residual.py), [테스트](../../tests/test_context_residual.py)',
              '- [저장 예측·평균 보존·순위·OOF·registry 검증](verification.json), [전체 테스트 XML](pytest.xml). 기본 경로의 재검증: `python3 scripts/run.py scripts/verify_context_residual.py`.',
              '- 행별 예측과 OOF 기록은 로컬 Parquet에 보존하고 Git에서 제외한다. MLflow experiment는 `fairjob-context-residual-v1`이다.',
              '- 이미 사용한 개발 데이터에 대한 후속 실험이다. 새로운 독립 test·온라인 CTR·수익 향상으로 해석하지 않는다. 새 독립 데이터에서의 검증 및 API 적용은 이번 실험에 포함하지 않았다.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    print('Saved', output/'RESULTS.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/context-residual'))
    main(parser.parse_args().output)
