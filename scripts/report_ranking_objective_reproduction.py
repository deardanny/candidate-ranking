"""Build the reproducible report and figure from saved metrics, without training."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairjob_ranker.io import read_json


def uncertainty(item):
    return f"{item['estimate']:+.3f} [{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"


def main():
    root = Path('reports/ranking-objective-reproduction')
    m, d, v = [read_json(root/name) for name in ['metrics.json', 'diagnostics.json', 'verification.json']]
    names = {'baseline': '기존 BCE + LogLoss 선택', 'bce_top1': 'BCE + Top1 선택',
             'rank_top1': 'LambdaRank + Top1 선택 (주 가설)', 'rank_user_top1': '사용자 균형 LambdaRank + Top1 선택'}
    text = ['# 광고 순서를 직접 학습하면 다른 사용자에게도 개선되는가', '',
        '**결과: 사전에 정한 개선 기준을 충족하지 못했다.** 새 모델 27개를 학습하고 기존 기준 모델 9개와 같은 사용자 분할에서 비교했다. '
        '주 가설인 LambdaRank는 무작위 표시 집단의 세 분할 모두 기존 모델보다 낮았다. '
        '모델 선택 기준만 바꾸거나 사용자별 학습 비중을 조절해도 재현되는 개선을 확보하지 못했다.', '',
        '## 실패에서 세운 가설', '',
        '이전 후보 수 분석에서 n≥11의 이득이 다른 사용자 분할에서 재현되지 않았다. 이번에는 후보 수로 적용 대상을 고르는 대신, '
        '전체 행의 클릭 확률을 맞추는 학습·선택 기준이 요청 안의 광고 선택에 맞지 않는지 실험했다. '
        '앞서 신경망 BCE+pairwise를 실험한 이력이 있다. 이번은 그 결과를 숨긴 새 아이디어가 아니라, 트리 기반 순위 학습과 Top1 선택 기준을 같은 사용자 분할에서 검증한 후속 실험이다.', '',
        '| 실험군 | 바꾼 부분 |', '|---|---|',
        '| 기존 기준 모델 | 앞선 실험에서 저장한 BCE + 내부 LogLoss 조기 종료 |',
        '| BCE + Top1 | 학습 경로는 유지하고, 내부 Top1 포착으로 checkpoint 선택 |',
        '| LambdaRank + Top1 | 같은 요청의 서로 다른 클릭 라벨 광고를 비교하는 순위 학습 |',
        '| 사용자 균형 LambdaRank + Top1 | 순위 학습에 유효한 요청이 많은 학습 사용자의 요청 가중치를 낮춤 |', '',
        '주 가설은 **일반 LambdaRank 대비 기존 BCE**다. 다른 실험군의 결과가 더 좋아도 사후에 주 가설로 바꾸지 않는다.', '',
        '## 실험 계약', '',
        '- 원래 train 835,874행 / 24,174명을 이전과 정확히 같은 외부 3개 사용자 fold와 내부 fit/validation으로 분할했다. 각 사용자는 외부 평가에 한 번만 등장한다.',
        '- 기존 fold별 학습 전용 변환기와 동일한 48개 피처, 트리 용량·정규화·학습률·sampling 설정을 사용했다. 최대 500회, patience 30, seed 42/43/44. 추가 hyperparameter 탐색은 없다.',
        '- 기존 BCE 9개는 재학습하지 않았다. 신규 27개 checkpoint를 모두 고정한 뒤 외부 평가를 시작했다. BCE 선택 대조군은 기존 모델과 공통 tree prefix의 예측이 정확히 같은지도 검사했다.',
        '- 모든 학습 행을 유지했다. singleton 및 라벨이 모두 같은 요청에서는 LambdaRank의 클릭/비클릭 쌍 학습 신호가 없다. 그룹을 impression 단위로 연속 배치하고, 초기 동점 순서에 과거 rank가 들어가지 않게 label-blind shuffle했다.',
        '- 내부 모델 선택은 클릭 없는 요청까지 포함한 전체 다중 후보 요청의 동점 평균 Top1 포착이다. 동점의 한 후보만 임의로 정답처럼 세지 않았다. 내부 무작위 표시 요청만으로 선택하지는 않았다.',
        '- 사용자 균형군은 fit 안의 유효 요청 수를 m_u라 할 때 요청별 1/m_u를 평균 1로 정규화한다. 그 요청의 모든 행에 같은 값을 적용한다. 사용자별 **요청 가중치의 합**이 같으며 실제 gradient 크기까지 같다는 뜻은 아니다.',
        '- 주 평가: 무작위 표시·후보 2개 이상에서 기존 BCE 대비 Top1 포착 차이. 세 fold의 점추정이 모두 양수이고 합친 사용자 bootstrap 95% 하한이 0보다 커야 한다. 전체 표시 집단에서도 하한 >0을 요구했다.',
        '- seed별 요청 성과를 평균했다. 예측 점수를 평균한 ensemble이 아니다. 사용자 단위 paired bootstrap 4,000회는 고정된 학습 모델에 조건부이며, fold별 학습 집합은 서로 겹친다.',
        '- 이미 관찰한 개발 데이터의 사용자 재분할이다. 새 독립 test로 부르지 않는다. 이번 실행에서 원래 validation/test 파일을 읽지 않았다.', '',
        '## 관측 클릭 광고를 1위로 선택한 횟수 / 1,000요청', '',
        '클릭 없는 요청도 분모에 포함한다. 광고 노출을 바꾸었을 때 실제로 추가될 클릭이나 온라인 CTR을 추정한 수치가 아니다.', '',
        '| 방식 | 전체 요청 | 무작위 표시 요청 | 무작위 표시에서 기존 대비 차이 [95% 구간] |',
        '|---|---:|---:|---|']
    a, r = m['pooled_oof']['all_multi'], m['pooled_oof']['randomized_multi']
    for key, label in names.items():
        diff = '기준' if key == 'baseline' else uncertainty(r['comparisons'][key+'-baseline-top1'])
        text.append(f"| {label} | {a['top1_per1000'][key]:.3f} | {r['top1_per1000'][key]:.3f} | {diff} |")
    text += [f"| 균등 무작위 후보 선택 | {a['top1_per1000']['uniform']:.3f} | {r['top1_per1000']['uniform']:.3f} | 참고 기준 |", '',
        f"전체 다중 후보 {a['requests']:,}요청 / {a['users']:,}명, 무작위 표시 {r['requests']:,}요청 / {r['users']:,}명이다. "
        f"무작위 표시 집단의 클릭 있는 요청은 {r['positive_requests']:,}개, 클릭 없는 요청은 {r['zero_click_requests']:,}개다.", '',
        '| 사용자 외부 분할 | BCE + Top1 − 기존 | LambdaRank − 기존 | 사용자 균형 LambdaRank − 기존 |',
        '|---|---|---|---|']
    for fold in range(3):
        values = m['folds'][str(fold)]['randomized_multi']['comparisons']
        text.append(f"| {fold} | "+' | '.join(uncertainty(values[n+'-baseline-top1']) for n in m['config']['variants'])+' |')
    text += ['', '단위는 무작위 표시 1,000요청당 차이다. 분할별 구간은 기술적 참고이며 각 fold를 독립 실험으로 간주하지 않는다.', '',
        '**주 가설 판정: 실패.** 무작위 표시에서 세 fold가 모두 음수였고 합친 차이의 구간도 음수였다. '
        '전체 표시에서 주 모델의 차이는 '+uncertainty(a['comparisons']['rank_top1-baseline-top1'])+'건이다. '
        '단순 선택 기준 변경의 구간은 0을 포함했다. 사용자 균형군도 무작위 표시에서 개선되지 않았다.', '',
        '전체 집단에서 LambdaRank는 균등 선택보다 +0.299건 [0.038, 0.569] 높았지만, 이 사실은 기존 모델보다 좋아졌다는 증거가 아니다. '
        '사전에 정한 비교 대상인 기존 BCE를 이겨야 한다.', '',
        '### 보조 분석', '',
        '- 주 모델의 무작위 표시 결과는 seed 42/43/44에서 모두 음수였다: '+', '.join(f"{s}: {x:+.3f}" for s,x in r['per_seed_delta_per1000']['rank_top1'].items())+'건 / 1,000요청.',
        '- n≥11에서 사용자 균형 모델의 전체 집단 점추정만 +0.175건이었지만 구간 [−0.783, +1.131]은 0을 포함했다. 이 하위집단을 새 성공 기준으로 삼지 않았다.',
        '- 사용자별 요청 성과를 먼저 평균한 동일 사용자 비중 평가에서도 신규 세 방식의 전체·무작위 표시 개선 구간은 모두 0을 포함했다.',
        '- Top3, 모든 seed, fold별 전체 집단 결과와 보조 비교는 metrics.json에 함께 남겼다. 보조 구간에는 다중 비교 보정을 적용하지 않았다.', '',
        '## 결과를 본 뒤 실시한 원인 진단 — 원인 확정 실험은 아님', '',
        '새 모델을 더 학습하거나 바깥 평가에 맞춰 설정을 바꾸지 않았다. 아래는 고정 모델의 fit/inner/held 성과와 학습 가능한 표본을 읽은 사후 진단이다.', '',
        '| fold | fit 전체 요청 | 클릭·비클릭이 함께 있는 요청 | 해당 사용자 | 그중 무작위 표시 요청 |', '|---|---:|---:|---:|---:|']
    diag = pd.DataFrame(d['fixed_model_partition_statistics'])
    for audit in m['split_audits']:
        fold = audit['fold']
        q = diag.loc[(diag.fold == fold)&(diag.partition == 'fit')].iloc[0]
        text.append(f"| {fold} | {audit['all_queries']:,} | {audit['informative_queries']:,} | {audit['informative_users']:,} | {int(q.informative_randomized_requests):,} |")
    text += ['', '약 44만 행이 있어도 요청 내 순서를 직접 가르칠 수 있는 학습 요청은 약 2천 개, 사용자 698–735명이다. '
        '유효 요청은 fit 전체의 약 2.0–2.2%이며, 그중 무작위 표시 요청은 186–234개다. '
        '이것은 정보량의 제약을 보여주지만 표본 부족이 실패의 유일한 원인임을 증명하지는 않는다.', '',
        '| 방식 | 내부 선택 집단 Top1 / 1,000 | 외부 사용자 Top1 / 1,000 | 외부 유효 요청의 최고 점수 동점 비율 |', '|---|---:|---:|---:|']
    means = diag.groupby(['partition', 'variant']).mean(numeric_only=True)
    for key, label in names.items():
        inner, held = means.loc[('inner', key)], means.loc[('held', key)]
        text.append(f"| {label} | {inner.multi_top1_per1000:.3f} | {held.multi_top1_per1000:.3f} | {held.informative_top_score_tied_fraction:.2%} |")
    text += ['', '위 진단 표는 3개 fold × 3개 seed의 비가중 평균이다. 앞의 OOF 합산 지표와 집계 방식이 다르다. '
        '내부 Top1은 모델 선택에 사용했으므로 새 방식에 유리한 선택 편향이 있다. 내부 개선이 외부 사용자 개선으로 이어지지 않았다는 사실만 읽는다.', '',
        '일반 LambdaRank의 선택된 트리는 1–11개였고, 기존 BCE는 38–49개였다. '
        '일반 LambdaRank는 외부 유효 요청 약 98.5%에서 후보 점수를 구별하지만, 최고 점수 동점 비율은 약 11.2%로 기존 BCE 0.5%보다 높았다. '
        '따라서 동점 처리의 공정성은 중요하며, 목표만 바꿨다고 광고 구별력이 충분해졌다고 할 수 없다. '
        '고정된 용량·정규화와 희소한 Top1 선택 지표도 가능한 제약이다. 이번 결과를 모든 LambdaRank 설정의 실패로 일반화하지 않는다.', '',
        '광고 피처가 차지하는 split gain 비율은 일반 LambdaRank 평균 약 95.6%, 기존 BCE 약 49.7%였다. '
        '이는 학습이 광고 피처 쪽으로 이동했다는 기술적 관측이다. gain은 인과적 기여도나 개인화 품질의 증거가 아니다.', '',
        '무작위 표시와 비무작위 표시 간 개선 효과 차이는 '+uncertainty(d['random_minus_nonrandom_effect_interaction_per1000'])+'건 / 1,000요청이었다. '
        '표시 집단에 따른 차이가 남아 있지만, 과거 위치 편향이 실패를 일으켰다고 단정할 수는 없다. '
        '무작위 표시에서 가장 불리한 사용자 5명을 사후 제외해도 점추정은 −0.669건이었다. 이는 개선 평가를 대체하는 필터가 아니라 영향도 진단이다.', '',
        '## 이번 결정과 다음 가설', '',
        '**기존 champion 13을 유지한다.** 이번 실험으로 더 좋은 광고 선택 모델을 만들었다고 주장하지 않는다. '
        '“후보 수가 많을 때만 적용”과 “랭킹 손실이면 해결”이라는 두 단순한 결론 모두 근거가 부족했다.', '',
        '다음 후보는 **이전 학습 사용자에게서 얻은 광고 반응 신호를 새 사용자에게 전달하는 표현**이다. '
        '구체적으로 학습 사용자 내부에서 OOF로 만든 smoothing 광고·광고 속성 반응 통계와 컨텍스트 상호작용을 추가하고, '
        '기존 피처 모델과 비교한다. 이미 본 광고와 처음 보는 광고를 구분해 검증해야 하며, 자기 라벨·외부 사용자 라벨을 통계에 섞지 않아야 한다. '
        '이 표현 실험은 아직 실행하지 않았고, 이번 결과가 성공을 보장하지 않는다. '
        '외부 로그를 얻는다면 후보 목록·표시 확률·시각·광고/사용자 맥락이 함께 있는 자료가 다음 검증 범위를 넓힌다.', '',
        '## 프로젝트 스토리', '',
        '> 광고 클릭 확률을 잘 맞추는 모델이 실제 광고를 잘 고르는지 확인하던 중, 후보가 많을 때 보이던 이득이 다른 사용자에게 재현되지 않았습니다. '
        '그래서 적용 대상을 고르는 규칙 대신 학습 문제를 바꿨습니다. 같은 요청 안의 광고를 직접 비교하는 LambdaRank, 1위 선택 기준, 사용자별 가중치를 대조해 27개 모델을 학습했습니다. '
        '내부 검증에서는 성능이 좋아졌지만, 무작위 표시의 다른 사용자에서는 세 분할 모두 기존 모델보다 낮았습니다. '
        '약 44만 학습 행 중 실제 클릭·비클릭 비교가 가능한 요청은 약 2천 개라는 점도 확인했습니다. '
        '이 결과를 숨기고 새 모델을 배포하지 않고, 다음 검증 대상을 광고 반응 표현과 새 사용자 전이로 좁혔습니다.', '',
        '## 검증·재현', '',
        f"- 검증 상태: {v['status']}. 신규 27개 모델을 다시 불러 모든 held 행의 저장 점수와 일치시켰다. 별도 요청별 평가기로 동점 평균 Top1/Top3를 검산했다.",
        '- fit 전용 변환기 3개를 독립 재적합하고 24,174명 / 835,874행의 한 번씩 외부 평가를 확인했다. 이전 모델·데이터·source 해시를 보존했다.',
        '- 27개 checkpoint가 내부 지표의 최초 최대값과 일치했다. BCE 선택 대조군 9개의 공통 학습 prefix도 기존 모델과 정확히 일치했다.',
        '- MLflow 신규 완료 run 27개, 기존 champion 13 유지. 관련 테스트 9개 통과(신규 3개 포함). 전체 서비스 테스트 스위트는 재실행하지 않았다.',
        '- 순위 모델 출력은 raw ranking score다. CTR 확률이 아니므로 sigmoid를 씌워 LogLoss 개선을 주장하거나 기존 확률 API에 자동 연결하지 않았다.', '',
        '```bash', 'python3 scripts/run.py -m pytest tests/test_ranking_objective_reproduction.py tests/test_candidate_count_reproduction.py tests/test_business_validation.py -q',
        '# 기존 결과 디렉터리가 있으면 덮어쓰지 않는다. 재학습하려면 별도 --output/--models 경로를 지정한다.',
        'python3 scripts/run.py scripts/ranking_objective_reproduction.py',
        'python3 scripts/run.py scripts/verify_ranking_objective_reproduction.py',
        'MPLCONFIGDIR=.tools/matplotlib python3 scripts/run.py scripts/diagnose_ranking_objective_reproduction.py',
        'python3 scripts/run.py scripts/report_ranking_objective_reproduction.py', '```', '',
        '핵심 파일: [실험 설정](../../configs/ranking_objective_reproduction.json), [학습 전 protocol](protocol.json), '
        '[전체 지표](metrics.json), [검증](verification.json), [사후 진단 계약](diagnostic_protocol.json), [사후 진단](diagnostics.json).', '',
        '구현 근거: [LightGBM 4.6 LambdaRank 설정](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#lambdarank_truncation_level), '
        '[query group/instance weight API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Dataset.html), '
        '[custom evaluation API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html), '
        '[query gradient에 instance weight가 적용되는 공식 소스](https://github.com/microsoft/LightGBM/blob/v4.6.0/src/objective/rank_objective.hpp).', '',
        '![외부 사용자에서의 순위 학습 비교](objective_comparison.png)', '']
    (root/'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1.15, 1]})
    panels = [
        (axes[0], ['BCE: Top1 checkpoint', 'LambdaRank: Top1 checkpoint', 'User-balanced LambdaRank'],
         [r['comparisons'][n+'-baseline-top1'] for n in m['config']['variants']], 'New recipes vs existing BCE'),
        (axes[1], ['User fold 0', 'User fold 1', 'User fold 2', 'Pooled OOF'],
         [m['folds'][str(f)]['randomized_multi']['comparisons']['rank_top1-baseline-top1'] for f in range(3)]+
         [r['comparisons']['rank_top1-baseline-top1']], 'Primary LambdaRank contrast')]
    for ax, labels, items, title in panels:
        for j, item in enumerate(items):
            point, ci = item['estimate'], item['ci95']
            color = '#ac3a37' if ci[1] < 0 else '#4b6076'
            ax.errorbar(point, j, xerr=[[point-ci[0]], [ci[1]-point]], fmt='o', color=color, capsize=4, lw=2)
        ax.axvline(0, color='#7e8792', linestyle='--', lw=1)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, loc='left', pad=13)
        ax.set_xlabel('Change in captured clicks / 1,000 requests', fontsize=9)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-4, 1.5)
    fig.suptitle('Ranking loss did not improve ad choice on held-out users', fontsize=15, y=1.02)
    fig.text(.5, -.04, 'Randomized-display cohort • observed historical clicks • paired user bootstrap 95% intervals\n'
             '27 new models; 3 user folds × 3 seeds × 3 recipes. Reused development data, not online CTR lift.',
             ha='center', fontsize=9, color='#4b5563')
    fig.tight_layout(w_pad=2)
    fig.savefig(root/'objective_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Wrote RESULTS.md and objective_comparison.png')


if __name__ == '__main__':
    main()
