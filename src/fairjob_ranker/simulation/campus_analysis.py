"""Audit exposure → detail click → coupon redemption; use all visits as denominator."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import binomtest

from ..io import sha256, write_json
from ..log_contract import RequestSnapshot, ClosedFeedback
from .analysis import interval
from .service import assignment
from .world import digest
from .campus_service import Coupon


def analyze(run, report):
    run, report = Path(run), Path(report)
    protocol = json.loads((run/'protocol.json').read_text()); cfg = protocol['configuration']
    people = [json.loads(x) for x in (run/'personas.jsonl').read_text().splitlines()]
    arms = {p['user_id']: assignment(cfg['assignment_salt'], p['user_id']) for p in people}
    ads = {a['item_id']: a for a in json.loads((run/'catalog.json').read_text())}
    execution = json.loads((run/'execution.json').read_text())
    errors, clients = [], {}
    expected = cfg['evaluation_users']*cfg['visits']*len(cfg['experiments'])
    for name, h in protocol['artifact_sha256'].items():
        if sha256(run/name) != h:
            errors.append('Changed artifact: '+name)
    if sha256(run/'design_frozen.json') != protocol['design_sha256'] or sha256(run/'protocol.json') != execution['protocol_sha256']:
        errors.append('Frozen protocol mismatch')
    for line in (run/'client.jsonl').open():
        c = json.loads(line); rid = c['request']['request_id']
        if rid in clients or not c['ok']:
            errors.append('Duplicate or failed client request: '+rid)
        clients[rid] = c
    if len(clients) != expected:
        errors.append('Scheduled visit accounting mismatch')
    db = sqlite3.connect(f'file:{(run / "events.sqlite").resolve()}?mode=ro', uri=True); db.row_factory = sqlite3.Row
    if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchall():
        errors.append('Database integrity failure')
    if db.execute("SELECT value FROM metadata WHERE key='protocol_sha256'").fetchone()[0] != sha256(run/'protocol.json'):
        errors.append('Database protocol mismatch')
    counts = {t: db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('assignments', 'requests', 'feedback', 'coupons')}
    if any(counts[t] != expected for t in ('requests', 'feedback', 'coupons')) or counts['assignments'] != cfg['evaluation_users']*len(cfg['experiments']):
        errors.append('Missing exposure, click closure or coupon closure; do not fill negatives')
    rows = defaultdict(list)
    for r in db.execute('''SELECT r.*,f.body AS click_body,f.clicked_item_id,q.body AS coupon_body,
                          q.redeemed_item_id,q.received_at AS coupon_received
                          FROM requests r LEFT JOIN feedback f USING(request_id)
                          LEFT JOIN coupons q USING(request_id)'''):
        try:
            c = clients[r['request_id']]
            snapshot = RequestSnapshot.model_validate_json(r['snapshot'])
            response = json.loads(r['response'])
            click = ClosedFeedback.model_validate_json(r['click_body'])
            coupon = Coupon.model_validate_json(r['coupon_body'])
            click.validate_against(snapshot)
            checks = [digest(c['request']) == r['request_digest'],
                digest(c['response']) == digest(response), snapshot.digest() == r['snapshot_digest'],
                response['request_snapshot_sha256'] == snapshot.digest(),
                r['arm'] == arms[r['user_id']] == response['arm'],
                len(snapshot.candidates) == 8, len(response['rankings']) == 1,
                digest(c['coupon_event']) == digest(coupon.model_dump()),
                coupon.request_snapshot_sha256 == snapshot.digest(),
                coupon.display_sha256 == r['display_digest'] == response['display_sha256'],
                digest({'request_snapshot_sha256': snapshot.digest(), 'rankings': response['rankings']}) == r['display_digest']]
            top = response['rankings'][0]['item_id']
            if top not in {a.item_id for a in snapshot.candidates}:
                raise ValueError('Display outside candidate pool')
            expected_policy = next(e['policies'][r['arm']] for e in cfg['experiments'] if e['id'] == r['experiment_id'])
            checks += [r['policy'] == expected_policy == response['policy'] == snapshot.policy_version,
                       bool(r['clicked_item_id']) == c['outcome']['clicked'],
                       bool(r['redeemed_item_id']) == c['outcome']['redeemed'],
                       r['clicked_item_id'] in (None, top),
                       click.clicked_item_ids == ([] if r['clicked_item_id'] is None else [top]),
                       coupon.redeemed_item_id == r['redeemed_item_id'],
                       r['redeemed_item_id'] is None or r['redeemed_item_id'] == r['clicked_item_id'],
                       coupon.request_id == r['request_id']]
            if not all(checks):
                raise ValueError('Client/server attribution or assignment mismatch')
            rows[r['experiment_id']].append({'user_id': r['user_id'], 'arm': r['arm'],
                'click': int(r['clicked_item_id'] is not None), 'coupon': int(r['redeemed_item_id'] is not None)})
        except (ValueError, KeyError, TypeError, StopIteration) as e:
            errors.append(f"{r['request_id']}: {e}")
    db.close()
    results = []
    for exp in cfg['experiments']:
        group = rows[exp['id']]
        complete = not errors and len(group) == cfg['evaluation_users']*cfg['visits']
        arm_stats, rates = {}, {}
        for arm in ('A', 'B'):
            users = [u for u in arms if arms[u] == arm]
            selected = [r for r in group if r['arm'] == arm]
            visits = Counter(r['user_id'] for r in selected)
            if any(visits[u] != cfg['visits'] for u in users):
                complete = False
            totals = {m: Counter(r['user_id'] for r in selected if r[m]) for m in ('click', 'coupon')}
            rates[arm] = {m: np.array([totals[m][u]/cfg['visits'] for u in users]) for m in totals}
            clicks, coupons = sum(totals['click'].values()), sum(totals['coupon'].values())
            arm_stats[arm] = {'users': len(users), 'visits': len(selected), 'clicks': clicks, 'coupons': coupons,
                'click_rate': clicks/(len(users)*cfg['visits']), 'coupon_rate': coupons/(len(users)*cfg['visits']),
                'clicked_without_coupon_per_visit': (clicks-coupons)/(len(users)*cfg['visits']),
                'coupon_per_click_descriptive': coupons/clicks if clicks else None}
        effects = {}
        for metric in ('click', 'coupon'):
            if complete:
                ci = interval(rates['A'][metric], rates['B'][metric], cfg['seed'], cfg['bootstrap_resamples'])
                effects[metric] = {'difference_pp': float((rates['B'][metric].mean()-rates['A'][metric].mean())*100),
                                  'conditional_ci95_pp': [x*100 for x in ci]}
            else:
                effects[metric] = None
                for arm in arm_stats.values():
                    for name in ('click_rate', 'coupon_rate', 'clicked_without_coupon_per_visit', 'coupon_per_click_descriptive'):
                        arm[name] = None
        results.append({'experiment_id': exp['id'], 'policies': exp['policies'],
            'travel_constraint': exp['travel_constraint'], 'complete': complete, 'arms': arm_stats, 'effects': effects})
    example = None
    if not errors:
        for uid in sorted(u for u in arms if arms[u] == 'B'):
            a, b = [clients[f'{exp}:{uid}:0'] for exp in ('aa-click', 'ab-coupon')]
            ia, ib = [c['response']['rankings'][0]['item_id'] for c in (a, b)]
            if ia != ib:
                example = {'selection': cfg['example_rule'], 'user_id': uid, 'visit_index': 0,
                    'context': a['request']['context'], 'candidate_ids': a['request']['candidate_ids'],
                    'click_policy': {'ad': ads[ia], 'observed_simulated_outcome': a['outcome']},
                    'coupon_policy': {'ad': ads[ib], 'observed_simulated_outcome': b['outcome']},
                    'interpretation': 'Same invented persona in two separate simulation conditions, not a person receiving both arms in the randomized comparison.'}
                break
    result = {'synthetic': True, 'humans': 0, 'llm_respondent_calls': 0,
        'configuration': cfg, 'experiments': results, 'example': example,
        'integrity': {'passed': not errors and all(r['complete'] for r in results),
            'error_count': len(errors), 'errors': errors[:30], 'database_counts': counts,
            'users_per_arm': dict(Counter(arms.values())),
            'assignment_srm_p': float(binomtest(sum(v == 'A' for v in arms.values()), len(arms), .5).pvalue),
            'client_execution': execution['counts'], 'missing_outcomes_imputed': False},
        'elapsed_seconds': execution['elapsed_seconds'],
        'evidence_sha256': {n: sha256(run/n) for n in ('protocol.json', 'execution.json', 'events.sqlite', 'client.jsonl', 'metrics.prom')},
        'uncertainty': 'Within-arm user bootstrap; conditional on invented world, not population uncertainty about humans.',
        'run_directory': str(run.resolve())}
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'results.json', result)
    if example:
        write_json(report/'example.json', example)
    write_report(result, report)
    return result


def write_report(result, report):
    lines = ['# 클릭받은 광고가 손님을 데려왔을까?', '',
        '가상의 캠퍼스 앱에서 학교 주변 매장의 협찬 쿠폰 한 자리를 고르는 실험이다. 광고주는 쿠폰을 실제로 사용한 손님을 원한다고 가정한다. 사용자는 다음 수업까지 남은 시간과 예산이 있다.', '',
        '**모든 광고·사용자·클릭·쿠폰 사용은 합성이다. LLM 응답도 실제 사람도 없다.** 이전의 익명 숫자 피처 기반 FairJob 실험이나 뉴스 모델의 평가가 아니다.', '',
        '## 광고는 무엇인가', '',
        '| 가상 광고주 | 광고 상품 | 쿠폰 가격 | 편도 도보 | 왕복·이용 소요 |',
        '|---|---|---:|---:|---:|',
        '| North Gate Coffee | 아메리카노 50% 할인 | $3.00 | 15분 | 33분 |',
        '| Library Coffee | 아메리카노 10% 할인 | $4.50 | 3분 | 9분 |', '',
        '총 24개의 가상 매장이 커피·점심·작업 공간 쿠폰을 광고한다. 모두 유효하고 영업 중인 동일 조건의 협찬 광고로 가정한다. 후보 8개 중 화면에는 1개만 표시한다. 입찰가 차이, 캠페인 예산 소진, 재고 경쟁은 모델링하지 않는다.', '',
        '다음 수업까지 25분이라면 첫 광고는 눈길을 끌어도 다녀올 시간이 부족하다. 두 번째는 할인 폭이 작아도 이용할 수 있다. 이런 클릭과 이용의 차이를 일부러 넣은 스트레스 테스트이며, 실제 사람에게서 발견한 현상이라고 주장하지 않는다.', '',
        '## 바꾼 점', '',
        '- A: 광고 상세를 클릭할 가능성이 높은 광고를 선택한다.',
        '- B: 노출 뒤 클릭하고 쿠폰까지 사용할 가능성이 높은 광고를 선택한다.',
        '- 두 모델은 동일한 피처와 무작위 노출 훈련 로그를 사용한다. LightGBM의 학습 정답만 다르다.',
        '- 주 지표는 전체 예정 방문당 쿠폰 사용률이다. 클릭한 사람만으로 분모를 좁혀 개선을 주장하지 않는다.', '',
        '쿠폰 모델은 클릭한 표본만 학습하는 CVR 모델이 아니다. 관측 창이 끝난 전체 노출에서 `click AND redemption`을 예측한다. ESMM의 순차 행동 관점과 관련되지만 ESMM 자체를 구현한 것은 아니다.', '',
        '## 실제 실행 결과', '',
        '| 조건 | A 클릭률 | B 클릭률 | A 쿠폰 사용률 | B 쿠폰 사용률 | 쿠폰 차이, %p |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['experiments']:
        if r['complete']:
            a,b = r['arms']['A'],r['arms']['B']
            lines.append(f"| {r['experiment_id']} | {100*a['click_rate']:.2f}% | {100*b['click_rate']:.2f}% | {100*a['coupon_rate']:.2f}% | {100*b['coupon_rate']:.2f}% | {r['effects']['coupon']['difference_pp']:+.2f} |")
        else:
            lines.append(f"| {r['experiment_id']} | 불완전 | 불완전 | 분석 보류 | 분석 보류 | — |")
    lines += ['', '`aa-click`은 같은 클릭 모델끼리 비교한다. `ab-coupon`은 클릭 목표와 쿠폰 목표를 비교한다. `ab-no-travel`은 쿠폰 사용 단계의 이동·시간 제약만 제거하며 할인 관심·예산·취향은 유지한다. 같은 모델을 그대로 사용하고 재학습하지 않는다.', '',
        f"검증 통과: **{result['integrity']['passed']}**. 기록: {result['integrity']['database_counts']}. 사용자 배정: {result['integrity']['users_per_arm']}.",
        f"실행 집계: {result['integrity']['client_execution']}. HTTP 실행 구간: {result['elapsed_seconds']:.1f}초.", '',
        '같은 3,000명을 조건마다 8회씩 재사용했다. 원본 요청·피드백·쿠폰 이벤트를 모두 대조했다. 모든 방문의 쿠폰 관측 창이 닫힌 뒤 분석하며 누락을 무사용으로 채우지 않는다. 24시간 관측 창은 가상 시간을 가속해서 처리했고 실제로 하루를 기다린 기록이 아니다. 서버 수신 시각과 가상 쿠폰 사용 지연은 별도 필드다.', '',
        '조건부 95% 구간과 원본 해시는 [results.json](results.json)에 있다. 사용자 단위로 재표집하며 사람에 대한 신뢰구간이 아니다. [한 사용자 예시](example.json)는 사전에 정한 규칙으로 고른 두 조건의 시뮬레이터 재생이다.', '',
        '## 이 결과를 어떻게 쓰는가', '',
        '이 프로젝트의 스토리는 “실제 CTR을 올렸다”가 아니라, **광고 최적화 목표를 클릭에서 관측 가능한 후속 행동으로 바꾸고, 노출→클릭→쿠폰 귀속을 끝까지 구현했다**는 것이다. 반응 가정이 바뀌면 결과도 달라지는지 함께 확인한다. 실제 서비스로 확장하려면 매장 쿠폰 사용 이벤트와 실제 트래픽을 수집해야 한다.', '',
        '참고: [Google의 전환 측정 설명](https://support.google.com/google-ads/answer/1722022?hl=en), [노출→클릭→전환을 다룬 ESMM 논문](https://arxiv.org/abs/1804.07931). 이 실험의 계수를 이 자료에서 추정한 것은 아니다.', '']
    (report/'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    args = p.parse_args(); result = analyze(args.run, args.report)
    if not result['integrity']['passed']:
        raise SystemExit('Integrity audit failed')


if __name__ == '__main__':
    main()
