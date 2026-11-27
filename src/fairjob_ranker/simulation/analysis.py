"""Audit durable logs first; estimate only conditional synthetic-world effects."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import binomtest

from ..io import sha256, write_json
from ..log_contract import RequestSnapshot, ClosedFeedback
from .service import assignment
from .world import digest


def interval(a, b, seed, resamples):
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for i in range(resamples):
        draws[i] = np.mean(rng.choice(b, len(b), replace=True)) - np.mean(rng.choice(a, len(a), replace=True))
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def analyze(run, report):
    run, report = Path(run), Path(report)
    protocol = json.loads((run/'protocol.json').read_text())
    config = protocol['configuration']
    people = [json.loads(line) for line in (run/'personas.jsonl').read_text().splitlines()]
    offers = {a['item_id']: a for a in json.loads((run/'catalog.json').read_text())}
    arms = {p['user_id']: assignment(config['assignment_salt'], p['user_id']) for p in people}
    requested = defaultdict(list)
    errors, client_records = [], {}
    attempt_counts, failures, retried, parity_receipt = Counter(), [], 0, json.loads((run/'execution.json').read_text())
    latency = defaultdict(list)
    for line in (run/'client.jsonl').open():
        row = json.loads(line)
        request_id = row['request']['request_id']
        if request_id in client_records:
            errors.append(f'Duplicate client visit: {request_id}')
        client_records[request_id] = {
            'request_digest': digest({'schema_version': 'synthetic-rank-v1', **row['request']}),
            'response_digest': digest(row['response']) if row.get('response') else None,
            'choice': row.get('chosen_item_id')}
        experiment = row['experiment_id']
        attempt_counts[experiment] += 1
        if not (row['rank_ok'] and row['feedback_ok']) or row.get('error'):
            failures.append({'request_id': row['request']['request_id'], 'error': row.get('error', 'incomplete')})
        if row.get('rank_latency_ms') is not None:
            latency[experiment].append(row['rank_latency_ms'])
        retried += int(row.get('idempotent_retry_checked', False))
    for name, expected in protocol['artifact_sha256'].items():
        if sha256(run/name) != expected:
            errors.append(f'Artifact mismatch: {name}')
    if sha256(run/'design_frozen.json') != protocol['design_sha256']:
        errors.append('Frozen design changed')
    if parity_receipt['protocol_sha256'] != sha256(run/'protocol.json'):
        errors.append('Execution protocol mismatch')
    db = sqlite3.connect(f'file:{(run / "events.sqlite").resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    if db.execute("SELECT value FROM metadata WHERE key='protocol_sha256'").fetchone()[0] != sha256(run/'protocol.json'):
        errors.append('Database protocol mismatch')
    if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        errors.append('SQLite integrity check failed')
    if db.execute('PRAGMA foreign_key_check').fetchall():
        errors.append('Foreign key failure')
    closed_count = 0
    for row in db.execute('''SELECT r.*,f.body AS feedback_body,f.clicked_item_id
                            FROM requests r LEFT JOIN feedback f USING(request_id)'''):
        try:
            snapshot = RequestSnapshot.model_validate_json(row['snapshot'])
            response = json.loads(row['response'])
            client_record = client_records.get(row['request_id'])
            if (client_record is None or client_record['request_digest'] != row['request_digest']
                    or client_record['response_digest'] != digest(response)):
                raise ValueError('Client/server request or response mismatch')
            if snapshot.digest() != row['snapshot_digest'] or response['request_snapshot_sha256'] != snapshot.digest():
                raise ValueError('Snapshot digest mismatch')
            expected_display = digest({'request_snapshot_sha256': snapshot.digest(), 'rankings': response['rankings']})
            if row['display_digest'] != expected_display or response['display_sha256'] != expected_display:
                raise ValueError('Display digest mismatch')
            if row['arm'] != arms[row['user_id']] or response['arm'] != row['arm']:
                raise ValueError('Assignment changed')
            shown = [x['item_id'] for x in response['rankings']]
            if len(shown) != 3 or len(set(shown)) != 3 or not set(shown) <= {a.item_id for a in snapshot.candidates}:
                raise ValueError('Invalid displayed candidates')
            if len(snapshot.candidates) != 8:
                raise ValueError('Incomplete candidate set')
            policy = next(e['policies'][row['arm']] for e in config['experiments'] if e['id'] == row['experiment_id'])
            if row['policy'] != policy or snapshot.policy_version != policy:
                raise ValueError('Wrong experiment policy')
            clicked = row['clicked_item_id']
            if row['feedback_body'] is not None and client_record['choice'] != clicked:
                raise ValueError('Client/server feedback mismatch')
            closed = row['feedback_body'] is not None
            if closed:
                feedback = ClosedFeedback.model_validate_json(row['feedback_body'])
                feedback.validate_against(snapshot)
                if feedback.clicked_item_ids != ([] if clicked is None else [clicked]) or (clicked is not None and clicked not in shown):
                    raise ValueError('Click attribution mismatch')
                closed_count += 1
            requested[row['experiment_id']].append({
                'user_id': row['user_id'], 'visit_index': row['visit_index'], 'arm': row['arm'],
                'closed': closed, 'clicked': int(clicked is not None),
                'top1_click': int(clicked == shown[0]),
                'cold_eligible': any(offers[c.item_id]['cold'] for c in snapshot.candidates),
                'cold_click': int(clicked is not None and offers[clicked]['cold']),
                'cold_displayed': sum(offers[i]['cold'] for i in shown)})
        except (ValueError, KeyError, StopIteration) as error:
            errors.append(f"{row['request_id']}: {error}")
    db_counts = {table: db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in ('assignments', 'requests', 'feedback')}
    db.close()
    expected_visits = config['evaluation_users'] * config['visits_per_user']
    total_expected = expected_visits * len(config['experiments'])
    if len(client_records) != total_expected or any(attempt_counts[e['id']] != expected_visits for e in config['experiments']):
        errors.append('Client scheduled-visit accounting mismatch')
    arm_users = Counter(arms.values())
    srm_p = float(binomtest(arm_users['A'], len(people), .5).pvalue)
    results = []
    for experiment in config['experiments']:
        rows = requested[experiment['id']]
        valid = (len(rows) == expected_visits and all(r['closed'] for r in rows) and not errors
                 and not any(f['request_id'].startswith(experiment['id'] + ':') for f in failures))
        rates = {}
        summaries = {}
        for arm in ('A', 'B'):
            group = [r for r in rows if r['arm'] == arm]
            user_clicks = Counter(r['user_id'] for r in group if r['clicked'])
            user_counts = Counter(r['user_id'] for r in group)
            eligible = [r for r in group if r['cold_eligible']]
            uid = [u for u in arms if arms[u] == arm]
            if any(user_counts[u] != config['visits_per_user'] for u in uid):
                valid = False
            rates[arm] = np.array([user_clicks[u] / config['visits_per_user'] for u in uid])
            summaries[arm] = {'policy': experiment['policies'][arm], 'users': len(uid),
                'scheduled_visits': len(uid) * config['visits_per_user'], 'served_visits': len(group),
                'closed_feedback': sum(r['closed'] for r in group), 'simulated_clicks': sum(r['clicked'] for r in group),
                'simulated_visit_click_rate': float(rates[arm].mean()) if valid else None,
                'top1_click_per_scheduled_visit': sum(r['top1_click'] for r in group) / (len(uid) * config['visits_per_user']) if valid else None,
                'cold_eligible_visits': len(eligible), 'cold_clicks': sum(r['cold_click'] for r in eligible),
                'cold_click_per_eligible_visit': sum(r['cold_click'] for r in eligible) / len(eligible) if valid and eligible else None,
                'cold_displayed_slots': sum(r['cold_displayed'] for r in eligible)}
        if not valid:
            # Never expose an earlier-computed rate when a later arm invalidates the experiment.
            for arm in summaries.values():
                for name in ('simulated_visit_click_rate', 'top1_click_per_scheduled_visit', 'cold_click_per_eligible_visit'):
                    arm[name] = None
        ci = interval(rates['A'], rates['B'], config['seed'], config['bootstrap_resamples']) if valid else None
        results.append({'experiment_id': experiment['id'], 'response_model': experiment['response'],
            'complete': valid, 'arms': summaries,
            'difference_pp': float((rates['B'].mean() - rates['A'].mean()) * 100) if valid else None,
            'conditional_simulation_ci95_pp': [v * 100 for v in ci] if ci else None,
            'rank_http_p50_ms': float(np.percentile(latency[experiment['id']], 50)) if latency[experiment['id']] else None,
            'rank_http_p95_ms': float(np.percentile(latency[experiment['id']], 95)) if latency[experiment['id']] else None,
            'attempted_visits': attempt_counts[experiment['id']]})
    integrity = {'passed': not errors and not failures and all(r['complete'] for r in results)
                 and db_counts['requests'] == db_counts['feedback'] == total_expected
                 and db_counts['assignments'] == config['evaluation_users'] * len(config['experiments']),
        'errors': errors[:30], 'error_count': len(errors), 'client_failures': failures[:30],
        'client_failure_count': len(failures), 'database_counts': db_counts,
        'expected_visits_per_experiment': expected_visits, 'validated_closed_feedback': closed_count,
        'idempotent_request_and_feedback_retries': retried,
        'offline_http_parity_requests': parity_receipt['offline_http_parity_requests'],
        'missing_feedback_imputed_as_zero': False,
        'users_per_arm': dict(arm_users), 'user_level_srm_binomial_p': srm_p,
        'srm_flag_at_0_001': srm_p < .001}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(),
        'synthetic': True, 'human_participants': 0, 'real_clicks': 0, 'llm_respondent_calls': 0,
        'unique_evaluation_personas': len(people), 'visits_per_person_per_experiment': config['visits_per_user'],
        'response_source': config['response_source'], 'comparison': config['comparison'],
        'primary_metric': config['primary_metric'], 'uncertainty': config['uncertainty'],
        'limitations': config['limitations'], 'integrity': integrity, 'experiments': results,
        'elapsed_seconds': parity_receipt['elapsed_seconds'],
        'evidence_sha256': {name: sha256(run/name) for name in
            ('protocol.json', 'execution.json', 'events.sqlite', 'client.jsonl', 'metrics.prom')},
        'run_directory': str(run.resolve())}
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'results.json', result)
    write_report(report, result)
    return result


def write_report(out, result):
    lines = ['# 가상 사용자 3,000명 모의 A/B 테스트', '',
        '**실제 사람·실제 클릭·LLM 응답은 각각 0개다.** 확률형 행동 시뮬레이터로 생성한 결과다.', '',
        '기존 FairJob/EB-NeRD 모델을 재평가한 결과가 아니다. 가상 광고 환경에서 별도로 학습한 인기순과 문맥 LightGBM을 비교했다.', '',
        '## 결과', '', '| 실험 | A 선택률 | B 선택률 | B − A, %p | 조건부 95% 구간, %p |',
        '|---|---:|---:|---:|---:|']
    for row in result['experiments']:
        if row['complete']:
            a = row['arms']['A']['simulated_visit_click_rate'] * 100
            b = row['arms']['B']['simulated_visit_click_rate'] * 100
            low, high = row['conditional_simulation_ci95_pp']
            lines.append(f"| {row['experiment_id']} | {a:.2f}% | {b:.2f}% | {row['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
        else:
            lines.append(f"| {row['experiment_id']} | 불완전 | 불완전 | 계산 보류 | 계산 보류 |")
    lines += ['', '선택률은 방문당 모의 클릭 여부를 사용자별로 평균 낸 값이다. 각 사용자는 8회 방문한다. 구간은 사용자 단위 부트스트랩이며, 우리가 정의한 시뮬레이터 내부의 표본 변동만 나타낸다. 실제 사람에 대한 신뢰구간이 아니다.', '',
        '- `aa-check`: 양쪽 모두 동일한 인기순 랭커. 같은 모델에서도 표본 차이는 발생할 수 있다.',
        '- `ab-main`: 인기순과 문맥 LightGBM 비교. 사용자 취향·제약을 반영하는 반응 모델.',
        '- `ab-null`: 같은 두 랭커를 비교하되 반응이 광고 정체성과 무관하도록 설정한 음성 대조군.', '',
        '세 실험은 같은 3,000명을 재사용한다. 9,000명의 독립된 사용자가 아니다. 각 조건의 결과를 보고 시뮬레이터나 모델을 재조정하지 않는다.', '',
        '## 연결한 실행 경로', '',
        '`가상 사용자 → 실제 HTTP /rank → 사용자별 고정 배정 → 모델 추론 → SQLite 노출 스냅샷 → 별도 반응 생성기 → HTTP /events/feedback → 중복 제거·귀속 검증 → 사용자 단위 분석`', '',
        'FastAPI, LightGBM, 기존 RequestSnapshot/ClosedFeedback 검증 계약을 사용했다. 서버가 클릭을 만들어 자기 성능을 평가하지 않도록 반응 생성은 클라이언트에서 실행한다. 요청·피드백은 실제 서버 시각으로 기록하고, 가상 방문 순서는 별도로 저장한다.', '',
        '## 실행 무결성', '',
        f"- 전체 통과: **{result['integrity']['passed']}**",
        f"- 사용자 배정: {result['integrity']['users_per_arm']}",
        f"- DB 기록: {result['integrity']['database_counts']}",
        f"- 닫힌 피드백 검증: {result['integrity']['validated_closed_feedback']:,}건",
        f"- 요청·피드백 재전송 중복 방지 확인: {result['integrity']['idempotent_request_and_feedback_retries']}쌍",
        f"- HTTP/직접 추론 일치: {result['integrity']['offline_http_parity_requests']}건",
        f"- 클라이언트 실패: {result['integrity']['client_failure_count']}건",
        f"- 사용자 수 비율 검사 p: {result['integrity']['user_level_srm_binomial_p']:.4f}",
        f"- HTTP 실험 소요 시간: {result['elapsed_seconds']:.1f}초", '',
        '서버와 생성기는 같은 컴퓨터에서 실행했다. 이 시간과 지연은 폐회로 모의 실행 결과이며, 운영 트래픽 처리 용량 측정이 아니다. 노출을 받지 못하거나 피드백이 빠진 방문은 무클릭으로 채우지 않는다. 피드백이 불완전하면 효과 계산을 보류한다.', '',
        '## 해석 한계', '']
    lines += ['- ' + s for s in result['limitations']]
    lines += ['', '시뮬레이터를 바꿔도 효과가 그대로 유지되는지 보는 것이 중요하다. 클릭 수가 늘었다는 사실만으로 실제 광고 효과를 주장하지 않는다. 이 실험에서 확인한 결과물은 재현 가능한 실험 시스템과 설정한 환경에서의 정책 차이다.', '',
        '설계 참고: [RecSim](https://arxiv.org/abs/1909.04847)은 사용자 상태와 반응 가정을 명시적으로 조절하는 추천 시뮬레이션을 다룬다. 이 구현은 RecSim 라이브러리를 사용하거나 인간 행동으로 보정한 모델이 아니다.', '',
        '[정확한 집계·검증 결과](results.json) · [실행 및 API 안내](../../docs/SYNTHETIC_AB.md)', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.report)
    if not result['integrity']['passed']:
        raise SystemExit('Integrity check failed')


if __name__ == '__main__':
    main()
