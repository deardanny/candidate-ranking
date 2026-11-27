from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3

from fastapi.testclient import TestClient
import pytest

from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.simulation.analysis import analyze
from fairjob_ranker.simulation.policies import train
from fairjob_ranker.simulation.service import assignment, create_app
from fairjob_ranker.simulation.world import catalog, choose, people, visit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fixture_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-policy')
    cfg = json.loads((ROOT/'configs/simulation_ab.json').read_text())
    cfg.update(training_users=40, validation_users=15)
    offers = catalog(cfg['seed'])
    write_json(root/'configuration.json', cfg)
    write_json(root/'catalog.json', offers)
    train(cfg, offers, root/'models')
    return root


@pytest.fixture
def lab(tmp_path, fixture_bundle):
    root = tmp_path/'lab'
    shutil.copytree(fixture_bundle, root)
    cfg = json.loads((root/'configuration.json').read_text())
    write_json(root/'protocol.json', {'configuration': cfg, 'artifact_sha256': {
        str(p.relative_to(root)): sha256(p) for p in [root/'catalog.json', root/'models/model.txt',
            root/'models/popularity.json', root/'models/metadata.json']}})
    p = people(cfg['seed'], 1, 'eval')[0]
    offers = json.loads((root/'catalog.json').read_text())
    ctx, ids = visit(cfg['seed'], p, 0, offers)
    body = {'experiment_id': 'ab-main', 'request_id': 'test-0', 'user_id': p['user_id'],
            'visit_index': 0, 'context': ctx, 'candidate_ids': ids}
    return root, cfg, body


def feedback(response):
    return {'request_id': response['request_id'],
            'request_snapshot_sha256': response['request_snapshot_sha256'],
            'display_sha256': response['display_sha256'],
            'attribution_window_closed_at': datetime.now(timezone.utc).isoformat(),
            'clicked_item_ids': [response['rankings'][0]['item_id']], 'synthetic': True}


def test_http_inference_sticky_assignment_and_exactly_once_retry(lab):
    root, cfg, body = lab
    with TestClient(create_app(root)) as c:
        r = c.post('/rank', json=body)
        assert r.status_code == 200, r.text
        result = r.json()
        assert result['arm'] == assignment(cfg['assignment_salt'], body['user_id'])
        assert c.post('/rank', json=body).json() == result
        f = feedback(result)
        assert c.post('/events/feedback', json=f).json()['duplicate'] is False
        assert c.post('/events/feedback', json=f).json()['duplicate'] is True
        later = deepcopy(body); later.update(request_id='later', visit_index=1)
        assert c.post('/rank', json=later).json()['arm'] == result['arm']
        assert 'synthetic_feedback_total{status="duplicate"} 1.0' in c.get('/metrics').text
    with sqlite3.connect(root/'events.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0] == 1
    # Real restart preserves original response and does not create a second impression.
    with TestClient(create_app(root)) as c:
        assert c.post('/rank', json=body).json() == result
        assert c.post('/events/feedback', json=f).json()['duplicate'] is True


def test_concurrent_retries_create_one_exposure(lab):
    root, _, body = lab
    with TestClient(create_app(root)) as c:
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: c.post('/rank', json=body), range(12)))
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json() == responses[0].json() for r in responses)
    with sqlite3.connect(root/'events.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 1


@pytest.mark.parametrize('corruption', ['unshown', 'snapshot_hash', 'display_hash', 'future_time', 'past_time'])
def test_feedback_rejects_bad_attribution_without_writing(lab, corruption):
    root, _, body = lab
    with TestClient(create_app(root)) as c:
        result = c.post('/rank', json=body).json()
        f = feedback(result)
        if corruption == 'unshown':
            shown = {r['item_id'] for r in result['rankings']}
            f['clicked_item_ids'] = [next(i for i in body['candidate_ids'] if i not in shown)]
        elif corruption == 'snapshot_hash':
            f['request_snapshot_sha256'] = '0' * 64
        elif corruption == 'display_hash':
            f['display_sha256'] = '0' * 64
        elif corruption == 'future_time':
            f['attribution_window_closed_at'] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        else:
            f['attribution_window_closed_at'] = '2000-01-01T00:00:00Z'
        assert c.post('/events/feedback', json=f).status_code == 400
    with sqlite3.connect(root/'events.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0] == 0


def test_no_click_is_explicit_and_conflicting_retries_are_rejected(lab):
    root, _, body = lab
    with TestClient(create_app(root)) as c:
        result = c.post('/rank', json=body).json()
        changed = deepcopy(body); changed['context']['budget'] += 1
        assert c.post('/rank', json=changed).status_code == 409
        changed = deepcopy(body); changed['request_id'] = 'second-id-same-visit'
        assert c.post('/rank', json=changed).status_code == 409
        f = feedback(result); f['clicked_item_ids'] = []
        assert c.post('/events/feedback', json=f).status_code == 200
        f['clicked_item_ids'] = [result['rankings'][0]['item_id']]
        assert c.post('/events/feedback', json=f).status_code == 409
    with sqlite3.connect(root/'events.sqlite') as db:
        assert db.execute('SELECT clicked_item_id FROM feedback').fetchone()[0] is None


def test_client_cannot_select_arm_leak_label_or_use_invalid_candidates(lab):
    root, _, body = lab
    with TestClient(create_app(root)) as c:
        for key, value in [('arm', 'B'), ('clicked', True), ('private_preferences', [1, 2])]:
            bad = {**body, key: value}
            assert c.post('/rank', json=bad).status_code == 422
        bad = deepcopy(body); bad['candidate_ids'][1] = bad['candidate_ids'][0]
        assert c.post('/rank', json=bad).status_code == 422
        bad = deepcopy(body); bad['candidate_ids'][0] = 'unknown'
        assert c.post('/rank', json=bad).status_code == 400
        bad = deepcopy(body); bad['candidate_ids'][0] = 'ad-036'
        assert c.post('/rank', json=bad).status_code == 400


def test_corrupt_model_or_rebound_database_is_rejected(lab):
    root, _, _ = lab
    with TestClient(create_app(root)):
        pass
    protocol = json.loads((root/'protocol.json').read_text())
    protocol['configuration']['assignment_salt'] = 'changed'
    write_json(root/'protocol.json', protocol)
    with pytest.raises(ValueError, match='another frozen protocol'):
        with TestClient(create_app(root)):
            pass
    with (root/'models/model.txt').open('a') as stream:
        stream.write('\nmodified\n')
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        create_app(root)


def test_lightgbm_ranking_is_candidate_order_invariant(lab):
    root, cfg, body = lab
    users = [f'eval-{i:05d}' for i in range(50) if assignment(cfg['assignment_salt'], f'eval-{i:05d}') == 'B']
    with TestClient(create_app(root)) as c:
        body['user_id'] = users[0]
        a = c.post('/rank', json=body).json()
        body['user_id'] = users[1]; body['request_id'] = 'reordered'
        body['candidate_ids'].reverse()
        b = c.post('/rank', json=body).json()
        assert a['policy'] == b['policy'] == 'lightgbm'
        assert a['rankings'] == b['rankings']


def test_null_response_ignores_item_identity_and_policy_history():
    offers = catalog(123)
    for p in people(123, 30, 'eval'):
        context, _ = visit(123, p, 0, offers)
        a = offers[:3]; b = offers[8:11]
        x = choose(123, p, 0, context, a, {}, 'item_independent')
        y = choose(123, p, 0, context, b, {i['item_id']: 999 for i in b}, 'item_independent')
        assert (None if x is None else [i['item_id'] for i in a].index(x)) == (None if y is None else [i['item_id'] for i in b].index(y))


def test_missing_feedback_blocks_effect_estimation(lab, tmp_path):
    root, cfg, body = lab
    write_json(root/'design_frozen.json', {'configuration': cfg})
    protocol = json.loads((root/'protocol.json').read_text())
    protocol['design_sha256'] = sha256(root/'design_frozen.json')
    write_json(root/'protocol.json', protocol)
    (root/'personas.jsonl').write_text('\n'.join(json.dumps(p) for p in people(cfg['seed'], 3000, 'eval')) + '\n')
    with TestClient(create_app(root)) as c:
        assert c.post('/rank', json=body).status_code == 200
        (root/'metrics.prom').write_text(c.get('/metrics').text)
    (root/'client.jsonl').write_text(json.dumps({'experiment_id': 'ab-main', 'user_id': body['user_id'],
        'request': body, 'rank_ok': True, 'feedback_ok': False, 'rank_latency_ms': 1.0}) + '\n')
    write_json(root/'execution.json', {'protocol_sha256': sha256(root/'protocol.json'),
                                      'offline_http_parity_requests': 0, 'elapsed_seconds': 1})
    result = analyze(root, tmp_path/'report')
    assert result['integrity']['passed'] is False
    assert all(r['difference_pp'] is None and r['conditional_simulation_ci95_pp'] is None for r in result['experiments'])
    assert all(r['arms'][a]['simulated_visit_click_rate'] is None for r in result['experiments'] for a in ('A', 'B'))
