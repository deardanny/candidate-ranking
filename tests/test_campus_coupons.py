from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

from fastapi.testclient import TestClient
import pytest

from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.simulation.campus_world import offers, people, visit, respond, train
from fairjob_ranker.simulation.campus_service import create_app
from fairjob_ranker.simulation.campus_analysis import analyze

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    p = tmp_path_factory.mktemp('campus-bundle')
    cfg = json.loads((ROOT/'configs/campus_coupons.json').read_text())
    cfg.update(training_users=40, validation_users=20)
    ads = offers(cfg['seed']); write_json(p/'catalog.json', ads); write_json(p/'configuration.json', cfg)
    train(cfg, ads, p/'models')
    return p


@pytest.fixture
def lab(bundle, tmp_path):
    p = tmp_path/'lab'; shutil.copytree(bundle, p)
    cfg = json.loads((p/'configuration.json').read_text())
    write_json(p/'design_frozen.json', cfg)
    write_json(p/'protocol.json', {'configuration': cfg, 'design_sha256': sha256(p/'design_frozen.json'),
        'artifact_sha256': {str(f.relative_to(p)): sha256(f) for f in
            [p/'catalog.json', p/'models/click.txt', p/'models/coupon.txt', p/'models/metadata.json']}})
    cohort = people(cfg['seed'], 3000, 'eval')
    (p/'personas.jsonl').write_text('\n'.join(json.dumps(x) for x in cohort)+'\n')
    ctx, ids = visit(cfg['seed'], cohort[0], 0, offers(cfg['seed']))
    body = {'request_id': 'example', 'experiment_id': 'ab-coupon', 'user_id': cohort[0]['user_id'],
            'visit_index': 0, 'context': ctx, 'candidate_ids': ids}
    return p, cfg, body


def events(r, clicked=True):
    common = {k: r[k] for k in ('request_id', 'request_snapshot_sha256', 'display_sha256')}
    item = r['rankings'][0]['item_id']
    click = {**common, 'clicked_item_ids': [item] if clicked else [],
             'attribution_window_closed_at': datetime.now(timezone.utc).isoformat(), 'synthetic': True}
    coupon = {**common, 'redeemed_item_id': item, 'virtual_redemption_delay_minutes': 8.0,
              'virtual_observation_window_minutes': 1440, 'virtual_window_closed': True,
              'clock': 'accelerated-simulation', 'synthetic': True}
    return click, coupon


def test_coupon_requires_click_and_retries_are_idempotent(lab):
    p, cfg, body = lab
    with TestClient(create_app(p)) as c:
        response = c.post('/rank', json=body)
        assert response.status_code == 200, response.text
        r = response.json(); assert len(r['rankings']) == 1
        click, coupon = events(r)
        assert c.post('/events/coupon', json=coupon).status_code == 400
        assert c.post('/events/feedback', json=click).status_code == 200
        assert c.post('/events/coupon', json=coupon).json()['duplicate'] is False
        assert c.post('/events/coupon', json=coupon).json()['duplicate'] is True
        changed = deepcopy(coupon); changed['redeemed_item_id'] = None; changed['virtual_redemption_delay_minutes'] = None
        assert c.post('/events/coupon', json=changed).status_code == 409
    with TestClient(create_app(p)) as c:
        assert c.post('/rank', json=body).json() == r
        assert c.post('/events/coupon', json=coupon).json()['duplicate'] is True
    with sqlite3.connect(p/'events.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM coupons').fetchone()[0] == 1


@pytest.mark.parametrize('bad', ['no_click', 'different_item', 'hash', 'outside_window', 'unclosed'])
def test_invalid_coupon_cannot_be_counted(lab, bad):
    p, _, body = lab
    with TestClient(create_app(p)) as c:
        r = c.post('/rank', json=body).json()
        click, coupon = events(r, clicked=bad != 'no_click')
        assert c.post('/events/feedback', json=click).status_code == 200
        if bad == 'different_item':
            coupon['redeemed_item_id'] = next(i for i in body['candidate_ids'] if i != r['rankings'][0]['item_id'])
        elif bad == 'hash':
            coupon['display_sha256'] = '0'*64
        elif bad == 'outside_window':
            coupon['virtual_redemption_delay_minutes'] = 1441.0
        elif bad == 'unclosed':
            coupon['virtual_window_closed'] = False
        assert c.post('/events/coupon', json=coupon).status_code in (400, 422)
    with sqlite3.connect(p/'events.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM coupons').fetchone()[0] == 0


def test_travel_constraint_changes_only_redemption_not_click():
    ads = offers(1)
    assert ads[0]['price'] == 3 and ads[0]['round_trip_minutes'] == 33
    assert ads[1]['price'] == 4.5 and ads[1]['round_trip_minutes'] == 9
    for p in people(1, 50, 'eval'):
        ctx = {'interest': 0, 'budget': 10., 'minutes_until_class': 25.}
        a = respond(1, p, 0, ctx, ads[0], True)
        b = respond(1, p, 0, ctx, ads[0], False)
        assert a['clicked'] == b['clicked']
        assert a['redeemed'] is False
        assert not b['redeemed'] or b['clicked']


def test_unclosed_coupon_window_blocks_effect_estimation(lab, tmp_path):
    p, cfg, body = lab
    with TestClient(create_app(p)) as c:
        r = c.post('/rank', json=body).json()
        click, coupon = events(r)
        assert c.post('/events/feedback', json=click).status_code == 200
        (p/'metrics.prom').write_text(c.get('/metrics').text)
    (p/'client.jsonl').write_text(json.dumps({'request': body, 'ok': False, 'response': r})+'\n')
    write_json(p/'execution.json', {'protocol_sha256': sha256(p/'protocol.json'),
        'counts': {'visits': 1, 'errors': 1, 'parity': 0, 'retry_triplets': 0}, 'elapsed_seconds': 1})
    result = analyze(p, tmp_path/'report')
    assert not result['integrity']['passed']
    assert all(r['effects']['coupon'] is None for r in result['experiments'])
    assert all(r['arms'][a]['coupon_rate'] is None for r in result['experiments'] for a in ('A', 'B'))
