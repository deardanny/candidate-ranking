"""Run the campus coupon story over real HTTP, preserving the earlier experiment."""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import numpy as np

from ..io import sha256, write_json
from .world import canonical
from .campus_world import offers, people, visit, respond, train, Policies

ROOT = Path(__file__).resolve().parents[3]


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare(config_path, out):
    cfg = json.loads(Path(config_path).read_text())
    if cfg['evaluation_users'] != 3000 or cfg['visits'] != 8:
        raise ValueError('Frozen study requires 3000 users and eight visits')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'configuration.json', cfg)
    ads = offers(cfg['seed']); write_json(out/'catalog.json', ads)
    with (out/'personas.jsonl').open('w') as f:
        for p in people(cfg['seed'], cfg['evaluation_users'], 'eval'):
            f.write(canonical(p)+'\n')
    sources = list((ROOT/'src/fairjob_ranker/simulation').glob('*.py')) + [ROOT/'src/fairjob_ranker/log_contract.py']
    write_json(out/'design_frozen.json', {'frozen_utc': utc(), 'configuration': cfg,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'evaluation_outcomes_generated': False})
    print('Frozen campus story; training click and joint-redemption targets', flush=True)
    train(cfg, ads, out/'models')
    write_json(out/'protocol.json', {'frozen_utc': utc(), 'configuration': cfg,
        'evaluation_outcomes_generated': False, 'design_sha256': sha256(out/'design_frozen.json'),
        'artifact_sha256': {str(p.relative_to(out)): sha256(p) for p in
            [out/'configuration.json', out/'catalog.json', out/'personas.jsonl',
             out/'models/click.txt', out/'models/coupon.txt', out/'models/metadata.json']}})
    return cfg


@contextmanager
def server(out):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    process = None
    try:
        listener.bind(('127.0.0.1', 0)); listener.listen(128)
        port = listener.getsockname()[1]
        env = os.environ.copy(); env['CAMPUS_COUPON_RUN'] = str(out.resolve()); env['OMP_NUM_THREADS'] = '1'
        with (out/'server.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn',
                'fairjob_ranker.simulation.campus_service:app_factory', '--factory', '--fd', str(listener.fileno()),
                '--no-access-log', '--log-level', 'warning'], env=env, pass_fds=(listener.fileno(),),
                stdout=log, stderr=subprocess.STDOUT)
            listener.close()
            url = f'http://127.0.0.1:{port}'; deadline = time.monotonic()+30
            with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Campus server exited; see server.log')
                    try:
                        r = client.get('/readyz')
                        if r.status_code == 200 and r.json()['ready']:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server readiness timed out')
                    time.sleep(.1)
            yield url
    finally:
        listener.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)


async def drive(out, url, cfg):
    cohort = [json.loads(line) for line in (out/'personas.jsonl').read_text().splitlines()]
    ads = json.loads((out/'catalog.json').read_text()); by_id = {a['item_id']: a for a in ads}
    direct = Policies(out/'models')
    sem = asyncio.Semaphore(cfg['concurrency'])
    counts = {'visits': 0, 'errors': 0, 'parity': 0, 'retry_triplets': 0}
    started = time.perf_counter()
    with (out/'client.jsonl').open('w', buffering=1) as log:
        async with httpx.AsyncClient(base_url=url, timeout=30, trust_env=False,
            limits=httpx.Limits(max_connections=cfg['concurrency'])) as client:
            for exp in cfg['experiments']:
                completed = 0

                async def one(person):
                    nonlocal completed
                    async with sem:
                        for i in range(cfg['visits']):
                            ctx, ids = visit(cfg['seed'], person, i, ads)
                            body = {'experiment_id': exp['id'], 'request_id': f"{exp['id']}:{person['user_id']}:{i}",
                                'user_id': person['user_id'], 'visit_index': i, 'context': ctx, 'candidate_ids': ids}
                            record = {'request': body, 'synthetic': True, 'ok': False}
                            start = time.perf_counter()
                            try:
                                rank = await client.post('/rank', json=body); rank.raise_for_status()
                                record['rank_latency_ms'] = (time.perf_counter()-start)*1000
                                response = rank.json(); record['response'] = response
                                ad = by_id[response['rankings'][0]['item_id']]
                                outcome = respond(cfg['seed'], person, i, ctx, ad, exp['travel_constraint'])
                                record['outcome'] = outcome
                                common = {'request_id': body['request_id'],
                                    'request_snapshot_sha256': response['request_snapshot_sha256'],
                                    'display_sha256': response['display_sha256']}
                                click = {**common, 'clicked_item_ids': [ad['item_id']] if outcome['clicked'] else [],
                                    'attribution_window_closed_at': utc(), 'synthetic': True}
                                c = await client.post('/events/feedback', json=click); c.raise_for_status()
                                coupon = {**common, 'redeemed_item_id': ad['item_id'] if outcome['redeemed'] else None,
                                    'virtual_redemption_delay_minutes': outcome['virtual_redemption_delay_minutes'],
                                    'virtual_observation_window_minutes': 1440, 'virtual_window_closed': True,
                                    'clock': 'accelerated-simulation', 'synthetic': True}
                                d = await client.post('/events/coupon', json=coupon); d.raise_for_status()
                                record['coupon_event'] = coupon
                                if int(person['user_id'].split('-')[1]) < 5:
                                    scores = direct.score(response['policy'], ctx, [by_id[a] for a in ids])
                                    top, score = sorted(zip(ids, scores), key=lambda x: (-float(x[1]), x[0]))[0]
                                    assert top == ad['item_id']
                                    np.testing.assert_allclose(score, response['rankings'][0]['score'], atol=1e-12)
                                    counts['parity'] += 1
                                if i == 0 and int(person['user_id'].split('-')[1]) % 100 == 0:
                                    r = await client.post('/rank', json=body); r.raise_for_status()
                                    assert r.json() == response
                                    for endpoint, payload in [('/events/feedback', click), ('/events/coupon', coupon)]:
                                        r = await client.post(endpoint, json=payload); r.raise_for_status()
                                        assert r.json()['duplicate']
                                    counts['retry_triplets'] += 1
                                record['ok'] = True
                            except (httpx.HTTPError, ValueError, AssertionError) as error:
                                record['error'] = f'{type(error).__name__}: {error}'
                                counts['errors'] += 1
                            log.write(canonical(record)+'\n'); counts['visits'] += 1
                        completed += 1
                        if completed % 500 == 0:
                            print(f"{exp['id']}: {completed}/3000 users", flush=True)

                await asyncio.gather(*(one(p) for p in cohort))
            r = await client.get('/metrics'); r.raise_for_status(); (out/'metrics.prom').write_text(r.text)
    write_json(out/'execution.json', {'completed_utc': utc(), 'counts': counts,
        'elapsed_seconds': time.perf_counter()-started, 'protocol_sha256': sha256(out/'protocol.json'),
        'transport': 'actual HTTP to owned uvicorn child; real receipt timestamps; accelerated virtual coupon window',
        'humans': 0, 'llm_respondent_calls': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/campus_coupons.json')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/campus-coupons-v1')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/campus-coupons-v1')
    args = parser.parse_args()
    cfg = prepare(args.config, args.out)
    with server(args.out) as url:
        asyncio.run(drive(args.out, url, cfg))
    from .campus_analysis import analyze
    result = analyze(args.out, args.report)
    print(json.dumps({'passed': result['integrity']['passed'], 'report': str(args.report)}))
    if not result['integrity']['passed']:
        raise SystemExit('Invalid or incomplete experiment; no effect claim')


if __name__ == '__main__':
    main()
