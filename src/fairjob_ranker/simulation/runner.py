"""Create the synthetic fixture, serve through real loopback HTTP, collect all visits."""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
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
from .policies import Policies, train
from .world import catalog, canonical, choose, features, people, visit

ROOT = Path(__file__).resolve().parents[3]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def prepare(config_path, out):
    config = json.loads(Path(config_path).read_text())
    if (config['evaluation_users'] != 3000 or config['visits_per_user'] != 8 or
            config['candidates_per_request'] != 8 or config['display_size'] != 3):
        raise ValueError('This frozen study is designed for 3000 users, eight visits, eight candidates, three displays')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'configuration.json', config)
    write_json(out/'catalog.json', catalog(config['seed']))
    with (out/'personas.jsonl').open('w') as stream:
        for person in people(config['seed'], config['evaluation_users'], 'eval'):
            stream.write(canonical(person) + '\n')
    sources = list((ROOT/'src/fairjob_ranker/simulation').glob('*.py')) + [ROOT/'src/fairjob_ranker/log_contract.py']
    write_json(out/'design_frozen.json', {'frozen_utc': utcnow(), 'configuration': config,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'evaluation_outcomes_generated': False})
    print('Frozen design; generating separate training/validation cohorts', flush=True)
    train(config, json.loads((out/'catalog.json').read_text()), out/'models')
    write_json(out/'protocol.json', {
        'frozen_utc': utcnow(), 'configuration': config,
        'design_sha256': sha256(out/'design_frozen.json'), 'evaluation_outcomes_generated': False,
        'artifact_sha256': {str(p.relative_to(out)): sha256(p) for p in
            [out/'catalog.json', out/'personas.jsonl', out/'configuration.json',
             out/'models/model.txt', out/'models/popularity.json', out/'models/metadata.json']},
        'dependencies': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'lightgbm', 'fastapi', 'uvicorn', 'httpx', 'scipy', 'pydantic']},
        'response_source': config['response_source'], 'human_participants': 0,
        'real_clicks': 0, 'llm_respondent_calls': 0,
        'served_model_family': 'new synthetic-world models only',
        'timestamp_semantics': 'Actual server receipt/feature computation/serve/feedback receipt clocks; simulated visit index is a separate field.'})
    return config


@contextmanager
def server(out):
    # Pass a bound loopback socket to our child: no port race and no existing service is stopped.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0)); listener.listen(128)
    port = listener.getsockname()[1]
    env = os.environ.copy()
    env['SYNTHETIC_AB_RUN'] = str(out.resolve())
    env['OMP_NUM_THREADS'] = '1'
    process = None
    with (out/'server.log').open('w') as log:
        try:
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn',
                'fairjob_ranker.simulation.service:app_factory', '--factory',
                '--fd', str(listener.fileno()), '--no-access-log', '--log-level', 'warning'],
                env=env, pass_fds=(listener.fileno(),), stdout=log, stderr=subprocess.STDOUT)
            listener.close()
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 30
            with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'Server exited; see {out / "server.log"}')
                    try:
                        response = client.get('/readyz')
                        if response.status_code == 200 and response.json()['ready']:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('API startup timed out')
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


async def drive(out, url, config):
    persons = [json.loads(line) for line in (out/'personas.jsonl').read_text().splitlines()]
    offers = json.loads((out/'catalog.json').read_text())
    by_id = {a['item_id']: a for a in offers}
    policy_check = Policies(out/'models')
    receipts = []
    gate = asyncio.Semaphore(config['concurrency'])
    parity_checks = 0
    started = time.perf_counter()
    with (out/'client.jsonl').open('w', buffering=1) as stream:
        async with httpx.AsyncClient(base_url=url, timeout=30, trust_env=False,
                                     limits=httpx.Limits(max_connections=config['concurrency'])) as client:
            for experiment in config['experiments']:
                completed = 0

                async def one_user(person):
                    nonlocal completed, parity_checks
                    async with gate:
                        seen = {}
                        for index in range(config['visits_per_user']):
                            context, ids = visit(config['seed'], person, index, offers)
                            request = {'experiment_id': experiment['id'],
                                'request_id': f"{experiment['id']}:{person['user_id']}:{index}",
                                'user_id': person['user_id'], 'visit_index': index,
                                'context': context, 'candidate_ids': ids}
                            attempt = {'experiment_id': experiment['id'], 'user_id': person['user_id'],
                                'visit_index': index, 'request': request, 'attempted_at': utcnow(),
                                'rank_ok': False, 'feedback_ok': False, 'synthetic': True}
                            begin = time.perf_counter()
                            try:
                                result = await client.post('/rank', json=request)
                                attempt['rank_latency_ms'] = (time.perf_counter() - begin) * 1000
                                attempt['rank_http_status'] = result.status_code
                                result.raise_for_status()
                                ranked = result.json(); attempt['rank_ok'] = True
                                shown = [by_id[r['item_id']] for r in ranked['rankings']]
                                choice = choose(config['seed'], person, index, context, shown, seen, experiment['response'])
                                for ad in shown:
                                    seen[ad['item_id']] = seen.get(ad['item_id'], 0) + 1
                                # The simulated visit closes here. No missing event is silently treated as no-click.
                                event = {'request_id': request['request_id'],
                                    'request_snapshot_sha256': ranked['request_snapshot_sha256'],
                                    'display_sha256': ranked['display_sha256'],
                                    'clicked_item_ids': [] if choice is None else [choice],
                                    'attribution_window_closed_at': utcnow(), 'synthetic': True}
                                ack = await client.post('/events/feedback', json=event)
                                attempt['feedback_http_status'] = ack.status_code
                                ack.raise_for_status()
                                attempt['feedback_ok'] = ack.json()['accepted']
                                attempt['response'] = ranked
                                attempt['chosen_item_id'] = choice
                                attempt['end_to_end_ms'] = (time.perf_counter() - begin) * 1000
                                if int(person['user_id'].split('-')[1]) < 5:
                                    scores = policy_check.score(ranked['policy'], [features(context, by_id[i]) for i in ids], ids)
                                    expected = sorted(zip(ids, scores), key=lambda x: (-float(x[1]), x[0]))[:3]
                                    assert [a for a, _ in expected] == [r['item_id'] for r in ranked['rankings']]
                                    np.testing.assert_allclose([s for _, s in expected], [r['score'] for r in ranked['rankings']], atol=1e-12)
                                    parity_checks += 1
                                if index == 0 and int(person['user_id'].split('-')[1]) % 100 == 0:
                                    retry = await client.post('/rank', json=request)
                                    retry.raise_for_status()
                                    assert retry.json() == ranked
                                    again = await client.post('/events/feedback', json=event)
                                    again.raise_for_status()
                                    assert again.json()['duplicate'] is True
                                    attempt['idempotent_retry_checked'] = True
                            except (httpx.HTTPError, ValueError, AssertionError) as error:
                                attempt['error'] = f'{type(error).__name__}: {error}'
                            finally:
                                stream.write(canonical(attempt) + '\n')
                        completed += 1
                        if completed % 500 == 0:
                            print(f"{experiment['id']}: {completed}/{len(persons)} simulated users", flush=True)

                await asyncio.gather(*(one_user(p) for p in persons))
                receipts.append({'experiment_id': experiment['id'], 'completed_users': completed})
            metrics = await client.get('/metrics'); metrics.raise_for_status()
            (out/'metrics.prom').write_text(metrics.text)
    write_json(out/'execution.json', {'completed_utc': utcnow(), 'experiments': receipts,
        'transport': 'actual loopback HTTP to a separate uvicorn process',
        'elapsed_seconds': time.perf_counter() - started, 'offline_http_parity_requests': parity_checks,
        'concurrency': config['concurrency'], 'server_shutdown_after_run': True,
        'measurement_scope': 'Local shared-host closed-loop simulator, not a capacity or sustained-RPS benchmark',
        'protocol_sha256': sha256(out/'protocol.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/simulation_ab.json')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/synthetic-ab-v1')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/synthetic-ab-v1')
    args = parser.parse_args()
    config = prepare(args.config, args.out)
    with server(args.out) as url:
        print('HTTP serving ready; starting fixed-horizon experiments', flush=True)
        asyncio.run(drive(args.out, url, config))
    from .analysis import analyze
    result = analyze(args.out, args.report)
    print(json.dumps({'complete': result['integrity']['passed'], 'report': str(args.report)}, indent=2))
    if not result['integrity']['passed']:
        raise SystemExit('Integrity checks failed; see report, do not claim an effect')


if __name__ == '__main__':
    main()
