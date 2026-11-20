import asyncio
import json

import httpx
import pytest

from fairjob_ranker.loadtest import percentiles, run_phase
from fairjob_ranker.replay import load_requests
from fairjob_ranker.io import sha256, write_json


def test_open_loop_counts_client_capacity_and_queue_delay():
    async def scenario():
        async def handler(request):
            await asyncio.sleep(0.06)
            body = json.loads(request.content)
            return httpx.Response(200, json={"request_id": body['request_id'], "rankings": [{"product_id": 1}],
                                            "degraded": False, "timings_ms": {"inference": 1}, "scoring_method": "pytorch"})
        r = await run_phase("http://test", [{"request_id": "x", "candidate_ids": [1]}], rate=100,
                            duration=0.1, max_inflight=1, transport=httpx.MockTransport(handler))
        assert r['scheduled'] == 10
        assert r['issued'] + r['generator_dropped'] == 10
        assert r['successful'] == r['issued']
        assert r['generator_dropped'] > 0
        assert r['failure_rate'] == r['generator_dropped'] / 10
        assert r['latency_success_ms']['p50'] >= 60
        assert r['elapsed_including_drain_seconds'] >= .1
        assert len(r['records']) == 10
    asyncio.run(scenario())


def test_http_errors_and_timeouts_are_not_successful_qps():
    async def scenario():
        count = 0
        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(503)
            raise httpx.ReadTimeout('test timeout')
        r = await run_phase("http://test", [{"request_id": "x", "candidate_ids": [1]}], 20, .1,
                            transport=httpx.MockTransport(handler))
        assert r['status_counts'] == {'503': 1, 'timeout': 1}
        assert r['successful_qps'] == 0 and r['failure_rate'] == 1
        assert r['latency_success_ms']['p95'] is None
    asyncio.run(scenario())


def test_replay_integrity_check(tmp_path):
    path = tmp_path / 'requests.jsonl'
    path.write_text('{"request_id":"x"}\n')
    write_json(path.with_suffix('.manifest.json'), {'requests': 1, 'request_file_sha256': sha256(path)})
    assert len(load_requests(path)[0]) == 1
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        load_requests(path)


def test_empty_percentiles_are_explicit():
    assert percentiles([])['p95'] is None
