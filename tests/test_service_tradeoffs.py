from collections import Counter
from pathlib import Path
import runpy

import pytest


summarize = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/analyze_service_tradeoffs.py'))['summarize']


def fixture_rows():
    stages = dict(lookup=1, transform=6, inference=2, service=10, handler=12)
    return [
        dict(index=0, issued=True, status='200', latency_ms=50, send_lag_ms=0, timings_ms=stages),
        dict(index=1, issued=True, status='200', latency_ms=51, send_lag_ms=12, timings_ms=stages),
        dict(index=2, issued=True, status='503', latency_ms=2, send_lag_ms=0),
        dict(index=3, issued=False, status='client_capacity', send_lag_ms=11),
        dict(index=4, issued=True, status='timeout', latency_ms=5000, send_lag_ms=0),
    ]


def run_for(rows):
    return dict(scheduled=len(rows), successful=sum(r['status'] == '200' for r in rows),
                status_counts=dict(Counter(r['status'] for r in rows)),
                target_rps=50, latency_success_ms=dict(p95=50.95))


def test_deadline_success_counts_errors_and_unissued_requests_in_denominator():
    rows = fixture_rows()
    result = summarize(rows, run_for(rows), 50)
    assert result['on_time_success_rate'] == .2
    assert result['on_time_successful'] == 1  # Exactly 50ms meets the deadline.
    assert result['late_successful'] == 1
    assert result['failed'] == 3
    assert result['ideal_service_speedup_if_inference_free'] == 1.25


def test_incomplete_or_duplicate_trace_cannot_produce_success_claim():
    rows = fixture_rows()
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(rows[:-1], run_for(rows), 50)
    rows[-1]['index'] = 0
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows, run_for(rows), 50)


def test_status_summary_and_nonfinite_latency_are_rejected():
    rows = fixture_rows()
    run = run_for(rows)
    rows[2]['status'] = 'timeout'
    with pytest.raises(ValueError, match='status counts'):
        summarize(rows, run, 50)
    rows[0]['latency_ms'] = float('nan')
    with pytest.raises(ValueError, match='latency'):
        summarize(rows, run_for(rows), 50)
