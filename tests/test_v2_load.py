import asyncio
import httpx

from fairjob_ranker.v2.load import arrival_offset,run_load


def test_bursts_preserve_scheduled_volume_but_change_arrivals():
    assert [arrival_offset(i,100,1) for i in range(6)] == [0,.01,.02,.03,.04,.05]
    assert [arrival_offset(i,100,5) for i in range(6)] == [0,0,0,0,0,.05]


def test_invalid_load_parameters_fail_before_network():
    import pytest
    with pytest.raises(ValueError):
        asyncio.run(run_load('http://127.0.0.1:1',[],50,10))
