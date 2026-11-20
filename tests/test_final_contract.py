import json
from pathlib import Path

import pandas as pd
import pytest

from fairjob_ranker.final_evaluation import verify_frozen
from fairjob_ranker.io import sha256, write_json
from fairjob_ranker.replay import export_requests, load_requests
from fairjob_ranker.schema import NUMERICAL, VERSION


def test_test_replay_requires_freeze_and_drops_labels(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    frame = pd.DataFrame({'impression_id': [1,1], 'product_id': [10,20], 'cat0': [0,0], 'cat1': [1,1],
                          **{c:[.2,.2] for c in NUMERICAL}, 'click': [0,1], 'rank': [2,1], 'protected_attribute': [0,0]})
    frame.to_parquet(data / 'test.parquet', index=False)
    write_json(data / 'manifest.json', {'version': VERSION,'files':{'test.parquet': {'sha256':sha256(data/'test.parquet')}}})
    output = tmp_path / 'requests.jsonl'
    with pytest.raises(ValueError, match='frozen'):
        export_requests(data,'test',output)
    assert not output.exists()
    frozen = tmp_path / 'frozen.json'
    write_json(frozen, {'dataset_manifest_sha256':sha256(data/'manifest.json')})
    export_requests(data,'test',output,frozen)
    requests, manifest = load_requests(output)
    assert requests[0]['candidate_ids'] == [10,20]
    assert set(requests[0]) == {'request_id','schema_version','context','candidate_ids','top_k'}
    assert set(requests[0]['context']) == {'cat0','cat1','numeric'}
    assert manifest['labels_included'] is False
    with pytest.raises(FileExistsError):
        export_requests(data,'test',output,frozen)


def test_changed_model_or_serving_settings_invalidates_freeze(tmp_path):
    weights, settings = tmp_path/'weights.pt',tmp_path/'settings.json'
    weights.write_bytes(b'original weights')
    settings.write_text('{}')
    data = tmp_path / 'data'
    data.mkdir()
    (data/'manifest.json').write_text('{}')
    frozen = tmp_path/'frozen.json'
    write_json(frozen, {'artifact_sha256':{str(weights):sha256(weights)},'serving_settings_path':str(settings),
                       'serving_settings_sha256':sha256(settings),'data_dir':str(data),
                       'dataset_manifest_sha256':sha256(data/'manifest.json')})
    verify_frozen(frozen)
    settings.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='settings'):
        verify_frozen(frozen)
    weights.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='artifact'):
        verify_frozen(frozen)
