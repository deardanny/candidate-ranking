import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from fairjob_ranker.ebnerd.audit import candidate_audit, history_audit, partition, past_events


def test_history_uses_only_strict_past_and_sorts_without_losing_alignment():
    ids, times = past_events([30, 10, 20, 40],
        ['2023-05-23', '2023-05-21', '2023-05-22', '2023-05-24'], '2023-05-23')
    assert ids.tolist() == [10,20]
    assert (times < np.datetime64('2023-05-23')).all()
    # Changing the current/future events cannot change this user's past features.
    changed, _ = past_events([999,10,20,888],
        ['2023-05-23','2023-05-21','2023-05-22','2023-05-24'], '2023-05-23')
    np.testing.assert_array_equal(ids,changed)
    with pytest.raises(ValueError,match='Unaligned'):
        past_events([1,2],['2023-05-20'],'2023-05-23')


def test_temporal_boundaries_do_not_depend_on_input_order():
    times = pd.Series(pd.to_datetime(['2023-05-25 07:00:00','2023-05-23 06:59:59',
                                     '2023-05-23 07:00:00','2023-05-25 06:59:59']))
    assert partition(times).tolist() == ['test','train','validation','validation']
    with pytest.raises(ValueError):
        partition(pd.Series([pd.NaT]))


def test_cold_support_does_not_credit_known_clicks_and_masks_future_article_snapshots():
    b = pd.DataFrame({'impression_id':[1,2,3], 'user_id':[7,8,9],
        'impression_time':pd.to_datetime(['2023-05-26']*3),
        'article_ids_inview':[[10,20],[10,20],[20,30]], 'article_ids_clicked':[[10],[20],[30]]})
    a = pd.DataFrame({'article_id':[10,20,30],
        'published_time':pd.to_datetime(['2023-05-20','2023-05-25','2023-05-25']),
        'last_modified_time':pd.to_datetime(['2023-05-20','2023-05-27','2023-05-25'])})
    r = candidate_audit(b,a,{10},{10},pd.Timestamp('2023-05-25'),{10,20})
    c = r['cold_cohorts']['unseen_training_inputs']
    assert c['requests'] == 3 and c['clicked_requests'] == 2
    assert c['clicked_candidate_rows'] == 2
    assert c['all_candidates_cold_requests'] == 1
    assert c['snapshot_available_clicked_candidates'] == 1
    assert r['cold_cohorts']['unseen_inputs_and_history']['items'] == 1
    assert r['modified_after_request_candidates'] == 3


def test_history_audit_flags_future_equal_time_and_missing_user_history():
    h = pd.DataFrame({'user_id':[1], 'article_id_fixed':[[1,2,3]],
        'impression_time_fixed':[np.array(['2023-05-23','2023-05-22','2023-05-24'],dtype='datetime64[us]')],
        'read_time_fixed':[[1.,2.,3.]],'scroll_percentage_fixed':[[1.,2.,3.]]})
    b = pd.DataFrame({'user_id':[1,2], 'impression_time':pd.to_datetime(['2023-05-23']*2)})
    r=history_audit(h,b,{1,2,3})
    assert r['not_strictly_before_first_request'] == 2
    assert r['unsorted_rows'] == 1
    assert r['request_users_without_history'] == 1


def test_archive_extraction_rejects_traversal_and_differing_existing_files(tmp_path):
    spec=importlib.util.spec_from_file_location('download_ebnerd',Path(__file__).parents[1]/'scripts/download_ebnerd.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z: z.writestr('../escape','no')
    with pytest.raises(ValueError,match='Unsafe'): mod.extract(archive,tmp_path/'out')
    assert not (tmp_path/'escape').exists()
    with zipfile.ZipFile(archive,'w') as z: z.writestr('data','original')
    mod.extract(archive,tmp_path/'out')
    (tmp_path/'out/data').write_text('modified')
    with pytest.raises(ValueError,match='differs'): mod.extract(archive,tmp_path/'out')
