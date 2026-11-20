from copy import deepcopy
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from fairjob_ranker.log_contract import RequestSnapshot,ClosedFeedback


def example():
    # Synthetic schema example, never used as evidence of model improvement.
    return {'request_id':'synthetic-r1','user_key':'synthetic-u1','served_at':'2023-05-25T07:00:00Z',
            'model_version':'v1','policy_version':'p1','candidates':[{'item_id':'new-item',
            'features':{'price':{'value':10.,'version':'immutable-1','source_event_at':'2023-05-25T06:59:00Z',
            'computed_at':'2023-05-25T06:59:10Z','available_at':'2023-05-25T06:59:20Z'}}}]}


def test_rejects_future_features_and_target_in_request():
    data=example();data['candidates'][0]['features']['price']['available_at']='2023-05-25T07:00:01Z'
    with pytest.raises(ValidationError):RequestSnapshot.model_validate(data)
    data=example();data['clicked_item_ids']=['new-item']
    with pytest.raises(ValidationError):RequestSnapshot.model_validate(data)


def test_feedback_arrival_and_snapshot_are_enforced():
    request=RequestSnapshot.model_validate(example())
    feedback=ClosedFeedback.model_validate({'request_id':request.request_id,'request_snapshot_sha256':request.digest(),
        'attribution_window_closed_at':'2023-05-25T08:00:00Z','received_at':'2023-05-25T09:00:00Z','clicked_item_ids':['new-item']})
    feedback.validate_against(request)
    assert not feedback.usable_before(datetime(2023,5,25,9,tzinfo=timezone.utc))
    assert feedback.usable_before(datetime(2023,5,25,9,0,1,tzinfo=timezone.utc))
    changed=example();changed['candidates'][0]['features']['price']['value']=11.
    with pytest.raises(ValueError):feedback.validate_against(RequestSnapshot.model_validate(changed))


def test_no_naive_time_duplicate_candidate_or_unobserved_negative_fill():
    data=example();data['served_at']='2023-05-25T07:00:00'
    with pytest.raises(ValidationError):RequestSnapshot.model_validate(data)
    data=example();data['candidates'].append(deepcopy(data['candidates'][0]))
    with pytest.raises(ValidationError):RequestSnapshot.model_validate(data)
    r=RequestSnapshot.model_validate(example())
    f=ClosedFeedback.model_validate({'request_id':r.request_id,'request_snapshot_sha256':r.digest(),
      'attribution_window_closed_at':'2023-05-25T08:00:00Z','received_at':'2023-05-25T08:01:00Z','clicked_item_ids':['not-shown']})
    with pytest.raises(ValueError):f.validate_against(r)
