"""Audit-only in-memory scoring replay; no HTTP service or registry changes.

Rebuild causal state from raw exposures, then reproduce the last 2,000 requests'
features and scores. Timings exclude warmup, network, persistence and model load.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.exposure_audit import ExposureState
from fairjob_ranker.ebnerd.metadata_audit import EventOnlyState

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
train=pd.read_parquet(DATA/'strict_train_requests.parquet');test=pd.read_parquet(DATA/'strict_test_requests.parquet')
history=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/validation/history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
fixed=EventOnlyState(history,'2023-05-25 07:00:00');stream=ExposureState(60)
model=lgb.Booster(model_file=str(ROOT/'models/ebnerd-metadata-audit/audit_holdout-exposure_personal-s42/model.txt'))
groups=np.load(DATA/'strict_test_groups.npy');offsets=np.r_[0,np.cumsum(groups)]
features=np.load(DATA/'exposure_test_X.npy',mmap_mode='r');scores=np.load(DATA/'audit_holdout-exposure_personal-s42_scores.npy')
warm_start=time.perf_counter();count=len(test)-2000
for frame in [train,test.iloc[:count]]:
    stamps=frame.impression_time.to_numpy(dtype='datetime64[s]').astype(np.int64)
    for row,now in zip(frame.itertuples(index=False),stamps):
        stream.advance(now);stream.observe(now,row.user_id,row.article_ids_inview)
warm_seconds=time.perf_counter()-warm_start
durations=[];feature_error=0.;score_error=0.;new_items=set();seen=set(map(int,np.load(DATA/'strict_train_items.npy')))
for j,row in enumerate(test.iloc[count:].itertuples(index=False),start=count):
    start=time.perf_counter();now=int(pd.Timestamp(row.impression_time).value//1_000_000_000)
    x=np.column_stack([fixed.matrix(row.user_id,row.impression_time,row.device_type,row.article_ids_inview),stream.matrix(now,row.user_id,row.article_ids_inview)])
    prediction=model.predict(x,num_threads=1)
    stream.observe(now,row.user_id,row.article_ids_inview)
    durations.append((time.perf_counter()-start)*1000)
    a,b=offsets[j],offsets[j+1];feature_error=max(feature_error,float(np.max(abs(x-features[a:b]))))
    score_error=max(score_error,float(np.max(abs(prediction-scores[a:b]))));new_items.update(set(map(int,row.article_ids_inview))-seen)
assert feature_error==0. and score_error==0.
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'requests_scored':len(durations),
    'article_table_read':False,'model_seed':42,'model_new_item_ids_retained':len(new_items),
    'warmup_requests':len(train)+count,'warmup_seconds':warm_seconds,
    'feature_max_abs_error':feature_error,'score_max_abs_error':score_error,
    'in_memory_feature_score_observe_ms':{'p50':float(np.quantile(durations,.5)),'p95':float(np.quantile(durations,.95)),'p99':float(np.quantile(durations,.99)),'max':max(durations)},
    'scope':'Sequential historical in-memory replay, sampled during concurrent research workloads. Excludes network, persistence, startup, actual stream ingestion. Not an HTTP SLO, load test, or live deployment. Registry unchanged.'}
(OUT/'exposure_scoring_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
