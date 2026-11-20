"""20-minute scheduled in-memory replay of the NEW article-free scoring path.

This is not an HTTP test. Historical event time advances independently of wall
clock; a 60s event-arrival assumption is not measured by this performance test.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from fairjob_ranker.ebnerd.exposure_audit import ExposureState
from fairjob_ranker.ebnerd.metadata_audit import EventOnlyState

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
RUN=ROOT/'runs/ebnerd-metadata-audit';RUN.mkdir(parents=True,exist_ok=True)
target=50;duration=1200;count=target*duration;deadline_ms=50
protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'target_rps':target,'duration_seconds':duration,
    'requests':count,'deadline_ms':deadline_ms,'model':'audit_holdout-exposure_personal-s42',
    'checks':['Every feature and score equals its offline cache.','Every scheduled request is processed exactly once in chronological event order.','Report scheduled-to-completion delay, misses, queue delay, and RSS samples.'],
    'scope':'Single-process in-memory scheduled replay; excludes HTTP, persistence and actual log delivery. Logical historical event time is independent of benchmark wall time. No model selection or registration.'}
path=OUT/'exposure_stress_protocol.json'
if path.exists():raise SystemExit('Stress protocol exists; do not silently rerun')
path.write_text(json.dumps(protocol,indent=2)+'\n')
train=pd.read_parquet(DATA/'strict_train_requests.parquet');test=pd.read_parquet(DATA/'strict_test_requests.parquet')
history=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/validation/history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
fixed=EventOnlyState(history,'2023-05-25 07:00:00');stream=ExposureState(60)
model=lgb.Booster(model_file=str(ROOT/'models/ebnerd-metadata-audit/audit_holdout-exposure_personal-s42/model.txt'))
groups=np.load(DATA/'strict_test_groups.npy');offsets=np.r_[0,np.cumsum(groups)];begin=len(test)-count
features=np.load(DATA/'exposure_test_X.npy',mmap_mode='r');scores=np.load(DATA/'audit_holdout-exposure_personal-s42_scores.npy')
for frame in [train,test.iloc[:begin]]:
    stamps=frame.impression_time.to_numpy(dtype='datetime64[s]').astype(np.int64)
    for row,now in zip(frame.itertuples(index=False),stamps):stream.advance(now);stream.observe(now,row.user_id,row.article_ids_inview)
latencies=np.empty(count);processing=np.empty(count);queue=np.empty(count);memory=[];max_feature=0.;max_score=0.
process=psutil.Process();start=time.perf_counter();print('START scheduled replay',datetime.now(timezone.utc).isoformat(),flush=True)
for n,row in enumerate(test.iloc[begin:].itertuples(index=False)):
    due=start+n/target;wait=due-time.perf_counter()
    if wait>0:time.sleep(wait)
    commenced=time.perf_counter();now=int(pd.Timestamp(row.impression_time).value//1_000_000_000)
    x=np.column_stack([fixed.matrix(row.user_id,row.impression_time,row.device_type,row.article_ids_inview),stream.matrix(now,row.user_id,row.article_ids_inview)])
    prediction=model.predict(x,num_threads=1);stream.observe(now,row.user_id,row.article_ids_inview)
    finished=time.perf_counter();latencies[n]=(finished-due)*1000;processing[n]=(finished-commenced)*1000;queue[n]=(commenced-due)*1000
    a,b=offsets[begin+n],offsets[begin+n+1]
    max_feature=max(max_feature,float(np.max(abs(x-features[a:b]))));max_score=max(max_score,float(np.max(abs(prediction-scores[a:b]))))
    if (n+1)%(target*60)==0:
        memory.append({'processed':n+1,'elapsed_seconds':finished-start,'rss_mb':process.memory_info().rss/1024**2})
        print('progress',n+1,'/',count,'errors',max_feature,max_score,'p95_ms',float(np.quantile(latencies[:n+1],.95)),flush=True)
elapsed=time.perf_counter()-start
np.save(RUN/'exposure_stress_latencies.npy',np.column_stack([latencies,processing,queue]))
def quantiles(values):return {'p50':float(np.quantile(values,.5)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99)),'max':float(values.max())}
result={**protocol,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'processed':count,'elapsed_seconds':elapsed,
    'scheduled_to_complete_ms':quantiles(latencies),'processing_ms':quantiles(processing),'scheduler_queue_ms':quantiles(queue),
    'deadline_misses':int((latencies>deadline_ms).sum()),'on_time_fraction':float((latencies<=deadline_ms).mean()),
    'feature_max_abs_error':max_feature,'score_max_abs_error':max_score,'rss_samples':memory,
    'exact_parity_passed':max_feature==0 and max_score==0}
(OUT/'exposure_stress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if not result['exact_parity_passed']:raise SystemExit('Feature/score mismatch')
