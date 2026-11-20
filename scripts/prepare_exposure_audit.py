"""Build a delayed, prior-exposure replay without reading articles or clicks."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.exposure_audit import ExposureState,EXPOSURE_FEATURES
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--delay-seconds',type=int,choices=[60,3600,86400],default=60)
args=parser.parse_args(); delay=args.delay_seconds
prefix='exposure' if delay==60 else f'exposure_d{delay}'
path=OUT/f'{prefix}_protocol.json'
protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'reason':'Explore a metadata-independent recency signal after auditing snapshot dependence. Does not explain how much original improvement was leakage.',
    'evaluation':'Post-hoc exposed holdout; no selection, registry changes, or claim of new confirmation.',
    'availability_delay_seconds':delay,'strict_inequality':f'exposure_timestamp + {delay} < request_timestamp',
    'information':'Only candidate exposure IDs and timestamps of preceding logged requests; never prior/current/future clicked IDs, article tables, or publication times.',
    'caveats':['Assumed ingestion latency, not observed arrival timestamps.','Exposure counts reflect sampled active users and logging policy, not unbiased popularity.','Raw candidate sets retained; fixed historical-click features use original strictly preceding snapshots.'],
    'arms':['exposure_global','exposure_personal'],'seeds':[42,43,44],'folds':['early','late','audit_holdout'],
    'feature_code_sha256':sha256(ROOT/'src/fairjob_ranker/ebnerd/exposure_audit.py')}
if path.exists():raise SystemExit('Exposure protocol already exists')
path.write_text(json.dumps(protocol,indent=2)+'\n')
state=ExposureState(delay_seconds=delay);details={}
for split in ['train','test']:
    frame=pd.read_parquet(DATA/f'strict_{split}_requests.parquet',columns=['user_id','impression_time','article_ids_inview'])
    base=np.load(DATA/f'strict_{split}_X.npy',mmap_mode='r'); extra=np.empty((len(base),len(EXPOSURE_FEATURES)),dtype=np.float32)
    times=frame.impression_time.to_numpy(dtype='datetime64[s]').astype(np.int64);offset=0
    for j,(row,now) in enumerate(zip(frame.itertuples(index=False),times)):
        end=offset+len(row.article_ids_inview)
        extra[offset:end]=state.matrix(now,row.user_id,row.article_ids_inview)
        state.observe(now,row.user_id,row.article_ids_inview);offset=end
        if (j+1)%100000==0:print('prior exposures',split,j+1,flush=True)
    assert np.isfinite(extra).all() and offset==len(base)
    np.save(DATA/f'{prefix}_{split}_X.npy',np.column_stack([base,extra]))
    details[split]={'requests':len(frame),'candidate_rows':offset,'first_timestamp':str(frame.impression_time.min()),'last_timestamp':str(frame.impression_time.max())}
    del extra
(OUT/f'{prefix}_preparation.json').write_text(json.dumps({'features':STRICT_FEATURES+EXPOSURE_FEATURES,'splits':details,'protocol_sha256':sha256(path)},indent=2)+'\n')
print('Exposure replay prepared',flush=True)
