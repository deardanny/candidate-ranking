"""Recompute every delayed exposure feature using independent sorted-array joins."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES
from fairjob_ranker.ebnerd.exposure_audit import EXPOSURE_FEATURES

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--delay-seconds',type=int,choices=[60,3600,86400],default=60)
args=parser.parse_args();delay=args.delay_seconds
prefix='exposure' if delay==60 else f'exposure_d{delay}'
frames=[];arrays=[]
for split in ['train','test']:
    frame=pd.read_parquet(DATA/f'strict_{split}_requests.parquet',columns=['user_id','impression_time','article_ids_inview'])
    groups=np.load(DATA/f'strict_{split}_groups.npy');items=np.load(DATA/f'strict_{split}_items.npy')
    assert np.array_equal(np.concatenate(frame.article_ids_inview.to_numpy()),items)
    times=frame.impression_time.to_numpy(dtype='datetime64[s]').astype(np.int64)
    frames.append(pd.DataFrame({'item':items,'user':np.repeat(frame.user_id.to_numpy(),groups),'time':np.repeat(times,groups)}))
    full=np.load(DATA/f'{prefix}_{split}_X.npy',mmap_mode='r');base=np.load(DATA/f'strict_{split}_X.npy',mmap_mode='r')
    assert np.array_equal(full[:,:len(STRICT_FEATURES)],base)
    arrays.append(full[:,len(STRICT_FEATURES):])
events=pd.concat(frames,ignore_index=True);expected=np.empty((len(events),len(EXPOSURE_FEATURES)),dtype=np.float32)
t=events.time.to_numpy()
for indices in events.groupby('item',sort=False).indices.values():
    stamps=t[indices];assert np.all(np.diff(stamps)>=0)
    upper=np.searchsorted(stamps,stamps-delay,side='left')
    hour=np.searchsorted(stamps,stamps-3600,side='left');day=np.searchsorted(stamps,stamps-86400,side='left')
    known=upper>0;first_age=np.zeros(len(indices));last_age=np.zeros(len(indices))
    first_age[known]=np.log1p((stamps[known]-stamps[0])/3600.)
    last_age[known]=np.log1p((stamps[known]-stamps[upper[known]-1])/3600.)
    expected[indices,:6]=np.column_stack([np.log1p(upper),np.log1p(np.maximum(0,upper-hour)),
        np.log1p(np.maximum(0,upper-day)),first_age,last_age,~known])
print('All global candidate features recomputed',len(events),flush=True)
for indices in events.groupby(['user','item'],sort=False).indices.values():
    stamps=t[indices]
    expected[indices,6]=np.log1p(np.searchsorted(stamps,stamps-delay,side='left'))
actual=np.concatenate(arrays);error=np.max(np.abs(actual-expected),axis=0)
assert np.array_equal(actual,expected),dict(zip(EXPOSURE_FEATURES,error.tolist()))
result={'candidate_rows':len(events),'all_features_exactly_equal':True,'max_absolute_error_by_feature':dict(zip(EXPOSURE_FEATURES,error.tolist())),
    'method':'Independent per-item/per-user-item sorted timestamp arrays and searchsorted; compared every row, not queue-state code.',
    'label_columns_read':False,'article_table_read':False,'availability_delay_seconds':delay,
    'availability_boundary':f'searchsorted(side=left) excludes exposure_timestamp+{delay} >= request_timestamp'}
(OUT/f'{prefix}_feature_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
