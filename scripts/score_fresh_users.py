"""Freeze predictions on new large users WITHOUT reading their click labels."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from fairjob_ranker.ebnerd.metadata_audit import EventOnlyState
from fairjob_ranker.ebnerd.exposure_audit import ExposureState
from fairjob_ranker.ebnerd.feedback import FeedbackState
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-top1-feedback'; OUT=ROOT/'reports/ebnerd-top1-feedback'
BASE=ROOT/'data/processed/ebnerd-metadata-audit'; MODELS=ROOT/'models/ebnerd-top1-feedback'
release=json.loads((OUT/'frozen_release.json').read_text())
assert release['selection_sha256']==sha256(OUT/'selection.json')
for row in release['models']:
    assert row['model_sha256']==sha256(MODELS/row['name']/'model.txt')
if (OUT/'predictions_frozen.json').exists(): raise SystemExit('Fresh predictions already frozen')
selected=set(map(int,np.load(DATA/'fresh_user_ids.npy')))
columns=['impression_id','user_id','impression_time','device_type','article_ids_inview']
rows=[]
for batch in pq.ParquetFile(ROOT/'data/raw/ebnerd-large/validation/behaviors.parquet').iter_batches(batch_size=32768,columns=columns):
    frame=batch.to_pandas(); part=frame[frame.user_id.isin(selected)]
    if len(part):rows.append(part)
frame=pd.concat(rows,ignore_index=True).sort_values(['impression_time','impression_id'],kind='stable').reset_index(drop=True)
assert not frame.impression_id.duplicated().any() and not frame.impression_time.isna().any()
assert frame.article_ids_inview.map(lambda x:len(x)>0 and len(x)==len(set(x))).all()
frame.to_parquet(DATA/'fresh_requests.parquet',index=False)
history=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/validation/history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
state=EventOnlyState(history,'2023-05-25 07:00:00')
new_history=pd.read_parquet(DATA/'fresh_history.parquet')
new_state=EventOnlyState(new_history,'2023-05-25 07:00:00')
assert not (set(state.users)&set(new_state.users))
state.users.update(new_state.users)
del new_state,new_history,history
reference=pd.concat([pd.read_parquet(BASE/f'strict_{split}_requests.parquet') for split in ['train','test']],ignore_index=True)
reference=reference.sort_values(['impression_time','impression_id'],kind='stable').reset_index(drop=True)
assert not set(reference.user_id)&selected
ref_times=reference.impression_time.to_numpy(dtype='datetime64[s]').astype('int64')
times=frame.impression_time.to_numpy(dtype='datetime64[s]').astype('int64')
groups=frame.article_ids_inview.map(len).to_numpy(dtype='int32'); count=int(groups.sum())
items=np.concatenate(frame.article_ids_inview.to_numpy()).astype('int32')
for key,a in [('groups',groups),('items',items)]:np.save(DATA/f'fresh_{key}.npy',a)
states={d:(ExposureState(delay_seconds=d),FeedbackState(delay=d)) for d in [3600,86400]}
states['frozen']=(ExposureState(delay_seconds=3600),FeedbackState(delay=3600))
freeze_before=int(np.datetime64('2023-05-25 05:00:00','s').astype('int64'))
matrices={d:np.lib.format.open_memmap(DATA/f'fresh_d{d}_X.npy',mode='w+',dtype='float32',shape=(count,38)) for d in states}
ref_rows=iter(reference.itertuples(index=False)); next_ref=next(ref_rows,None)
offset=0
for j,(row,now) in enumerate(zip(frame.itertuples(index=False),times)):
    while next_ref is not None and int(np.datetime64(next_ref.impression_time,'s').astype('int64'))<now:
        r=next_ref; t=int(np.datetime64(r.impression_time,'s').astype('int64'))
        for mode,(e,f) in states.items():
            if mode=='frozen' and t>=freeze_before: continue
            e.advance(t);e.observe(t,r.user_id,r.article_ids_inview)
            f.advance(t);f.observe(t,r.article_ids_inview,r.article_ids_clicked)
        next_ref=next(ref_rows,None)
    ids=row.article_ids_inview; end=offset+len(ids)
    fixed=state.matrix(row.user_id,row.impression_time,row.device_type,ids)
    for d,(e,f) in states.items():
        matrices[d][offset:end]=np.column_stack([fixed,e.matrix(now,row.user_id,ids),f.matrix(now,ids)])
    offset=end
    if (j+1)%50000==0: print('Fresh features',j+1,'/',len(frame),flush=True)
for x in matrices.values():x.flush();assert np.isfinite(x).all()
cohort={'requests':len(frame),'users':int(frame.user_id.nunique()),'candidates':count,
    'start':str(frame.impression_time.min()),'end':str(frame.impression_time.max()),
    'fresh_labels_loaded':False,'reference_users_disjoint':True,'fixed_global_history':'small validation history only',
    'personal_history':'selected new users, strictly before 2023-05-25 07:00',
    'article_tables_loaded':False,'fresh_user_feedback_used':False}
(OUT/'fresh_feature_preparation.json').write_text(json.dumps(cohort,indent=2)+'\n')
selection=json.loads((OUT/'selection.json').read_text())['selected']
receipts=[]
for delay,x in matrices.items():
    arms=['exposure_ndcg','exposure_top1','feedback_ndcg','feedback_top1'] if delay==3600 else ['exposure_top1',selection]
    if delay=='frozen': arms=['exposure_ndcg','exposure_top1',selection]
    for arm in arms:
        size=20 if arm.startswith('exposure') else 38
        score=np.zeros(count)
        for seed in [42,43,44]:
            model=lgb.Booster(model_file=str(MODELS/f'final-{arm}-s{seed}'/'model.txt'))
            # Chunked inference avoids an additional full dense feature copy.
            for start in range(0,count,250000):
                end=min(count,start+250000)
                score[start:end]+=model.predict(np.array(x[start:end,:size]),num_threads=4)/3
        path=DATA/f'fresh_d{delay}_{arm}_scores.npy';np.save(path,score)
        receipts.append({'delay':delay,'model':arm,'sha256':sha256(path)})
        print('Scored',delay,arm,flush=True)
    for arm,index in [('ctr_1h',26),('ctr_24h',30)]:
        path=DATA/f'fresh_d{delay}_{arm}_scores.npy';np.save(path,np.array(x[:,index],dtype=float))
        receipts.append({'delay':delay,'model':arm,'sha256':sha256(path)})
(OUT/'predictions_frozen.json').write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),
    'release_sha256':sha256(OUT/'frozen_release.json'),'fresh_labels_loaded':False,
    'pipeline_source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [
        ROOT/'scripts/train_feedback.py',ROOT/'scripts/score_fresh_users.py',ROOT/'scripts/evaluate_feedback.py',
        ROOT/'src/fairjob_ranker/ebnerd/feedback.py',ROOT/'src/fairjob_ranker/ebnerd/exposure_audit.py',
        ROOT/'src/fairjob_ranker/ebnerd/metadata_audit.py']},
    'evaluation_clarifications_sha256':sha256(OUT/'evaluation_clarifications.json'),
    'requests_sha256':sha256(DATA/'fresh_requests.parquet'),'feature_sha256':{str(d):sha256(DATA/f'fresh_d{d}_X.npy') for d in states},
    'predictions':receipts},indent=2)+'\n')
print('PREDICTIONS FROZEN before target labels',flush=True)
