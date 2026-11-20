"""Extract official large validation files; select users from history IDs, not labels.

No validation behavior labels are loaded until evaluate_feedback.py is run after
the model selection and release hashes have been frozen.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/ebnerd-large'; DATA=ROOT/'data/processed/ebnerd-top1-feedback'; OUT=ROOT/'reports/ebnerd-top1-feedback'
if (OUT/'fresh_user_cohort.json').exists(): raise SystemExit('Cohort already sealed')
archive=RAW/'ebnerd_large.zip'
files=[]
with zipfile.ZipFile(archive) as z:
    print([(i.filename,i.file_size) for i in z.infolist()],flush=True)
    for member in z.infolist():
        name=PurePosixPath(member.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in member.filename: raise ValueError('Unsafe archive')
        if len(name.parts)<2 or name.parts[-2:] not in [('validation','history.parquet'),('validation','behaviors.parquet')]: continue
        target=RAW/'validation'/name.name; target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with z.open(member) as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
        files.append({'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha256(target)})
        print('Extracted',name.name,flush=True)
excluded=set()
for split in ['train','validation']:
    for name in ['history','behaviors']:
        p=ROOT/f'data/raw/ebnerd-small/extracted/{split}/{name}.parquet'
        excluded.update(map(int,pd.read_parquet(p,columns=['user_id']).user_id))
history_path=RAW/'validation/history.parquet'
user_ids=pq.read_table(history_path,columns=['user_id']).column('user_id').to_pylist()
selected={int(u) for u in user_ids if int(u) not in excluded and int.from_bytes(hashlib.sha256(f'ebnerd-top1-feedback-v1:{u}'.encode()).digest(),'big')%20==0}
rows=[]
for batch in pq.ParquetFile(history_path).iter_batches(batch_size=4096,columns=['user_id','article_id_fixed','impression_time_fixed']):
    frame=batch.to_pandas(); chosen=frame[frame.user_id.isin(selected)]
    if len(chosen):rows.append(chosen)
history=pd.concat(rows,ignore_index=True); history.to_parquet(DATA/'fresh_history.parquet',index=False)
np.save(DATA/'fresh_user_ids.npy',np.array(sorted(selected),dtype=np.int64))
receipt={'sealed_utc':datetime.now(timezone.utc).isoformat(),'source_url':'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip',
    'archive_sha256':sha256(archive),'files':files,'excluded_small_users':len(excluded),'eligible_large_history_users':len(user_ids),
    'selected_users':len(selected),'overlap_with_excluded':len(selected&excluded),'selected_user_hash':sha256(DATA/'fresh_user_ids.npy'),
    'selected_history_hash':sha256(DATA/'fresh_history.parquet'),'selection_uses_labels':False,
    'note':'Archive CRC checked while reading extracted members. No claim of publisher-supplied checksum. Same calendar week, fresh users only.'}
(OUT/'fresh_user_cohort.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
