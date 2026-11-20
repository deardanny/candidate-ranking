"""Build article-free raw-cohort features; never open articles.parquet here."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.metadata_audit import EventOnlyState,structurally_valid,build_event_matrix,STRICT_FEATURES
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; raw=ROOT/'data/raw/ebnerd-small/extracted'; out=ROOT/'data/processed/ebnerd-metadata-audit'
out.mkdir(parents=True,exist_ok=True)
if (out/'strict_manifest.json').exists(): raise SystemExit('Strict preparation already completed')
details={}
for name,start in [('train','2023-05-18 07:00:00'),('test','2023-05-25 07:00:00')]:
    source='train' if name=='train' else 'validation'
    frame=structurally_valid(pd.read_parquet(raw/source/'behaviors.parquet',columns=['impression_id','user_id','impression_time','device_type','article_ids_inview','article_ids_clicked']))
    history=pd.read_parquet(raw/source/'history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
    state=EventOnlyState(history,start); x,y,items,groups=build_event_matrix(frame,state)
    frame.to_parquet(out/f'strict_{name}_requests.parquet',index=False)
    for key,array in [('X',x),('y',y),('items',items),('groups',groups)]: np.save(out/f'strict_{name}_{key}.npy',array)
    details[name]={'requests':len(frame),'candidates':len(x),'users':int(frame.user_id.nunique()),'history_events':int(sum(map(len,history.article_id_fixed)))}
    if name=='train': (out/'strict_history_ids.json').write_text(json.dumps(sorted(state.pop))+'\n')
(out/'strict_manifest.json').write_text(json.dumps({'details':details,'features':STRICT_FEATURES,'article_table_read':False,
    'publication_filter_applied':False,'source_sha256':sha256(ROOT/'src/fairjob_ranker/ebnerd/metadata_audit.py')},indent=2)+'\n')
print(json.dumps(details,indent=2))
