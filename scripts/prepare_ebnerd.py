"""Build clean train features and serving snapshots; keep holdout features separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.features import FeatureState,ARTICLE_COLUMNS,FEATURES,clean_behaviors,build_matrix
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]
raw=ROOT/'data/raw/ebnerd-small/extracted'; out=ROOT/'data/processed/ebnerd-v1'
out.mkdir(parents=True,exist_ok=False)
articles=pd.read_parquet(raw/'articles.parquet',columns=ARTICLE_COLUMNS)
counts={}
for source,start in [('train','2023-05-18 07:00:00'),('validation','2023-05-25 07:00:00')]:
    frame,rejected=clean_behaviors(raw/source/'behaviors.parquet',articles)
    frame.to_parquet(out/f'{source}_requests.parquet',index=False)
    rejected.to_parquet(out/f'{source}_quarantine.parquet',index=False)
    history=pd.read_parquet(raw/source/'history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
    FeatureState.save(out/f'{source}_state',articles,history,start)
    counts[source]={'requests':len(frame),'quarantined':len(rejected),'users':int(frame.user_id.nunique())}
    if source=='train':
        state=FeatureState.load(out/'train_state')
        x,y,items,groups=build_matrix(frame,state,lambda s:print(s,flush=True))
        for name,values in [('X',x),('y',y),('items',items),('groups',groups)]: np.save(out/f'train_{name}.npy',values)
        print('train shape',x.shape,flush=True)
manifest={'counts':counts,'features':FEATURES,'original_source_manifest_sha256':sha256(ROOT/'data/source/ebnerd/manifest.json'),
    'protocol_sha256':sha256(ROOT/'configs/ebnerd_experiment.json'),
    'feature_source_sha256':sha256(ROOT/'src/fairjob_ranker/ebnerd/features.py'),
    'files':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(counts,indent=2))
