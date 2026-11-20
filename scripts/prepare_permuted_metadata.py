"""Coherent fake metadata for dependency tests; no claim of historical reconstruction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.features import FeatureState,ARTICLE_COLUMNS,FEATURES,BEHAVIOR,USER,FRESH,build_matrix
from fairjob_ranker.ebnerd.metadata_audit import permute_metadata
from fairjob_ranker.ebnerd.bundle import Bundle
from fairjob_ranker.ebnerd.evaluation import outcomes,METRICS,paired_interval
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; raw=ROOT/'data/raw/ebnerd-small/extracted'; base=ROOT/'data/processed/ebnerd-v1'; out=ROOT/'reports/ebnerd-metadata-audit'; data=ROOT/'data/processed/ebnerd-metadata-audit'
if not (out/'control_protocol.json').exists(): raise SystemExit('Missing fixed control protocol')
articles=pd.read_parquet(raw/'articles.parquet',columns=ARTICLE_COLUMNS); permuted,info=permute_metadata(articles,100042)
np.save(data/'article_permutation.npy',info.pop('mapping'))
fixed=[FEATURES.index(c) for c in BEHAVIOR+USER+FRESH]; checks={}
for source,name,start in [('train','train','2023-05-18 07:00:00'),('validation','test','2023-05-25 07:00:00')]:
    frame=pd.read_parquet(base/('train_requests.parquet' if name=='train' else 'validation_requests.parquet'))
    history=pd.read_parquet(raw/source/'history.parquet',columns=['user_id','article_id_fixed','impression_time_fixed'])
    state=FeatureState(permuted,history,start); x,y,ids,groups=build_matrix(frame,state,lambda s:print(name,s,flush=True))
    np.save(data/f'permuted_{name}_X.npy',x)
    original=np.load(base/f'{name}_X.npy',mmap_mode='r')
    checks[name+'_all_behavior_user_publication_features_identical']=bool(np.array_equal(x[:,fixed],original[:,fixed]))
    if name=='test':
        bundle=Bundle(ROOT/'models/ebnerd-v1/final-personalized'); scores=bundle.predict(x); np.save(data/'coherent_permutation_frozen_scores.npy',scores)
        result=outcomes(y,scores,groups,ids,set(map(int,ids))-bundle.seen)
        baseline=pd.read_parquet(base/'test-personalized_outcomes.parquet')
        result.insert(0,'user_id',frame.user_id); result.insert(0,'impression_id',frame.impression_id)
        result.to_parquet(data/'coherent_permutation_frozen_outcomes.parquet',index=False)
        info['frozen_model_metrics']=result[METRICS].mean().to_dict()
        info['minus_original']={m:paired_interval(frame.user_id,result[m]-baseline[m]) for m in ['ndcg5','hit1','cold_hit1']}
info.update(checks=checks,control_protocol_sha256=sha256(out/'control_protocol.json'))
(out/'coherent_permutation.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
