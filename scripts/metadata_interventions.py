"""Frozen-model missing-value interventions and a quarantined leakage positive control."""
from datetime import datetime,timezone
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.bundle import Bundle
from fairjob_ranker.ebnerd.features import FEATURES,META,PERSONAL,FRESH
from fairjob_ranker.ebnerd.metadata_audit import mask_columns
from fairjob_ranker.ebnerd.evaluation import outcomes,paired_interval,METRICS
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; base=ROOT/'data/processed/ebnerd-v1'; data=ROOT/'data/processed/ebnerd-metadata-audit'; out=ROOT/'reports/ebnerd-metadata-audit'
protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'parent_protocol_sha256':sha256(out/'protocol.json'),
    'frozen_model_masks':{'metadata_all_missing':META+PERSONAL,'category_missing':META+['category_affinity','recent_category_affinity'],
        'topics_missing':['topic_affinity','topic_max_affinity'],'entities_missing':['entity_affinity','entity_max_affinity'],'publication_missing':FRESH},
    'synthetic_leakage_positive_control':'Score a candidate by the SAME user\'s entire evaluation-week clicked-item count, intentionally including the target. Compare after permuting users. This is explicitly invalid and never a usable model.',
    'permuted_retraining':'Coherent per-article metadata reassignment within publication-date strata; fixed mapping seed 100042; 3 training seeds and original 2 development folds. No historical-snapshot interpretation.'}
path=out/'control_protocol.json'
if path.exists(): raise SystemExit('Control protocol already exists')
path.write_text(json.dumps(protocol,indent=2)+'\n')
bundle=Bundle(ROOT/'models/ebnerd-v1/final-personalized'); x=np.load(base/'test_X.npy',mmap_mode='r'); y=np.load(base/'test_y.npy'); ids=np.load(base/'test_items.npy'); groups=np.load(base/'test_groups.npy')
frame=pd.read_parquet(base/'validation_requests.parquet'); cold=set(map(int,ids))-bundle.seen
original=pd.read_parquet(base/'test-personalized_outcomes.parquet'); results={}
for name,cols in protocol['frozen_model_masks'].items():
    print('mask',name,flush=True); changed=mask_columns(x,cols); scores=bundle.predict(changed)
    metrics=outcomes(y,scores,groups,ids,cold); metrics.insert(0,'user_id',frame.user_id); metrics.insert(0,'impression_id',frame.impression_id)
    metrics.to_parquet(data/(name+'_outcomes.parquet'),index=False); np.save(data/(name+'_scores.npy'),scores)
    results[name]={'features':cols,'metrics':metrics[METRICS].mean().to_dict(),
        'minus_frozen_original':{m:paired_interval(frame.user_id,metrics[m]-original[m]) for m in ['ndcg5','hit1','cold_hit1']},
        'interpretation':'Out-of-distribution information removal, not an estimate of leakage contribution.'}
    del changed; print(name,results[name]['metrics'],flush=True)
(out/'frozen_interventions.json').write_text(json.dumps(results,indent=2)+'\n')
# Intentionally impossible-at-serving positive control; do not fit/register it.
future={}
for row in frame.itertuples(index=False): future.setdefault(int(row.user_id),Counter()).update(set(map(int,row.article_ids_clicked)))
users=np.array(sorted(future)); shuffled=np.random.default_rng(1042).permutation(users); mapping=dict(zip(map(int,users),map(int,shuffled)))
score=np.empty(len(y)); broken=np.empty(len(y)); offset=0
for row in frame.itertuples(index=False):
    candidate=list(map(int,row.article_ids_inview)); end=offset+len(candidate)
    score[offset:end]=[future[int(row.user_id)][i] for i in candidate]
    broken[offset:end]=[future[mapping[int(row.user_id)]][i] for i in candidate]; offset=end
invalid=outcomes(y,score,groups,ids,cold); invalid_shuffled=outcomes(y,broken,groups,ids,cold)
positive={'label':'DELIBERATELY LEAKING POSITIVE CONTROL, NOT A MODEL RESULT',
    'future_user_item_counts':invalid.mean().to_dict(),'permuted_future_user_item_counts':invalid_shuffled.mean().to_dict(),
    'aligned_minus_permuted':{m:paired_interval(frame.user_id,invalid[m]-invalid_shuffled[m]) for m in ['ndcg5','hit1']},
    'conclusion_scope':'A score can use future outcomes and still deteriorate when user associations are permuted. User-shuffle tests alone cannot certify no leakage.'}
(out/'deliberately_invalid_positive_control.json').write_text(json.dumps(positive,indent=2)+'\n')
print(json.dumps(positive,indent=2))
