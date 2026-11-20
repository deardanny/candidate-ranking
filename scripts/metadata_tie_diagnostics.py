"""Identical-input limits in the article-free model; not deployable oracle results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.evaluation import outcomes,METRICS
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
x=np.load(DATA/'strict_test_X.npy',mmap_mode='r');y=np.load(DATA/'strict_test_y.npy');groups=np.load(DATA/'strict_test_groups.npy')
offsets=np.r_[0,np.cumsum(groups)];oracle=np.zeros(len(y));same=np.zeros(len(groups),dtype=bool);classes=np.zeros(len(groups),dtype=np.int32)
for j,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
    block=x[start:end];labels=y[start:end]
    if np.all(block==block[0]):
        same[j]=True;classes[j]=1;continue
    _,inverse=np.unique(block,axis=0,return_inverse=True)
    size=np.bincount(inverse);positives=np.bincount(inverse,weights=labels)
    oracle[start:end]=(positives/size)[inverse];classes[j]=len(size)
random=outcomes(y,np.zeros(len(y)),groups);ceiling=outcomes(y,oracle,groups)
actual=np.mean([pd.read_parquet(DATA/f'audit_holdout-events_user-s{s}_outcomes.parquet')[METRICS].to_numpy() for s in [42,43,44]],axis=0)
unknown=x[:,STRICT_FEATURES.index('history_item_unknown')]>0
all_unknown=np.add.reduceat(unknown,offsets[:-1])==groups
assert np.array_equal(same,all_unknown) or np.all(same[all_unknown])
assert np.allclose(actual[same,0],random.loc[same,'ndcg5'])
assert np.allclose(actual[same,1],random.loc[same,'hit1'])
result={'population':'All 244647 raw evaluation requests; no article-file eligibility check.',
    'all_candidates_identical_inputs':int(same.sum()),'fraction_all_identical':float(same.mean()),
    'all_candidates_absent_from_fixed_history':int(all_unknown.sum()),
    'mean_distinct_feature_vectors':float(classes.mean()),
    'metrics':{'random':random[['ndcg5','hit1']].mean().to_dict(),
               'event_user_model_mean':dict(zip(['ndcg5','hit1'],map(float,actual[:,:2].mean(axis=0)))),
               'label_oracle_constrained_to_identical_feature_ties':ceiling[['ndcg5','hit1']].mean().to_dict()},
    'identical_requests':{'random_ndcg5':float(random.loc[same,'ndcg5'].mean()),'learned_ndcg5':float(actual[same,0].mean())},
    'interpretation':'The oracle assigns each within-request identical-feature class its TRUE click fraction. It is an optimistic representational upper bound for deterministic per-candidate scoring with exactly these features, not a trained or servable model. The frozen history schedule is part of this limitation.'}
(OUT/'tie_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
