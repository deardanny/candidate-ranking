"""Describe the information lost under longer declared log-arrival delays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed/ebnerd-metadata-audit';OUT=ROOT/'reports/ebnerd-metadata-audit'
groups=np.load(DATA/'strict_test_groups.npy');offsets=np.r_[0,np.cumsum(groups)];y=np.load(DATA/'strict_test_y.npy').astype(bool)
results={};previous=None;monotonic=[]
for seconds,prefix,arm in [(60,'exposure','exposure_personal'),(3600,'exposure_d3600','exposure_1h'),(86400,'exposure_d86400','exposure_24h')]:
    x=np.load(DATA/f'{prefix}_test_X.npy',mmap_mode='r')
    unseen=x[:,18]>0;no_exposure=np.add.reduceat(unseen,offsets[:-1])==groups
    identical=np.all(np.maximum.reduceat(x,offsets[:-1],axis=0)==np.minimum.reduceat(x,offsets[:-1],axis=0),axis=1)
    if previous is not None:
        ok=bool(np.all(previous[:,[13,19]]>=x[:,[13,19]]));assert ok;monotonic.append(ok)
    previous=x
    score=np.load(DATA/f'audit_holdout-{arm}-s42_scores.npy')
    score_tie=np.maximum.reduceat(score,offsets[:-1])==np.minimum.reduceat(score,offsets[:-1])
    assert score_tie[identical].all()
    per_query=np.mean([pd.read_parquet(DATA/f'audit_holdout-{arm}-s{s}_outcomes.parquet')[['ndcg5','hit1']].to_numpy() for s in [42,43,44]],axis=0)
    results[str(seconds)]={'candidate_rows_without_arrived_exposure':int(unseen.sum()),'candidate_fraction_without_arrived_exposure':float(unseen.mean()),
        'clicked_candidate_fraction_without_arrived_exposure':float(unseen[y].mean()),
        'requests_all_candidates_without_arrived_exposure':int(no_exposure.sum()),
        'requests_all_feature_vectors_identical':int(identical.sum()),'fraction_all_feature_vectors_identical':float(identical.mean()),
        'identical_inputs_imply_model_score_ties':True,
        'metrics_all_identical':dict(zip(['ndcg5','hit1'],map(float,per_query[identical].mean(axis=0)))) if identical.any() else None,
        'metrics_other':dict(zip(['ndcg5','hit1'],map(float,per_query[~identical].mean(axis=0))))}
(OUT/'exposure_delay_diagnostics.json').write_text(json.dumps({'results':results,'longer_delay_never_increases_global_or_user_past_exposure_counts':all(monotonic),
    'interpretation':'Descriptive information-availability mechanism; cohort composition changes with delay, so subgroup differences are not a decomposition of the total metric change.'},indent=2)+'\n')
print(json.dumps(results,indent=2))
