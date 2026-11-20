"""Separate the article-dependent row filter from article-free feature effects."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.features import FEATURES
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES,ARMS
from fairjob_ranker.ebnerd.evaluation import outcomes,METRICS,paired_interval

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'data/processed/ebnerd-v1'; DATA=ROOT/'data/processed/ebnerd-metadata-audit'
OUT=ROOT/'reports/ebnerd-metadata-audit'; MODELS=ROOT/'models/ebnerd-metadata-audit'
checks={}; alignment={}
for split in ['train','test']:
    raw=pd.read_parquet(DATA/f'strict_{split}_requests.parquet')
    clean=pd.read_parquet(BASE/('train_requests.parquet' if split=='train' else 'validation_requests.parquet'))
    included=raw.impression_id.isin(clean.impression_id).to_numpy()
    assert np.array_equal(raw.loc[included,'impression_id'],clean.impression_id)
    rg=np.load(DATA/f'strict_{split}_groups.npy'); cg=np.load(BASE/f'{split}_groups.npy')
    rows=np.repeat(included,rg)
    x=np.load(DATA/f'strict_{split}_X.npy',mmap_mode='r'); original=np.load(BASE/f'{split}_X.npy',mmap_mode='r')
    for block in ['items','y']:
        assert np.array_equal(np.load(DATA/f'strict_{split}_{block}.npy')[rows],np.load(BASE/f'{split}_{block}.npy'))
    assert np.array_equal(rg[included],cg)
    assert np.array_equal(x[rows],original[:,[FEATURES.index(c) for c in STRICT_FEATURES]])
    checks[split]={'identical_all_13_event_features_on_common_candidates':True,'candidate_ids_labels_groups_identical':True,
        'raw_requests':len(raw),'clean_requests':len(clean),'excluded_requests':int((~included).sum()),
        'common_candidates':int(rows.sum())}
    alignment[split]=included

raw=pd.read_parquet(DATA/'strict_test_requests.parquet'); included=alignment['test']
x=np.load(DATA/'strict_test_X.npy',mmap_mode='r'); y=np.load(DATA/'strict_test_y.npy'); ids=np.load(DATA/'strict_test_items.npy'); groups=np.load(DATA/'strict_test_groups.npy')
results={}
for arm in ['events_only','events_user']:
    features=ARMS[arm]; ix=[STRICT_FEATURES.index(c) for c in features]; clean_results=[]; raw_results=[]; seed_details=[]
    for seed in [42,43,44]:
        model=lgb.Booster(model_file=str(MODELS/f'audit_holdout-clean_{arm}-s{seed}/model.txt'))
        score=model.predict(np.array(x[:,ix],order='C'),num_threads=4)
        cr=outcomes(y,score,groups); rr=pd.read_parquet(DATA/f'audit_holdout-{arm}-s{seed}_outcomes.parquet')
        assert np.array_equal(rr.impression_id,raw.impression_id)
        clean_results.append(cr); raw_results.append(rr)
        stored=pd.read_parquet(DATA/f'audit_holdout-clean_{arm}-s{seed}_outcomes.parquet')
        assert np.array_equal(cr.loc[included,['ndcg5','hit1','mrr','auc']].to_numpy(),stored[['ndcg5','hit1','mrr','auc']].to_numpy())
        seed_details.append({'seed':seed,'clean_trained_raw_evaluation':cr[['ndcg5','hit1']].mean().to_dict(),
            'raw_trained_raw_evaluation':rr[['ndcg5','hit1']].mean().to_dict()})
    cr=np.mean([a[['ndcg5','hit1']].to_numpy() for a in clean_results],axis=0)
    rr=np.mean([a[['ndcg5','hit1']].to_numpy() for a in raw_results],axis=0)
    result={'seeds':seed_details,'raw_minus_clean_training_on_same_requests':{},'row_selection_effect':{}}
    for name,mask in [('all_raw',np.ones(len(raw),dtype=bool)),('common_clean',included),('excluded',~included)]:
        result['raw_minus_clean_training_on_same_requests'][name]={'requests':int(mask.sum()),'metrics':{}}
        for j,m in enumerate(['ndcg5','hit1']):
            result['raw_minus_clean_training_on_same_requests'][name]['metrics'][m]=paired_interval(raw.loc[mask,'user_id'],(rr-cr)[mask,j])
    for j,m in enumerate(['ndcg5','hit1']):
        result['row_selection_effect'][m]={'same_raw_model_all_raw':float(rr[:,j].mean()),'same_raw_model_common_clean':float(rr[included,j].mean()),
            'difference_clean_minus_raw':float(rr[included,j].mean()-rr[:,j].mean())}
    results[arm]=result
(OUT/'eligibility.json').write_text(json.dumps({'checks':checks,'results':results,
    'scope':'Whole-request eligibility changes, same candidates/labels on retained rows. Cold-hit not compared across differing training vocabularies.'},indent=2)+'\n')
print(json.dumps({'checks':checks,'results':results},indent=2))
