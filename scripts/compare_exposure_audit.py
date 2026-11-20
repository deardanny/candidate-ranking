"""Same-query, same-cold-definition comparisons for exposure and snapshot models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.evaluation import paired_interval

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-metadata-audit'; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; BASE=ROOT/'data/processed/ebnerd-v1'
raw=pd.read_parquet(DATA/'strict_test_requests.parquet');clean=pd.read_parquet(BASE/'validation_requests.parquet')
included=raw.impression_id.isin(clean.impression_id).to_numpy();groups=np.load(DATA/'strict_test_groups.npy');rowmask=np.repeat(included,groups)
assert np.array_equal(raw.loc[included,'impression_id'],clean.impression_id)
items=np.load(DATA/'strict_test_items.npy')[rowmask];y=np.load(DATA/'strict_test_y.npy')[rowmask];cg=groups[included];off=np.r_[0,np.cumsum(cg)]
seen=set(map(int,np.load(DATA/'strict_train_items.npy')))|set(json.loads((DATA/'strict_history_ids.json').read_text()))
cold=~np.isin(items,list(seen));allcold=np.add.reduceat(cold,off[:-1])==cg
train_users=set(pd.read_parquet(DATA/'strict_train_requests.parquet',columns=['user_id']).user_id)
newuser=~clean.user_id.isin(train_users).to_numpy()
extra=np.load(DATA/'exposure_test_X.npy',mmap_mode='r')[rowmask]
all_first_seen=np.add.reduceat(extra[:,18]>0,off[:-1])==cg  # 13 event features + exposure_unseen at offset 5

arms=['full_personalized','full_metadata','exposure_global','exposure_personal','exposure_1h','exposure_24h']
values={};seed_metrics={}
for arm in arms:
    parts=[];seeds=[]
    for seed in [42,43,44]:
        prefix=f'audit_holdout-{arm}-s{seed}';v=pd.read_parquet(DATA/f'{prefix}_outcomes.parquet')
        scores=np.load(DATA/f'{prefix}_scores.npy')
        if arm.startswith('exposure_'):v=v.loc[included].reset_index(drop=True);scores=scores[rowmask]
        assert np.array_equal(v.impression_id,clean.impression_id)
        top=scores==np.repeat(np.maximum.reduceat(scores,off[:-1]),cg)
        v['common_cold_hit1']=np.add.reduceat(top & y.astype(bool) & cold,off[:-1])/np.add.reduceat(top,off[:-1])
        seeds.append({'seed':seed,'ndcg5':float(v.ndcg5.mean()),'hit1':float(v.hit1.mean()),'common_cold_hit1':float(v.common_cold_hit1.mean())})
        parts.append(v[['ndcg5','hit1','common_cold_hit1']].to_numpy())
    values[arm]=np.mean(parts,axis=0);seed_metrics[arm]=seeds
cohorts={'all_common':np.ones(len(clean),bool),'all_model_cold':allcold,'new_user_with_fixed_history':newuser,
         'all_candidates_first_seen_with_60s_arrival':all_first_seen}
comparison={}
for arm in ['exposure_global','exposure_personal','exposure_1h','exposure_24h']:
    comparison[arm]={}
    for name,mask in cohorts.items():
        comparison[arm][name]={'requests':int(mask.sum()),'candidate_source':'Unchanged complete slates',
            'metrics':dict(zip(['ndcg5','hit1','common_cold_hit1'],map(float,values[arm][mask].mean(axis=0)))),
            'minus_full_personalized':{m:paired_interval(clean.loc[mask,'user_id'],(values[arm]-values['full_personalized'])[mask,j]) for j,m in enumerate(['ndcg5','hit1','common_cold_hit1'])}}
daily={}
for arm in ['exposure_personal','exposure_1h','exposure_24h']:
    delta=values[arm]-values['full_personalized']
    f=pd.DataFrame({'day':(clean.impression_time-pd.Timedelta(hours=7)).dt.strftime('%Y-%m-%d'),'ndcg_delta':delta[:,0],'hit1_delta':delta[:,1]})
    daily[arm]=f.groupby('day').agg(requests=('ndcg_delta','size'),ndcg_delta=('ndcg_delta','mean'),hit1_delta=('hit1_delta','mean')).reset_index().to_dict('records')
result={'evaluation_requests':len(clean),'evaluation_users':int(clean.user_id.nunique()),
    'common_cold_definition':'Absent from raw training candidates AND train fixed history. This shared stricter set is used for every arm, regardless of its training eligibility filter.',
    'models':{a:{'mean':dict(zip(['ndcg5','hit1','common_cold_hit1'],map(float,v.mean(axis=0)))),'seeds':seed_metrics[a]} for a,v in values.items()},
    'cohorts':comparison,'daily_deltas_vs_full_personalized':daily,
    'scope':'Post-hoc replay. Exposure arms trained on 232887 raw requests, snapshot arms on 232246 cleaned requests; training-population difference is separately audited in eligibility.json. All comparisons here use identical 244587 evaluation slates. No test selection, no model promotion.'}
(OUT/'exposure_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['models'],indent=2))
