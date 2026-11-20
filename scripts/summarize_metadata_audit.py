"""Aggregate all audit arms without selecting a winner or changing a release."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.evaluation import METRICS,paired_interval

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-metadata-audit'
DATA=ROOT/'data/processed/ebnerd-metadata-audit'; BASE=ROOT/'data/processed/ebnerd-v1'
families=['retraining','full_retraining','permuted_retraining','eligibility_retraining','matched_category_retraining','exposure_retraining','exposure_latency_retraining']
runs=[r for f in families for r in json.loads((OUT/(f+'.json')).read_text())['runs']]
table=pd.DataFrame([{**{k:r[k] for k in ['name','arm','fold','seed','rounds','population']},**r['metrics']} for r in runs])
frames={p:pd.read_parquet(path) for p,path in [('raw',DATA/'strict_test_requests.parquet'),('clean',BASE/'validation_requests.parquet')]}
summaries=[]
for (arm,fold),sub in table.groupby(['arm','fold'],sort=False):
    summaries.append({'arm':arm,'fold':fold,'population':sub.population.iloc[0],'seeds':sub.seed.tolist(),
        'rounds':sub['rounds'].tolist(),'mean':sub[METRICS].mean().to_dict(),
        'min':sub[METRICS].min().to_dict(),'max':sub[METRICS].max().to_dict()})

def values(arm,fold):
    parts=[pd.read_parquet(DATA/f'{fold}-{arm}-s{s}_outcomes.parquet') for s in [42,43,44]]
    for p in parts[1:]:
        assert p[['impression_id','user_id']].equals(parts[0][['impression_id','user_id']])
    result=parts[0].copy()
    result[METRICS]=np.mean([p[METRICS].to_numpy() for p in parts],axis=0)
    return result

pairs=[('full_personalized','full_metadata'),('pub_category_user','pub_category'),
       ('pub_category_user','pub_category_no_affinity'),('pub_topic_user','pub_user'),
       ('pub_entity_user','pub_user'),('full_personalized','pub_category_user'),
       ('full_personalized','permuted_personalized'),('permuted_personalized','pub_user'),
       ('events_user','events_only'),('exposure_global','events_only'),
       ('exposure_personal','events_user'),('exposure_personal','exposure_global'),
       ('exposure_1h','events_user'),('exposure_24h','events_user')]
comparisons={}; by_day={}
for a,b in pairs:
    label=a+'__minus__'+b; comparisons[label]={}
    for fold in ['early','late','audit_holdout']:
        av=values(a,fold); bv=values(b,fold)
        assert av[['impression_id','user_id']].equals(bv[['impression_id','user_id']])
        # Average outcomes across seeds, then resample USERS; not 3x as many observations.
        comparisons[label][fold]={m:paired_interval(av.user_id,av[m]-bv[m]) for m in ['ndcg5','hit1','cold_hit1']}
        if fold=='audit_holdout':
            pop='raw' if a.startswith(('events_','exposure_')) else 'clean'
            frame=frames[pop]; assert np.array_equal(frame.impression_id,av.impression_id)
            daily=pd.DataFrame({'day':(frame.impression_time-pd.Timedelta(hours=7)).dt.strftime('%Y-%m-%d'),'ndcg_delta':av.ndcg5-bv.ndcg5,'hit1_delta':av.hit1-bv.hit1})
            by_day[label]=daily.groupby('day').agg(requests=('ndcg_delta','size'),ndcg_delta=('ndcg_delta','mean'),hit1_delta=('hit1_delta','mean')).reset_index().to_dict('records')
result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'models_fitted':len(runs),
        'evaluation_status':'Post-hoc exposed holdout; no new independent confirmation or promotion.',
        'uncertainty':'95% paired user-cluster bootstrap, 2000 draws, on per-query outcomes averaged over 3 seeds. Conditional on these periods and three fits; not temporal deployment uncertainty or causal leakage bounds.',
        'arms':summaries,'comparisons':comparisons,'daily_deltas':by_day}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(table[table.fold=='audit_holdout'].groupby('arm')[['ndcg5','hit1','cold_hit1']].agg(['mean','min','max']).to_string())
print('Saved',len(runs),'fits; all arms and comparisons retained.')
