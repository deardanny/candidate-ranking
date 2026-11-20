"""Refit frozen development choices, then evaluate local holdout exactly once."""
import json
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from fairjob_ranker.io import sha256
from fairjob_ranker.ebnerd.features import FEATURES,VARIANTS,USER,PERSONAL,FeatureState,build_matrix
from fairjob_ranker.ebnerd.bundle import Bundle
from fairjob_ranker.ebnerd.evaluation import outcomes,paired_interval,METRICS

ROOT=Path(__file__).resolve().parents[1]; data=ROOT/'data/processed/ebnerd-v1'
out=ROOT/'reports/ebnerd-experiment'; models=ROOT/'models/ebnerd-v1'
config=json.loads((ROOT/'configs/ebnerd_experiment.json').read_text())
selection=json.loads((out/'selection.json').read_text())
if (out/'final_metrics.json').exists(): raise SystemExit('Holdout already evaluated; refusing a second selection/evaluation run')
x=np.load(data/'train_X.npy',mmap_mode='r'); y=np.load(data/'train_y.npy'); groups=np.load(data/'train_groups.npy'); items=np.load(data/'train_items.npy')
train_state=FeatureState.load(data/'train_state'); seen=sorted(set(map(int,items))|set(train_state.pop))
refit={}
for variant in config['variants']:
    target=models/('final-'+variant); columns=VARIANTS[variant]; indices=[FEATURES.index(c) for c in columns]
    if not (target/'metadata.json').exists():
        target.mkdir(exist_ok=True); started=time.monotonic(); print('refit',variant,flush=True)
        xt=np.array(x[:,indices],order='C'); cats=['category'] if 'category' in columns else []
        ds=lgb.Dataset(xt,label=y,group=groups,feature_name=columns,categorical_feature=cats)
        model=lgb.train({**config['params'],'seed':config['seed'],'data_random_seed':config['seed']},ds,
            num_boost_round=selection['rounds_for_refit'][variant])
        model.save_model(str(target/'model.txt')); (target/'features.json').write_text(json.dumps(columns)+'\n')
        (target/'seen_items.json').write_text(json.dumps(seen)+'\n')
        meta={'schema_version':'ebnerd-v1','score_type':'ranking_score','variant':variant,
            'model_version':variant+'-'+sha256(target/'model.txt')[:12],
            'fit_source':'clean official train; fixed prior history','selection_sha256':sha256(out/'selection.json'),
            'content_snapshot_assumption':bool('category' in columns),'publication_stability_assumption':bool('publication_age' in columns),
            'rounds':selection['rounds_for_refit'][variant],'seconds':time.monotonic()-started,
            'sha256':{n:sha256(target/n) for n in ['model.txt','features.json','seen_items.json']}}
        (target/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); del xt,ds,model
    refit[variant]={'bundle':str(target.relative_to(ROOT)),'metadata_sha256':sha256(target/'metadata.json')}
frozen={'selected_variant':selection['selected_variant'],'refit':refit,'protocol_sha256':sha256(out/'protocol.json'),
        'selection_sha256':sha256(out/'selection.json'),'frozen_before_holdout_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{n:sha256(ROOT/n) for n in ['src/fairjob_ranker/ebnerd/features.py','src/fairjob_ranker/ebnerd/evaluation.py','scripts/finish_ebnerd.py']}}
(out/'frozen_release.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('release frozen; opening holdout',flush=True)
frame=pd.read_parquet(data/'validation_requests.parquet'); state=FeatureState.load(data/'validation_state')
tx,ty,ti,tg=build_matrix(frame,state,lambda s:print('test '+s,flush=True))
for name,values in [('X',tx),('y',ty),('items',ti),('groups',tg)]: np.save(data/f'test_{name}.npy',values)
cold=set(map(int,ti))-set(seen); all_out={}; summary={}; predictions={}
for variant in ['random','historical_popularity',*config['variants']]:
    scores=(np.zeros(len(tx)) if variant=='random' else tx[:,FEATURES.index('history_pop')] if variant=='historical_popularity'
            else Bundle(models/('final-'+variant)).predict(tx))
    result=outcomes(ty,scores,tg,ti,cold); result.insert(0,'user_id',frame.user_id); result.insert(0,'impression_id',frame.impression_id)
    all_out[variant]=result; summary[variant]=result[METRICS].mean().to_dict(); predictions[variant]=scores
    result.to_parquet(data/('test-'+variant+'_outcomes.parquet'),index=False); np.save(data/('test-'+variant+'_scores.npy'),scores)
    print(variant,summary[variant],flush=True)
# Negative control is predeclared, uses only histories before this split and no labels.
users=np.array(sorted(state.users)); perm=np.random.default_rng(config['seed']+1000).permutation(users)
mapping=dict(zip(map(int,users),map(int,perm))); columns=[FEATURES.index(c) for c in USER+PERSONAL]
sx=tx.copy(); offset=0
for j,r in enumerate(frame.itertuples(index=False)):
    n=len(r.article_ids_inview); matrix=state.matrix(r.user_id,r.impression_time,r.device_type,r.article_ids_inview,history_user_id=mapping[int(r.user_id)])
    sx[offset:offset+n,columns]=matrix[:,columns]; offset+=n
    if (j+1)%50000==0: print('history control',j+1,flush=True)
scores=Bundle(models/'final-personalized').predict(sx); result=outcomes(ty,scores,tg,ti,cold)
result.insert(0,'user_id',frame.user_id); result.insert(0,'impression_id',frame.impression_id)
all_out['shuffled_history']=result; summary['shuffled_history']=result[METRICS].mean().to_dict()
result.to_parquet(data/'test-shuffled_history_outcomes.parquet',index=False); np.save(data/'test-shuffled_history_scores.npy',scores)
del sx
comparisons={}
for candidate,reference in [('personalized','metadata'),('personalized','shuffled_history'),('behavior_history','behavior'),('freshness','behavior'),('personalized','personalized_no_freshness')]:
    comparisons[candidate+' - '+reference]={m:paired_interval(frame.user_id,all_out[candidate][m]-all_out[reference][m],config['bootstrap_repeats'],config['bootstrap_seed']) for m in ['ndcg5','hit1','cold_hit1']}
cohorts={}; offsets=np.r_[0,np.cumsum(tg)]; coldrows=np.isin(ti,list(cold))
hascold=np.add.reduceat(coldrows,offsets[:-1])>0; allcold=np.add.reduceat(coldrows,offsets[:-1])==tg
trainusers=set(pd.read_parquet(data/'train_requests.parquet',columns=['user_id']).user_id)
masks={'has_cold':hascold,'all_cold':allcold,'mixed':hascold & ~allcold,'new_user_with_history':~frame.user_id.isin(trainusers).to_numpy()}
day=(frame.impression_time-pd.Timedelta(hours=7)).dt.strftime('%Y-%m-%d')
masks.update({'day_'+d:day.eq(d).to_numpy() for d in sorted(day.unique())})
for name,mask in masks.items():
    delta=all_out['personalized'].loc[mask,'ndcg5']-all_out['metadata'].loc[mask,'ndcg5']
    cohorts[name]={'requests':int(mask.sum()),'personalized':all_out['personalized'].loc[mask,METRICS].mean().to_dict(),
        'metadata':all_out['metadata'].loc[mask,METRICS].mean().to_dict(),
        'ndcg_difference':paired_interval(frame.loc[mask,'user_id'],delta,config['bootstrap_repeats'],config['bootstrap_seed'])}
primary=comparisons['personalized - metadata']
gate={'both_development_folds_positive':all(selection['comparisons'][f]['ndcg5']['estimate']>0 for f in ['early','late']),
      'holdout_ndcg_lower_positive':primary['ndcg5']['lower']>0,'hit1_nonnegative':primary['hit1']['estimate']>=0,
      'cold_hit1_nonnegative':primary['cold_hit1']['estimate']>=0}
result={'metrics':summary,'comparisons':comparisons,'cohorts':cohorts,'quality_checks':gate,'quality_passed':all(gate.values()),
    'selected_before_test':selection['selected_variant'],'frozen_release_sha256':sha256(out/'frozen_release.json'),
    'requests':len(frame),'users':int(frame.user_id.nunique()),'cold_items':len(cold),
    'evaluation_scope':'conditional observed-click ranking; metadata snapshots not point-in-time verified',
    'scored_at_utc':datetime.now(timezone.utc).isoformat()}
(out/'final_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print('FINAL',json.dumps({'quality_checks':gate,'metrics':summary,'primary':primary},indent=2),flush=True)
