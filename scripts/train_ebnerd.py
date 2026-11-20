"""Run predeclared development folds; this command never reads holdout labels."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from fairjob_ranker.ebnerd.features import FEATURES,VARIANTS,FeatureState
from fairjob_ranker.ebnerd.evaluation import outcomes,paired_interval,METRICS
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; data=ROOT/'data/processed/ebnerd-v1'
out=ROOT/'reports/ebnerd-experiment'; models=ROOT/'models/ebnerd-v1'
out.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'configs/ebnerd_experiment.json').read_text())
protocol=out/'protocol.json'
if not protocol.exists():
    protocol.write_text(json.dumps({'config':config,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'config_sha256':sha256(ROOT/'configs/ebnerd_experiment.json'),
        'prepared_manifest_sha256':sha256(data/'manifest.json')},indent=2)+'\n')
elif json.loads(protocol.read_text())['config_sha256']!=sha256(ROOT/'configs/ebnerd_experiment.json'):
    raise ValueError('Protocol changed')
frame=pd.read_parquet(data/'train_requests.parquet'); times=frame.impression_time.to_numpy()
x=np.load(data/'train_X.npy',mmap_mode='r'); y=np.load(data/'train_y.npy'); items=np.load(data/'train_items.npy'); groups=np.load(data/'train_groups.npy')
offset=np.r_[0,np.cumsum(groups)]; state=FeatureState.load(data/'train_state')
runs=[]; started=time.monotonic()
for fold in config['development_folds']:
    nf=int(np.searchsorted(times,np.datetime64(fold['fit_end']))); ne=int(np.searchsorted(times,np.datetime64(fold['validation_end'])))
    end=offset[nf]; ve=offset[ne]; queries=frame.iloc[nf:ne].reset_index(drop=True)
    known=set(map(int,items[:end]))|set(state.pop)
    cold=set(map(int,items[end:ve]))-known
    for variant in ['random','historical_popularity',*config['variants']]:
        name=fold['name']+'-'+variant; target=models/name; metric_path=out/(name+'.json')
        if metric_path.exists():
            runs.append(json.loads(metric_path.read_text())); print('resume',name,flush=True); continue
        print('training',name,'fit',nf,'validation',ne-nf,flush=True)
        columns=VARIANTS.get(variant,[]); indices=[FEATURES.index(c) for c in columns]
        t=time.monotonic()
        if variant=='random': scores=np.zeros(ve-end); rounds=0
        elif variant=='historical_popularity': scores=x[end:ve,FEATURES.index('history_pop')]; rounds=0
        else:
            params={**config['params'],'seed':config['seed'],'data_random_seed':config['seed'],'feature_fraction_seed':config['seed']}
            xt=np.array(x[:end,indices],order='C'); xv=np.array(x[end:ve,indices],order='C')
            cats=['category'] if 'category' in columns else []
            ds=lgb.Dataset(xt,label=y[:end],group=groups[:nf],feature_name=columns,categorical_feature=cats)
            valid=lgb.Dataset(xv,label=y[end:ve],group=groups[nf:ne],reference=ds,feature_name=columns,categorical_feature=cats)
            booster=lgb.train(params,ds,num_boost_round=config['max_rounds'],valid_sets=[valid],
                callbacks=[lgb.early_stopping(config['early_stopping_rounds'],verbose=False),lgb.log_evaluation(0)])
            rounds=booster.best_iteration; scores=booster.predict(xv,num_threads=4)
            target.mkdir(exist_ok=False); booster.save_model(str(target/'model.txt'))
            (target/'features.json').write_text(json.dumps(columns)+'\n')
            del xt,xv,ds,valid,booster
        result=outcomes(y[end:ve],scores,groups[nf:ne],items[end:ve],cold)
        result.insert(0,'user_id',queries.user_id); result.insert(0,'impression_id',queries.impression_id)
        result.to_parquet(data/(name+'_outcomes.parquet'),index=False)
        np.save(data/(name+'_scores.npy'),scores)
        row={'fold':fold['name'],'variant':variant,'fit_requests':nf,'validation_requests':ne-nf,
             'rounds':rounds,'seconds':time.monotonic()-t,'metrics':result[METRICS].mean().to_dict(),
             'model_sha256':sha256(target/'model.txt') if rounds else None}
        metric_path.write_text(json.dumps(row,indent=2)+'\n'); runs.append(row)
        print(json.dumps(row),flush=True)
comparisons={}
for fold in config['development_folds']:
    label=fold['name']; a=pd.read_parquet(data/(label+'-personalized_outcomes.parquet')); b=pd.read_parquet(data/(label+'-metadata_outcomes.parquet'))
    comparisons[label]={m:paired_interval(a.user_id,a[m]-b[m],config['bootstrap_repeats'],config['bootstrap_seed']) for m in ['ndcg5','hit1','cold_hit1']}
means={v:float(np.mean([r['metrics']['ndcg5'] for r in runs if r['variant']==v])) for v in config['variants']}
selected=max(means,key=means.get)
selection={'selected_variant':selected,'mean_development_ndcg5':means,'comparisons':comparisons,
    'runs':runs,'elapsed_seconds':time.monotonic()-started,'protocol_sha256':sha256(protocol),
    'test_not_scored':True,'rounds_for_refit':{v:next(r['rounds'] for r in runs if r['fold']=='late' and r['variant']==v) for v in config['variants']}}
(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print('SELECTED',selected,means,flush=True)
