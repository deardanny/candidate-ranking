"""All fixed ablations/seeds; reuses exposed holdout only for a post-hoc audit."""
import json
import argparse
from pathlib import Path
from datetime import datetime,timezone
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.metadata_audit import ARMS,STRICT_FEATURES
from fairjob_ranker.ebnerd.features import FEATURES
from fairjob_ranker.ebnerd.evaluation import outcomes,METRICS
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; base=ROOT/'data/processed/ebnerd-v1'; data=ROOT/'data/processed/ebnerd-metadata-audit'
out=ROOT/'reports/ebnerd-metadata-audit'; models=ROOT/'models/ebnerd-metadata-audit'
config=json.loads((ROOT/'configs/ebnerd_experiment.json').read_text()); protocol=json.loads((out/'protocol.json').read_text())
for relative,digest in protocol['previous_artifact_sha256'].items():
    if sha256(ROOT/relative)!=digest:raise ValueError(f'Frozen source changed: {relative}')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--family',choices=['base','full','permuted','eligibility','matched_category','exposure','exposure_latency'],default='base')
args=parser.parse_args()
arms=ARMS
if args.family=='full':
    from fairjob_ranker.ebnerd.features import VARIANTS
    arms={'full_metadata':VARIANTS['metadata'],'full_personalized':VARIANTS['personalized']}
elif args.family=='permuted':
    arms={'permuted_personalized':FEATURES}
elif args.family=='eligibility':
    arms={'clean_events_only':ARMS['events_only'],'clean_events_user':ARMS['events_user']}
elif args.family=='matched_category':
    arms={'pub_category_no_affinity':ARMS['pub_category_user'][:-2]}
elif args.family=='exposure':
    from fairjob_ranker.ebnerd.exposure_audit import EXPOSURE_FEATURES
    arms={'exposure_global':ARMS['events_only']+EXPOSURE_FEATURES[:-1],
          'exposure_personal':STRICT_FEATURES+EXPOSURE_FEATURES}
elif args.family=='exposure_latency':
    from fairjob_ranker.ebnerd.exposure_audit import EXPOSURE_FEATURES
    arms={'exposure_1h':STRICT_FEATURES+EXPOSURE_FEATURES,'exposure_24h':STRICT_FEATURES+EXPOSURE_FEATURES}
origins={}
for population in ['raw','clean']:
    strict=population=='raw'; directory=data if strict else base
    def filename(split,part): return directory/(f'strict_{split}_{part}' if strict else f'{split}_{part}')
    frames={s:pd.read_parquet(filename(s,'requests').with_suffix('.parquet') if strict else base/('train_requests.parquet' if s=='train' else 'validation_requests.parquet')) for s in ['train','test']}
    arrays={s:{k:np.load(filename(s,k).with_suffix('.npy'),mmap_mode='r') for k in ['X','y','items','groups']} for s in frames}
    if args.family=='permuted' and population=='clean':
        for s in arrays: arrays[s]['X']=np.load(data/f'permuted_{s}_X.npy',mmap_mode='r')
    if args.family=='exposure' and population=='raw':
        for s in arrays: arrays[s]['X']=np.load(data/f'exposure_{s}_X.npy',mmap_mode='r')
    origins[population]=(frames,arrays)
history_ids=set(json.loads((data/'strict_history_ids.json').read_text()))
rows=[]
for arm,features in arms.items():
    population='raw' if arm.startswith(('events_','exposure_')) else 'clean'; frames,arrays=origins[population]
    if args.family=='exposure_latency':
        delay={'exposure_1h':3600,'exposure_24h':86400}[arm]
        arrays={s:{**a,'X':np.load(data/f'exposure_d{delay}_{s}_X.npy',mmap_mode='r')} for s,a in arrays.items()}
    universe=STRICT_FEATURES if population=='raw' else FEATURES
    if args.family in ['exposure','exposure_latency']:universe=STRICT_FEATURES+EXPOSURE_FEATURES
    indices=[universe.index(c) for c in features]
    train=arrays['train']; test=arrays['test']; groups=train['groups']; offsets=np.r_[0,np.cumsum(groups)]
    for seed in protocol['seeds']:
        rounds={}
        for fold in [*config['development_folds'],{'name':'audit_holdout'}]:
            name=f"{fold['name']}-{arm}-s{seed}"; path=out/(name+'.json'); target=models/name
            if path.exists():
                record=json.loads(path.read_text())
                if record['model_sha256']!=sha256(target/'model.txt') or json.loads((target/'features.json').read_text())!=features:
                    raise ValueError(f'Resume artifact mismatch: {name}')
                rows.append(record); rounds[fold['name']]=record['rounds']; print('resume',name,flush=True); continue
            held=fold['name']=='audit_holdout'
            if not held:
                times=frames['train'].impression_time.to_numpy()
                nfit=int(np.searchsorted(times,np.datetime64(fold['fit_end']))); nend=int(np.searchsorted(times,np.datetime64(fold['validation_end'])))
                fit_end=offsets[nfit]; end=offsets[nend]; evaluation=frames['train'].iloc[nfit:nend].reset_index(drop=True)
                xv=np.array(train['X'][fit_end:end,indices],order='C'); yv=train['y'][fit_end:end]; gi=groups[nfit:nend]; ids=train['items'][fit_end:end]
            else:
                nfit=len(groups); fit_end=len(train['y']); evaluation=frames['test']; xv=np.array(test['X'][:,indices],order='C')
                yv=test['y']; gi=test['groups']; ids=test['items']
            started=time.monotonic(); print('audit train',name,nfit,flush=True)
            xt=np.array(train['X'][:fit_end,indices],order='C'); cats=['category'] if 'category' in features else []
            ds=lgb.Dataset(xt,label=train['y'][:fit_end],group=groups[:nfit],feature_name=features,categorical_feature=cats)
            params={**config['params'],'seed':seed,'data_random_seed':seed,'feature_fraction_seed':seed}
            if not held:
                valid=lgb.Dataset(xv,label=yv,group=gi,reference=ds,feature_name=features,categorical_feature=cats)
                model=lgb.train(params,ds,num_boost_round=config['max_rounds'],valid_sets=[valid],callbacks=[lgb.early_stopping(config['early_stopping_rounds'],verbose=False)])
                selected=model.best_iteration; del valid
            else:
                selected=rounds['late']; model=lgb.train(params,ds,num_boost_round=selected)
            target.mkdir(parents=True,exist_ok=False); model.save_model(str(target/'model.txt'))
            (target/'features.json').write_text(json.dumps(features)+'\n'); score=model.predict(xv,num_threads=4)
            cold=set(map(int,ids))-(set(map(int,train['items'][:fit_end]))|history_ids)
            result=outcomes(yv,score,gi,ids,cold); result.insert(0,'user_id',evaluation.user_id); result.insert(0,'impression_id',evaluation.impression_id)
            result.to_parquet(data/(name+'_outcomes.parquet'),index=False); np.save(data/(name+'_scores.npy'),score)
            record={'name':name,'arm':arm,'population':population,'fold':fold['name'],'seed':seed,'rounds':int(selected),
                'fit_requests':nfit,'evaluation_requests':len(evaluation),'metrics':result[METRICS].mean().to_dict(),
                'seconds':time.monotonic()-started,'model_sha256':sha256(target/'model.txt')}
            path.write_text(json.dumps(record,indent=2)+'\n'); rows.append(record); rounds[fold['name']]=int(selected)
            print(json.dumps(record),flush=True); del xt,xv,model,ds
summary_name='retraining.json' if args.family=='base' else args.family+'_retraining.json'
(out/summary_name).write_text(json.dumps({'runs':rows,'protocol_sha256':sha256(out/'protocol.json'),'completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
