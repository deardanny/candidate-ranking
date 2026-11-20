"""Predeclared 2x2 development experiment, freeze selection, then refit. No holdout read."""
from pathlib import Path
from datetime import datetime, timezone
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from fairjob_ranker.ebnerd.feedback import hit1
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/ebnerd-metadata-audit'
DATA=ROOT/'data/processed/ebnerd-top1-feedback'
OUT=ROOT/'reports/ebnerd-top1-feedback'
MODELS=ROOT/'models/ebnerd-top1-feedback'
protocol=json.loads((OUT/'protocol.json').read_text())
features=json.loads((OUT/'preparation.json').read_text())['features']
frame=pd.read_parquet(BASE/'strict_train_requests.parquet')
X=np.load(DATA/'train_X.npy',mmap_mode='r')
y=np.load(BASE/'strict_train_y.npy'); groups=np.load(BASE/'strict_train_groups.npy')
starts=np.r_[0,np.cumsum(groups)]; times=frame.impression_time.to_numpy()
params={**json.loads((ROOT/'configs/ebnerd_experiment.json').read_text())['params']}
runs=[]
for arm in protocol['arms']:
    size=20 if arm.startswith('exposure') else X.shape[1]
    top1=arm.endswith('top1')
    for seed in protocol['seeds']:
        for fold in protocol['folds']:
            name=f"{fold['name']}-{arm}-s{seed}"; record=OUT/(name+'.json')
            if record.exists():
                runs.append(json.loads(record.read_text())); continue
            a=int(np.searchsorted(times,np.datetime64(fold['fit_end']))); b=int(np.searchsorted(times,np.datetime64(fold['end'])))
            nfit=int(np.searchsorted(times,np.datetime64(fold['fit_end'])-np.timedelta64(7200,'s')))
            fit_last=starts[nfit]; end=starts[a]; last=starts[b]
            train=lgb.Dataset(np.array(X[:fit_last,:size]),label=y[:fit_last],group=groups[:nfit],feature_name=features[:size])
            valid=lgb.Dataset(np.array(X[end:last,:size]),label=y[end:last],group=groups[a:b],reference=train,feature_name=features[:size])
            p={**params, 'seed':seed, 'data_random_seed':seed, 'feature_fraction_seed':seed,
                'ndcg_eval_at':[1 if top1 else 5], 'lambdarank_truncation_level':1 if top1 else 8}
            if top1: p['metric']='None'
            def expected_top1(scores, dataset):
                return 'expected_hit1',float(hit1(dataset.get_label(),scores,dataset.get_group()).mean()),True
            print('TRAIN',name,flush=True); t=time.monotonic()
            model=lgb.train(p,train,num_boost_round=180,valid_sets=[valid],
                feval=expected_top1 if top1 else None, callbacks=[lgb.early_stopping(25,verbose=False)])
            score=model.predict(np.array(X[end:last,:size]),num_threads=4)
            values=hit1(y[end:last],score,groups[a:b])
            np.save(DATA/(name+'_hit1.npy'),values)
            target=MODELS/name; target.mkdir(exist_ok=False); model.save_model(str(target/'model.txt'))
            result={'name':name,'arm':arm,'fold':fold['name'],'seed':seed,'rounds':model.best_iteration,
                'hit1':float(values.mean()),'requests':len(values),'fit_requests':nfit,
                'fit_labels_before':str(np.datetime64(fold['fit_end'])-np.timedelta64(7200,'s')),
                'seconds':time.monotonic()-t,'model_sha256':sha256(target/'model.txt')}
            record.write_text(json.dumps(result,indent=2)+'\n'); runs.append(result)
            print(json.dumps(result),flush=True)
            del train,valid,model,score
means={a:float(np.mean([r['hit1'] for r in runs if r['arm']==a])) for a in protocol['arms']}
selected=max(['feedback_ndcg','feedback_top1'],key=lambda a:means[a])
selection={'frozen_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(OUT/'protocol.json'),
    'mean_development_hit1':means,'selected':selected,'reason':'Predeclared mean development Hit1 across 2 folds and 3 seeds',
    'large_holdout_labels_read':False,'development_runs':runs}
if (OUT/'selection.json').exists():
    old=json.loads((OUT/'selection.json').read_text())
    assert old['selected']==selected and old['mean_development_hit1']==means
else: (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print('SELECTION',selected,means,flush=True)
final=[]
final_nfit=int(np.searchsorted(times,np.datetime64('2023-05-25 05:00:00')))
final_end=starts[final_nfit]
for arm in protocol['arms']:
    size=20 if arm.startswith('exposure') else X.shape[1]; top1=arm.endswith('top1')
    for seed in protocol['seeds']:
        name=f'final-{arm}-s{seed}'; record=OUT/(name+'.json'); target=MODELS/name
        if record.exists(): final.append(json.loads(record.read_text())); continue
        rounds=next(r['rounds'] for r in runs if r['arm']==arm and r['seed']==seed and r['fold']=='late')
        train=lgb.Dataset(np.array(X[:final_end,:size]),label=y[:final_end],group=groups[:final_nfit],feature_name=features[:size])
        p={**params,'seed':seed,'data_random_seed':seed,'feature_fraction_seed':seed,
            'ndcg_eval_at':[1 if top1 else 5],'lambdarank_truncation_level':1 if top1 else 8}
        print('REFIT',name,rounds,flush=True)
        model=lgb.train(p,train,num_boost_round=rounds)
        target.mkdir(exist_ok=False); model.save_model(str(target/'model.txt'))
        (target/'features.json').write_text(json.dumps(features[:size])+'\n')
        result={'name':name,'arm':arm,'seed':seed,'rounds':rounds,'fit_requests':final_nfit,
                'fit_candidate_rows':int(final_end),'fit_labels_before':'2023-05-25 05:00:00','model_sha256':sha256(target/'model.txt')}
        record.write_text(json.dumps(result,indent=2)+'\n'); final.append(result)
        del train,model
(OUT/'frozen_release.json').write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),
    'selection_sha256':sha256(OUT/'selection.json'),'models':final,'holdout_labels_read':False},indent=2)+'\n')
print('FROZEN',flush=True)
