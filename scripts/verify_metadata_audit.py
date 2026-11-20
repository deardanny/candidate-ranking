"""Reload every audit fit, independently check sampled metrics, preserve originals."""
from datetime import datetime,timezone
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score,roc_auc_score
from fairjob_ranker.ebnerd.features import FEATURES
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES
from fairjob_ranker.ebnerd.exposure_audit import EXPOSURE_FEATURES
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-metadata-audit'; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; BASE=ROOT/'data/processed/ebnerd-v1'; MODELS=ROOT/'models/ebnerd-metadata-audit'
families=['retraining','full_retraining','permuted_retraining','eligibility_retraining','matched_category_retraining','exposure_retraining','exposure_latency_retraining']
records=[r for name in families for r in json.loads((OUT/(name+'.json')).read_text())['runs']]
protocol=json.loads((OUT/'protocol.json').read_text());checks={}
for path,digest in protocol['previous_artifact_sha256'].items():
    assert sha256(ROOT/path)==digest,path
checks['all_original_protocol_hashes_unchanged']=True
arrays={};frames={}
for pop in ['raw','clean']:
    for split in ['train','test']:
        directory=DATA if pop=='raw' else BASE;prefix='strict_' if pop=='raw' else ''
        arrays[pop,split]={k:np.load(directory/f'{prefix}{split}_{k}.npy',mmap_mode='r') for k in ['X','y','groups']}
        name=f'strict_{split}_requests.parquet' if pop=='raw' else ('train_requests.parquet' if split=='train' else 'validation_requests.parquet')
        frames[pop,split]=pd.read_parquet(directory/name,columns=['impression_id','user_id','impression_time'])
config=json.loads((ROOT/'configs/ebnerd_experiment.json').read_text());folds={f['name']:f for f in config['development_folds']}
verified=[]
for row in records:
    name=row['name'];arm=row['arm'];pop=row['population'];split='test' if row['fold']=='audit_holdout' else 'train'
    model_path=MODELS/name/'model.txt';assert sha256(model_path)==row['model_sha256']
    cols=json.loads((MODELS/name/'features.json').read_text());source=arrays[pop,split];frame=frames[pop,split]
    matrix=source['X'];universe=FEATURES if pop=='clean' else STRICT_FEATURES
    if arm.startswith('exposure_'):
        prefix={'exposure_1h':'exposure_d3600','exposure_24h':'exposure_d86400'}.get(arm,'exposure')
        matrix=np.load(DATA/f'{prefix}_{split}_X.npy',mmap_mode='r');universe=STRICT_FEATURES+EXPOSURE_FEATURES
    elif arm=='permuted_personalized':matrix=np.load(DATA/f'permuted_{split}_X.npy',mmap_mode='r')
    if split=='train':
        fold=folds[row['fold']];times=frame.impression_time.to_numpy()
        lo=int(np.searchsorted(times,np.datetime64(fold['fit_end'])));hi=int(np.searchsorted(times,np.datetime64(fold['validation_end'])))
    else:lo=0;hi=len(frame)
    offsets=np.r_[0,np.cumsum(source['groups'])];begin,end=offsets[lo],offsets[hi]
    groups=source['groups'][lo:hi];y=source['y'][begin:end];query=frame.iloc[lo:hi]
    scores=np.load(DATA/f'{name}_scores.npy');saved=pd.read_parquet(DATA/f'{name}_outcomes.parquet')
    assert np.array_equal(query.impression_id,saved.impression_id) and np.array_equal(query.user_id,saved.user_id)
    booster=lgb.Booster(model_file=str(model_path));ix=[universe.index(c) for c in cols]
    actual=booster.predict(np.array(matrix[begin:end,ix],order='C'),num_threads=4)
    assert np.array_equal(actual,scores),name
    sample=np.sort(np.random.default_rng(20261119).choice(len(groups),min(250,len(groups)),replace=False));qo=np.r_[0,np.cumsum(groups)]
    error={'ndcg5':0.,'auc':0.,'hit1':0.}
    for j in sample:
        a,b=qo[j],qo[j+1];labels=y[a:b];prediction=scores[a:b]
        independent={'ndcg5':ndcg_score([labels],[prediction],k=5),'auc':roc_auc_score(labels,prediction),
                     'hit1':float(labels[prediction==prediction.max()].mean())}
        for metric,value in independent.items():error[metric]=max(error[metric],abs(value-saved.iloc[j][metric]))
    assert max(error.values())<1e-12,(name,error)
    verified.append({'name':name,'prediction_rows':len(scores),'prediction_max_abs_error':0.,'independent_queries':len(sample),'metric_max_abs_error':error,'model_sha256':sha256(model_path)})
    print('verified',len(verified),'/',len(records),name,flush=True)
for old,new in [('metadata','full_metadata'),('personalized','full_personalized'),('behavior','clean_events_only'),('behavior_history','clean_events_user')]:
    for fold in ['early','late','audit_holdout']:
        oldprefix='test' if fold=='audit_holdout' else fold
        assert np.array_equal(np.load(BASE/f'{oldprefix}-{old}_scores.npy'),np.load(DATA/f'{fold}-{new}-s42_scores.npy')),(old,fold)
checks['original_seed42_all_four_replays_exact_in_all_three_folds']=True
for path in [ROOT/'models/ebnerd-v1/registry.json',ROOT/'models/v2/registry.json']:
    if path.exists():checks[str(path.relative_to(ROOT))+'_sha256_at_verification']=sha256(path)
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'models_verified':len(verified),
        'prediction_rows_recomputed':sum(r['prediction_rows'] for r in verified),'independent_metric_queries':sum(r['independent_queries'] for r in verified),'runs':verified}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('VERIFIED',len(verified),'models',flush=True)
