"""Independent timestamp-array reconstruction of response features and metrics."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-top1-feedback'; OUT=ROOT/'reports/ebnerd-top1-feedback'; BASE=ROOT/'data/processed/ebnerd-metadata-audit'
ref=pd.concat([pd.read_parquet(BASE/f'strict_{s}_requests.parquet') for s in ['train','test']],ignore_index=True)
ref=ref.sort_values(['impression_time','impression_id'],kind='stable').reset_index(drop=True)
t=ref.impression_time.to_numpy(dtype='datetime64[s]').astype('int64')
g=ref.article_ids_inview.map(len).to_numpy()
ids=np.concatenate(ref.article_ids_inview.to_numpy())
yy=np.concatenate([np.isin(ii,list(set(map(int,c)))) for ii,c in zip(ref.article_ids_inview,ref.article_ids_clicked)])
events=pd.DataFrame({'item':ids,'t':np.repeat(t,g),'click':yy})
by_id={int(i):(f.t.to_numpy(),f.loc[f.click,'t'].to_numpy()) for i,f in events.groupby('item',sort=False)}
prefix_n=np.r_[0,np.cumsum(g)]; prefix_c=np.r_[0,np.cumsum(ref.article_ids_clicked.map(lambda a:len(set(a))).to_numpy())]
empty=np.array([],dtype='int64'); rng=np.random.default_rng(20261119); verified=0; max_error=0.
checks=[]
sources=[('train',3600,ref.iloc[:len(pd.read_parquet(BASE/'strict_train_requests.parquet',columns=['user_id']))],DATA/'train_X.npy')]
if (DATA/'fresh_requests.parquet').exists():
    fresh=pd.read_parquet(DATA/'fresh_requests.parquet')
    sources.extend([('fresh',d,fresh,DATA/f'fresh_d{d}_X.npy') for d in [3600,86400,'frozen']])
for name,delay,frame,path in sources:
    x=np.load(path,mmap_mode='r'); gg=frame.article_ids_inview.map(len).to_numpy(); offsets=np.r_[0,np.cumsum(gg)]
    chosen=np.unique(np.r_[0,len(frame)-1,rng.integers(0,len(frame),size=1000)])
    for q in chosen:
        row=frame.iloc[q]; now=int(np.datetime64(row.impression_time,'s').astype('int64')); cutoff=now-(3600 if delay=='frozen' else delay)
        event_upper=min(cutoff,int(np.datetime64('2023-05-25 05:00:00','s').astype('int64'))) if delay=='frozen' else cutoff
        upper=int(np.searchsorted(t,event_upper,side='left')); columns=[]
        for window in [0,3600,86400,604800]:
            lower=min(upper,int(np.searchsorted(t,cutoff-window,side='left'))) if window else 0
            ne=prefix_n[upper]-prefix_n[lower]; nc=prefix_c[upper]-prefix_c[lower]; prior=nc/ne if ne else .1
            n=[];c=[]
            for item in row.article_ids_inview:
                te,tc=by_id.get(int(item),(empty,empty))
                n.append(np.searchsorted(te,event_upper,'left')-(np.searchsorted(te,min(event_upper,cutoff-window),'left') if window else 0))
                c.append(np.searchsorted(tc,event_upper,'left')-(np.searchsorted(tc,min(event_upper,cutoff-window),'left') if window else 0))
            n=np.array(n);c=np.array(c);p=(c+20*prior)/(n+20)
            columns.extend([np.log1p(n),np.log1p(c),p,np.sqrt(p*(1-p)/(n+21))])
        p=columns[10]; rank=np.array([np.mean(p<v)+.5*np.mean(p==v) for v in p])
        expected=np.column_stack([*columns,rank,p-p.max()]).astype('float32')
        actual=x[offsets[q]:offsets[q+1],20:]
        error=float(np.abs(actual-expected).max());max_error=max(error,max_error)
        np.testing.assert_array_equal(actual,expected)
        verified+=actual.size
    checks.append({'source':name,'delay':delay,'requests':len(chosen),'all_18_response_features_exact':True})
    print('Verified',name,delay,flush=True)
metric_checks=0
reload_values=0
if (OUT/'final_results.json').exists():
    y=np.load(DATA/'fresh_y.npy'); groups=np.load(DATA/'fresh_groups.npy'); offset=np.r_[0,np.cumsum(groups)]
    frozen=json.loads((OUT/'predictions_frozen.json').read_text())
    for p in frozen['predictions']:
        key=f"d{p['delay']}_{p['model']}";score=np.load(DATA/f'fresh_{key}_scores.npy');actual=np.load(DATA/f'{key}_hit1.npy')
        for q in rng.integers(0,len(groups),size=1000):
            s=score[offset[q]:offset[q+1]]; z=y[offset[q]:offset[q+1]]
            expected=float(z[s==max(s)].mean());assert abs(expected-actual[q])<1e-12;metric_checks+=1
        if p['model'].startswith(('exposure_','feedback_')):
            matrix=np.load(DATA/f"fresh_d{p['delay']}_X.npy",mmap_mode='r')
            chosen=rng.integers(0,len(y),size=10000);size=20 if p['model'].startswith('exposure') else 38
            reloaded=np.zeros(len(chosen))
            for seed in [42,43,44]:
                path=ROOT/f"models/ebnerd-top1-feedback/final-{p['model']}-s{seed}/model.txt"
                booster=lgb.Booster(model_file=str(path))
                reloaded+=booster.predict(np.array(matrix[chosen,:size]),num_threads=1)/3
            np.testing.assert_array_equal(reloaded,score[chosen]);reload_values+=len(chosen)
release=json.loads((OUT/'frozen_release.json').read_text())
assert all(r['fit_labels_before']=='2023-05-25 05:00:00' for r in release['models'])
selection=json.loads((OUT/'selection.json').read_text())
assert all(r['fit_labels_before']<('2023-05-21T07:00:00' if r['fold']=='early' else '2023-05-23T07:00:00') for r in selection['development_runs'])
report={'verified_utc':datetime.now(timezone.utc).isoformat(),'independent_reconstruction':checks,
    'response_feature_values_checked':verified,'max_absolute_error':max_error,'independent_query_metrics_checked':metric_checks,
    'reloaded_single_thread_predictions_exact':reload_values,
    'fit_label_embargo_verified':True,'release_sha256':sha256(OUT/'frozen_release.json'),
    'scope':'Event-time calculations and artifact boundaries verified; source ingestion timestamps unavailable.'}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
