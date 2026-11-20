"""Independently check frozen artifacts, cached features and heldout ranking counts."""
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score,roc_auc_score
from fairjob_ranker.io import sha256
from fairjob_ranker.ebnerd.features import FeatureState
from fairjob_ranker.ebnerd.bundle import Bundle

ROOT=Path(__file__).resolve().parents[1]; data=ROOT/'data/processed/ebnerd-v1'; report=ROOT/'reports/ebnerd-experiment'
frozen=json.loads((report/'frozen_release.json').read_text()); final=json.loads((report/'final_metrics.json').read_text())
checks={'frozen_release_digest':sha256(report/'frozen_release.json')==final['frozen_release_sha256'],
    'selection_unchanged':sha256(report/'selection.json')==frozen['selection_sha256'],
    'frozen_feature_and_evaluation_code':all(sha256(ROOT/n)==d for n,d in frozen['source_sha256'].items())}
state=FeatureState.load(data/'validation_state'); trainstate=FeatureState.load(data/'train_state')
frame=pd.read_parquet(data/'validation_requests.parquet'); train=pd.read_parquet(data/'train_requests.parquet')
checks['time_and_request_disjointness']=train.impression_time.max()<frame.impression_time.min() and not set(train.impression_id)&set(frame.impression_id)
x=np.load(data/'test_X.npy',mmap_mode='r'); y=np.load(data/'test_y.npy'); items=np.load(data/'test_items.npy'); groups=np.load(data/'test_groups.npy')
offsets=np.r_[0,np.cumsum(groups)]; selected=np.random.default_rng(123).choice(len(frame),1000,replace=False)
feature_error=0.
for index in selected:
    row=frame.iloc[int(index)]; fresh=state.matrix(row.user_id,row.impression_time,row.device_type,row.article_ids_inview)
    feature_error=max(feature_error,float(np.max(np.abs(fresh-x[offsets[index]:offsets[index+1]]))))
checks['heldout_feature_rebuild']=feature_error==0
trainx=np.load(data/'train_X.npy',mmap_mode='r'); trainoffsets=np.r_[0,np.cumsum(np.load(data/'train_groups.npy'))]
checks['prepared_features_unchanged_by_timestamp_validation_hardening']=True
for index in np.random.default_rng(321).choice(len(train),500,replace=False):
    row=train.iloc[int(index)]
    if not np.array_equal(trainstate.matrix(row.user_id,row.impression_time,row.device_type,row.article_ids_inview),trainx[trainoffsets[index]:trainoffsets[index+1]]): checks['prepared_features_unchanged_by_timestamp_validation_hardening']=False
model_error=0.; independent_error=0.; diagnostic={}
for variant,record in frozen['refit'].items():
    directory=ROOT/record['bundle']; bundle=Bundle(directory)
    checks[variant+'_artifact_digest']=sha256(directory/'metadata.json')==record['metadata_sha256']
    saved=np.load(data/('test-'+variant+'_scores.npy'))
    predicted=bundle.predict(x); model_error=max(model_error,float(np.abs(predicted-saved).max()))
    maxscore=np.maximum.reduceat(saved,offsets[:-1]); top=saved==np.repeat(maxscore,groups)
    weights=top/np.repeat(np.add.reduceat(top,offsets[:-1]),groups)
    hit=np.add.reduceat(weights*y,offsets[:-1]); cold=~np.isin(items,list(bundle.seen))
    coldhit=np.add.reduceat(weights*y*cold,offsets[:-1])
    checks[variant+'_independent_hit1']=bool(np.isclose(hit.mean(),final['metrics'][variant]['hit1'],atol=1e-12))
    checks[variant+'_independent_cold_hit1']=bool(np.isclose(coldhit.mean(),final['metrics'][variant]['cold_hit1'],atol=1e-12))
    expected=pd.read_parquet(data/('test-'+variant+'_outcomes.parquet'))
    for index in selected[:250]:
        lo,hi=offsets[index:index+2]
        ndcg=ndcg_score([y[lo:hi]],[saved[lo:hi]],k=5); auc=roc_auc_score(y[lo:hi],saved[lo:hi])
        independent_error=max(independent_error,abs(ndcg-expected.ndcg5.iloc[index]),abs(auc-expected.auc.iloc[index]))
    if variant in ['personalized','metadata']:
        positives=pd.DataFrame({'article_id':items[(y==1)&cold], 'capture':weights[(y==1)&cold]})
        peritem=positives.groupby('article_id').capture.mean()
        diagnostic[variant]={'cold_clicked_items':len(peritem),'cold_item_macro_capture':float(peritem.mean()),
            'cold_clicked_item_capture':peritem.to_dict()}
        qid=np.repeat(np.arange(len(groups)),groups)
        order=np.lexsort((items,-saved,qid))
        positions=np.arange(len(y))-np.repeat(offsets[:-1],groups)+1
        discounts=np.where(positions<=5,1/np.log2(positions+1),0)
        ideal=np.cumsum(1/np.log2(np.arange(2,7)))[np.minimum(np.add.reduceat(y,offsets[:-1]),5)-1]
        diagnostic[variant]['service_id_tie_ndcg5']=float((np.add.reduceat(y[order]*discounts,offsets[:-1])/ideal).mean())
        diagnostic[variant]['service_id_tie_hit1']=float(y[order[offsets[:-1]]].mean())
checks['all_six_reloaded_models_match']=model_error<1e-12
checks['independent_sklearn_ndcg_auc']=independent_error<1e-12
metadata=diagnostic['metadata'].pop('cold_clicked_item_capture'); personal=diagnostic['personalized'].pop('cold_clicked_item_capture')
deltas=np.array([personal[k]-metadata[k] for k in sorted(personal)])
diagnostic['item_macro_delta']=float(deltas.mean()); diagnostic['items_improved']=int((deltas>1e-12).sum()); diagnostic['items_worsened']=int((deltas < -1e-12).sum())
connection=sqlite3.connect('file:'+str(ROOT/'runs/v2/mlflow.db')+'?mode=ro',uri=True)
checks['fairjob_champion_untouched']=connection.execute("select version from registered_model_aliases where name='fairjob-v2' and alias='champion'").fetchone()[0]==13
connection.close()
verification={'passed':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()},'maximum_feature_error':feature_error,
    'maximum_reloaded_prediction_error':model_error,'maximum_independent_metric_error':independent_error,
    'rebuilt_train_requests':500,'rebuilt_test_requests':1000,'independent_metric_requests_per_model':250,
    'diagnostics':diagnostic,'source_sha256':sha256(Path(__file__))}
(report/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
if not verification['passed']: raise SystemExit('Experiment verification failed')
