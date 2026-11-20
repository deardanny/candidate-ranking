"""One-time evaluation: join fresh click labels only after predictions are frozen."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fairjob_ranker.ebnerd.feedback import hit1
from fairjob_ranker.ebnerd.evaluation import paired_interval
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-top1-feedback'; OUT=ROOT/'reports/ebnerd-top1-feedback'; BASE=ROOT/'data/processed/ebnerd-metadata-audit'
if (OUT/'final_results.json').exists():raise SystemExit('Already evaluated: no test-driven rerun/tuning')
freeze=json.loads((OUT/'predictions_frozen.json').read_text())
assert not freeze['fresh_labels_loaded']
assert freeze['release_sha256']==sha256(OUT/'frozen_release.json')
for path,digest in freeze['pipeline_source_sha256'].items(): assert sha256(ROOT/path)==digest
assert freeze['evaluation_clarifications_sha256']==sha256(OUT/'evaluation_clarifications.json')
for p in freeze['predictions']:
    assert p['sha256']==sha256(DATA/f"fresh_d{p['delay']}_{p['model']}_scores.npy")
frame=pd.read_parquet(DATA/'fresh_requests.parquet'); selected_ids=set(map(int,frame.impression_id))
assert freeze['requests_sha256']==sha256(DATA/'fresh_requests.parquet')
parts=[]
for batch in pq.ParquetFile(ROOT/'data/raw/ebnerd-large/validation/behaviors.parquet').iter_batches(batch_size=65536,columns=['impression_id','article_ids_clicked']):
    block=batch.to_pandas();take=block[block.impression_id.isin(selected_ids)]
    if len(take):parts.append(take)
labels=pd.concat(parts,ignore_index=True).set_index('impression_id')
assert not labels.index.duplicated().any() and len(labels)==len(frame)
labels=labels.loc[frame.impression_id].article_ids_clicked
groups=np.load(DATA/'fresh_groups.npy'); items=np.load(DATA/'fresh_items.npy'); starts=np.r_[0,np.cumsum(groups)[:-1]]
y=np.concatenate([np.isin(ids,list(set(map(int,clicked)))) for ids,clicked in zip(frame.article_ids_inview,labels)]).astype('int8')
assert all(set(map(int,c))<=set(map(int,i)) for c,i in zip(labels,frame.article_ids_inview))
np.save(DATA/'fresh_y.npy',y)
fit_end=json.loads((OUT/'frozen_release.json').read_text())['models'][0]['fit_candidate_rows']
seen=set(map(int,np.load(BASE/'strict_train_items.npy')[:fit_end]))|set(json.loads((BASE/'strict_history_ids.json').read_text()))
cold=~np.isin(items,list(seen)); x=np.load(DATA/'fresh_d3600_X.npy',mmap_mode='r')
zero=cold&(x[:,20]==0); no_signal=zero&(x[:,5]==0)&(x[:,11]==0)
any_in=lambda a:np.logical_or.reduceat(a,starts)
cold_target=any_in(cold&(y==1))
masks={'overall':np.ones(len(frame),dtype=bool),'cold_target':cold_target,
    'all_cold':~any_in(~cold),'zero_feedback_target':any_in(zero&(y==1)),
    'adapted_cold_target':any_in(cold&(x[:,20]>0)&(y==1)),
    'zero_signal_target':any_in(no_signal&(y==1)),
    'all_candidates_zero_signal':~any_in(~no_signal)}
selection=json.loads((OUT/'selection.json').read_text()); chosen=selection['selected']
users=frame.user_id.to_numpy(); results={}; predictions={}; segment_values={}
segment_labels={'overall':y,'cold_target':y*cold,'all_cold':y,
    'zero_feedback_target':y*zero,'adapted_cold_target':y*cold*(x[:,20]>0),
    'zero_signal_target':y*no_signal,'all_candidates_zero_signal':y}
prior_n=np.rint(np.expm1(x[:,20])).astype('int64')
for name,lo,hi in [('cold_1to4_exposures',1,5),('cold_5to19_exposures',5,20),('cold_20plus_exposures',20,np.iinfo('int64').max)]:
    target=y*cold*(prior_n>=lo)*(prior_n<hi)
    masks[name]=any_in(target==1);segment_labels[name]=target
for p in freeze['predictions']:
    key=f"d{p['delay']}_{p['model']}"; s=np.load(DATA/f"fresh_d{p['delay']}_{p['model']}_scores.npy")
    h=hit1(y,s,groups);predictions[key]=h
    segment_values[key]={name:hit1(segment_labels[name],s,groups) for name in masks}
    results[key]={name:{'requests':int(mask.sum()),'users':int(np.unique(users[mask]).size),
        'hit1':float(segment_values[key][name][mask].mean()) if mask.any() else None} for name,mask in masks.items()}
    np.save(DATA/(key+'_hit1.npy'),h)
    np.savez(DATA/(key+'_segment_hit1.npz'),**segment_values[key])
    print('EVAL',key,results[key]['overall'],flush=True)
results['uniform']={name:{'requests':int(mask.sum()),'hit1':float((np.add.reduceat(segment_labels[name],starts)/groups)[mask].mean()) if mask.any() else None} for name,mask in masks.items()}
comparisons={}
for delay in [3600,86400,'frozen']:
    a=f'd{delay}_{chosen}'
    baselines=['exposure_top1','ctr_1h','ctr_24h']+(['exposure_ndcg'] if delay in [3600,'frozen'] else [])
    for baseline in baselines:
        b=f'd{delay}_{baseline}'; pair=a+'-minus-'+b
        comparisons[pair]={}
        for segment in ['overall','cold_target','all_cold','zero_feedback_target','adapted_cold_target','zero_signal_target']:
            mask=masks[segment]
            if mask.any():comparisons[pair][segment]=paired_interval(users[mask],(segment_values[a][segment]-segment_values[b][segment])[mask],repeats=2000)
        print('CI',pair,comparisons[pair]['overall'],flush=True)
dev_deltas={}
for fold in ['early','late']:
    dev_deltas[fold]=float(np.mean([r['hit1'] for r in selection['development_runs'] if r['arm']==chosen and r['fold']==fold])-np.mean([r['hit1'] for r in selection['development_runs'] if r['arm']=='exposure_top1' and r['fold']==fold]))
gate=all(v>0 for v in dev_deltas.values()) and all(comparisons[f'd3600_{chosen}-minus-d3600_{b}'][seg]['lower']>0 for b in ['exposure_top1','ctr_1h','ctr_24h'] for seg in ['overall','cold_target'])
strong_gate=gate and all(comparisons[f'd3600_{chosen}-minus-d3600_exposure_ndcg'][seg]['lower']>0 for seg in ['overall','cold_target'])
days=frame.impression_time.dt.strftime('%Y-%m-%d').to_numpy()
daily={day:{k:float(v[days==day].mean()) for k,v in predictions.items()} for day in sorted(set(days))}
zero_counts=np.add.reduceat(no_signal,starts);zero_positive_counts=np.add.reduceat(y*no_signal,starts)
zero_mask=masks['zero_signal_target']; identical_zero_groups=0
for q in np.flatnonzero(zero_counts>1):
    end=starts[q]+groups[q]; candidate_rows=x[starts[q]:end][no_signal[starts[q]:end]]
    assert np.array_equal(candidate_rows,np.broadcast_to(candidate_rows[0],candidate_rows.shape))
    identical_zero_groups+=1
zero_bound=float((zero_positive_counts[zero_mask]/zero_counts[zero_mask]).mean()) if zero_mask.any() else None
report={'evaluated_utc':datetime.now(timezone.utc).isoformat(),'selection':chosen,'new_users_same_calendar_period':True,
    'predictions_frozen_sha256':sha256(OUT/'predictions_frozen.json'),'protocol_sha256':sha256(OUT/'protocol.json'),
    'cohort':json.loads((OUT/'fresh_feature_preparation.json').read_text()),'positive_requests':int(any_in(y==1).sum()),
    'model_cold_unique_items':int(len(set(items[cold]))),'metrics':results,'paired_user_bootstrap':comparisons,
    'development_selected_minus_exposure_top1':dev_deltas,'predeclared_success_gate_passed':bool(gate),
    'also_beats_exposure_ndcg_overall_and_cold':bool(strong_gate),'daily_hit1':daily,
    'zero_signal_identical_candidate_rows':int(masks['all_candidates_zero_signal'].sum()),
    'zero_signal_indistinguishability':{'queries_with_multiple_identical_zero_signal_candidates':identical_zero_groups,
        'zero_signal_target_hit1_oracle_upper_bound':zero_bound,
        'scope':'For this 38-feature 1h representation under uniform tie evaluation. Every zero-signal candidate within a request has identical input. Bound assumes always choosing that tied group; it is not a deployable result or a bound on other feature sets.'},
    'cold_metric_definition':'On cold_target requests, Top1 must itself be a clicked cold item; choosing a clicked warm item is NOT credited. All warm and cold candidates still compete. Analogous rules apply to zero-feedback/adapted/zero-signal targets.',
    'frozen_control':'Same frozen models, but no reference request from 2023-05-25 05:00 or later can update features. All evaluation-week reference click labels are excluded from this control. This is a feature-staleness intervention, not a separately optimized frozen model.',
    'limits':['Feedback mature by one hour is an unobserved availability assumption; 24h delays only online features while keeping the same 1h-maturity trained model, not a full-system 24h availability guarantee.',
              'Fresh users are disjoint, but dates and reference feed were previously explored; confidence intervals condition on this feed and week.',
              'Cold means absent from model fitting candidates and original history, not newly published globally.',
              'An observed clicked item ranked first is not causal CTR/revenue lift; data contains clicked requests and logging-policy bias.',
              'Cold subgroups based on positive item status can overlap when multiple positives occur. All original candidates compete.']}
(OUT/'final_results.json').write_text(json.dumps(report,indent=2)+'\n')
print('FINAL GATE',gate,flush=True)
