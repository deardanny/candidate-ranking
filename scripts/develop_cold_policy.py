"""A bounded follow-up after the first fresh-user zero-signal failure.

Tune only on the old two development folds. A second user hash bucket is reserved
for confirmation iff the predeclared development gate passes. No model refit.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.feedback import hit1
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-cold-policy'
BASE=ROOT/'data/processed/ebnerd-metadata-audit'; DATA=ROOT/'data/processed/ebnerd-top1-feedback'; MODELS=ROOT/'models/ebnerd-top1-feedback'
if (OUT/'development.json').exists():raise SystemExit('Development policy already selected')
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'reason':'First fresh-user experiment improved aggregate cold but hurt zero-signal positives. Test explicit allocation to unknown candidates without pretending to identify them.',
 'grid':[0,.02,.05,.1,.2,.3,.4,.5],
 'policy':'With probability alpha choose uniformly from zero-signal candidates if any; otherwise choose from tied highest feedback-model scores uniformly. Non-selected candidates keep model order.',
 'inputs':'Zero signal is exposure_count=0 AND fixed global click count=0 AND user click count=0. No labels determine routing.',
 'development_gate':'For BOTH development folds, overall Hit1, cold-target Hit1, and zero-signal-target Hit1 must each exceed the best of exposure_ndcg, exposure_top1, ctr_1h and ctr_24h by at least 0.25 percentage points. Select highest mean overall among eligible alphas.',
 'failure_rule':'If no alpha passes development, reject this policy family; do not open another confirmation cohort and do not tune on first fresh-user labels.',
 'confirmation':'If eligible: new users sha256 namespace as first study modulo20==1; exclude all small users; same week, same frozen base models, one-hour availability assumption. Freeze policy/predictions before labels.',
 'second_holdout_gate':'Selected policy minus every fixed baseline, including unmodified feedback model, positive overall/cold/zero-signal 95% paired-user CI would be Pareto improvement; otherwise report tradeoffs. Do not claim policy distinguishes identical new candidates.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
old=json.loads((ROOT/'reports/ebnerd-top1-feedback/protocol.json').read_text())
frame=pd.read_parquet(BASE/'strict_train_requests.parquet');x=np.load(DATA/'train_X.npy',mmap_mode='r');y=np.load(BASE/'strict_train_y.npy');ids=np.load(BASE/'strict_train_items.npy');g=np.load(BASE/'strict_train_groups.npy');offsets=np.r_[0,np.cumsum(g)];times=frame.impression_time.to_numpy()
history=set(json.loads((BASE/'strict_history_ids.json').read_text()));rows=[]
for fold in old['folds']:
    a=int(np.searchsorted(times,np.datetime64(fold['fit_end'])));b=int(np.searchsorted(times,np.datetime64(fold['end'])));nfit=int(np.searchsorted(times,np.datetime64(fold['fit_end'])-np.timedelta64(7200,'s')))
    left,right=offsets[a],offsets[b];xx=np.array(x[left:right]);yy=y[left:right];gg=g[a:b];starts=np.r_[0,np.cumsum(gg)[:-1]]
    cold=~np.isin(ids[left:right],list(set(map(int,ids[:offsets[nfit]]))|history));zero=(xx[:,20]==0)&(xx[:,5]==0)&(xx[:,11]==0)
    anyzero=np.logical_or.reduceat(zero,starts);nz=np.add.reduceat(zero,starts)
    targets={'overall':yy,'cold':yy*cold,'zero':yy*zero}
    masks={'overall':np.ones(len(gg),dtype=bool),'cold':np.logical_or.reduceat(targets['cold']>0,starts),'zero':np.logical_or.reduceat(targets['zero']>0,starts)}
    scores={'ctr_1h':xx[:,26],'ctr_24h':xx[:,30]}
    for arm in ['exposure_ndcg','exposure_top1','feedback_ndcg']:
        size=20 if arm.startswith('exposure') else 38;scores[arm]=np.zeros(len(yy))
        for seed in [42,43,44]:
            m=lgb.Booster(model_file=str(MODELS/f"{fold['name']}-{arm}-s{seed}"/'model.txt'))
            scores[arm]+=m.predict(xx[:,:size],num_threads=4)/3
    values={arm:{seg:hit1(ty,sc,gg) for seg,ty in targets.items()} for arm,sc in scores.items()}
    baseline_means={arm:{seg:float(v[masks[seg]].mean()) for seg,v in val.items()} for arm,val in values.items()}
    best={seg:max(baseline_means[arm][seg] for arm in ['exposure_ndcg','exposure_top1','ctr_1h','ctr_24h']) for seg in targets}
    for alpha in protocol['grid']:
        policy={}
        for seg,ty in targets.items():
            uniform_zero=np.divide(np.add.reduceat(ty*zero,starts),nz,out=np.zeros(len(gg)),where=nz>0)
            h=np.where(anyzero,(1-alpha)*values['feedback_ndcg'][seg]+alpha*uniform_zero,values['feedback_ndcg'][seg])
            policy[seg]=float(h[masks[seg]].mean())
        rows.append({'fold':fold['name'],'alpha':alpha,'metrics':policy,'best_fixed_baselines':best,
            'baseline_metrics':baseline_means,'pass':all(policy[seg]>=best[seg]+.0025 for seg in targets)})
    print('DEVELOPMENT',fold['name'],json.dumps(rows[-len(protocol['grid']):]),flush=True)
eligible=[alpha for alpha in protocol['grid'] if all(row['pass'] for row in rows if row['alpha']==alpha)]
selected=max(eligible,key=lambda alpha:np.mean([r['metrics']['overall'] for r in rows if r['alpha']==alpha])) if eligible else None
result={'completed_utc':datetime.now(timezone.utc).isoformat(),'selected_alpha':selected,'gate_passed':selected is not None,'results':rows,
 'protocol_sha256':sha256(OUT/'protocol.json'),'second_cohort_labels_read':False}
(OUT/'development.json').write_text(json.dumps(result,indent=2)+'\n');print('POLICY DEVELOPMENT GATE',selected,flush=True)
