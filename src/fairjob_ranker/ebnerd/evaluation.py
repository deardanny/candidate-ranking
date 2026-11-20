"""Request metrics with exact expected tie handling and paired user uncertainty."""
from functools import lru_cache
from math import comb
import numpy as np
import pandas as pd

METRICS=['ndcg5','hit1','mrr','auc','cold_hit1']
DISCOUNT=np.r_[0.,1/np.log2(np.arange(2,130))]
PREFIX=np.cumsum(DISCOUNT)


@lru_cache(maxsize=16384)
def first_reciprocal(start,size,positives):
    total=comb(size,positives)
    return sum(comb(size-k,positives-1)/total/(start+k) for k in range(1,size-positives+2))


def outcomes(labels,scores,groups,items=None,cold_items=None):
    labels=np.asarray(labels); scores=np.asarray(scores)
    if len(labels)!=len(scores) or sum(groups)!=len(labels) or not np.isfinite(scores).all(): raise ValueError('Invalid evaluation arrays')
    rows=np.zeros((len(groups),5)); offset=0
    cold=np.isin(items,list(cold_items)) if cold_items is not None else np.zeros(len(labels),dtype=bool)
    for j,n in enumerate(groups):
        end=offset+int(n); y=labels[offset:end].astype(bool); s=scores[offset:end]; c=int(y.sum())
        if c==0 or c==n: raise ValueError('Ranking requires mixed labels')
        best=s.max(); top=s==best
        rows[j,1]=y[top].mean(); rows[j,4]=(y & cold[offset:end])[top].mean()
        dcg=0.; auc=0.
        for score in s[y]:
            greater=int((s>score).sum()); equal=int((s==score).sum())
            lo=min(5,greater); hi=min(5,greater+equal)
            dcg+=(PREFIX[hi]-PREFIX[lo])/equal
            auc+=((s[~y]<score).sum()+.5*(s[~y]==score).sum())/(c*(n-c))
        rows[j,0]=dcg/PREFIX[min(c,5)]; rows[j,3]=auc
        highest=s[y].max(); start=int((s>highest).sum()); tied=s==highest
        rows[j,2]=first_reciprocal(start,int(tied.sum()),int((y&tied).sum()))
        offset=end
    return pd.DataFrame(rows,columns=METRICS)


def paired_interval(users,differences,repeats=2000,seed=20261119):
    frame=pd.DataFrame({'user':np.asarray(users),'delta':np.asarray(differences)})
    agg=frame.groupby('user').delta.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(seed); draws=np.empty(repeats)
    for start in range(0,repeats,50):
        ix=rng.integers(0,len(agg),size=(min(50,repeats-start),len(agg)))
        draws[start:start+len(ix)]=agg[ix,0].sum(axis=1)/agg[ix,1].sum(axis=1)
    return {'estimate':float(frame.delta.mean()),'lower':float(np.quantile(draws,.025)),
            'upper':float(np.quantile(draws,.975)),'users':len(agg),'requests':len(frame)}
