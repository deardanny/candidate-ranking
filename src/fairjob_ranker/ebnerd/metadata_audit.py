"""Post-hoc metadata audit helpers; deliberately separate from frozen releases."""
from collections import Counter
import numpy as np
import pandas as pd

from .features import BEHAVIOR,USER,FRESH,FEATURES,seconds,log_age

STRICT_FEATURES=BEHAVIOR+USER
ARMS={
    'events_only':BEHAVIOR,
    'events_user':STRICT_FEATURES,
    'pub_user':BEHAVIOR+USER+FRESH,
    'pub_category':BEHAVIOR+FRESH+['category','category_prior'],
    'pub_category_user':BEHAVIOR+USER+FRESH+['category','category_prior','category_affinity','recent_category_affinity'],
    'pub_topic_user':BEHAVIOR+USER+FRESH+['topic_affinity','topic_max_affinity'],
    'pub_entity_user':BEHAVIOR+USER+FRESH+['entity_affinity','entity_max_affinity'],
}


class EventOnlyState:
    """This object has no article table, publication fields or content dependency."""
    def __init__(self,history,available_from):
        self.available_from=seconds(available_from); self.users={}; self.pop=Counter(); self.recent_pop=Counter(); self.first={}
        if history.user_id.duplicated().any(): raise ValueError('Duplicate history user')
        for row in history.itertuples(index=False):
            ids=list(map(int,row.article_id_fixed)); ts=np.asarray(row.impression_time_fixed,dtype='datetime64[s]')
            if len(ids)!=len(ts) or np.isnat(ts).any(): raise ValueError('Invalid history alignment/time')
            ts=ts.astype(np.int64)
            if len(ts) and (np.max(ts)>=self.available_from or (np.diff(ts)<0).any()): raise ValueError('Future/unsorted history')
            last={}
            for item,timestamp in zip(ids,ts):
                t=int(timestamp); last[item]=t; self.pop[item]+=1
                self.first[item]=min(self.first.get(item,t),t); self.recent_pop[item]+=int(t>=self.available_from-86400)
            self.users[int(row.user_id)]={'n':len(ids),'latest':int(ts[-1]) if len(ts) else None,'counts':Counter(ids),'last':last}

    def matrix(self,user_id,request_time,device,ids):
        now=seconds(request_time)
        if now<self.available_from: raise ValueError('Future history snapshot')
        ids=list(map(int,ids))
        if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty/duplicate candidates')
        t=pd.Timestamp(request_time); hour=t.hour+t.minute/60
        context=[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),t.dayofweek,int(device),len(ids)]
        user=self.users.get(int(user_id)); n=user['n'] if user else 0
        rows=[context+[np.log1p(self.pop[i]),np.log1p(self.recent_pop[i]),log_age(now,self.first.get(i)),int(i not in self.first),
            np.log1p(n),log_age(now,user['latest']) if user else 0.,np.log1p(user['counts'].get(i,0)) if user else 0.,
            log_age(now,user['last'].get(i)) if user else 0.] for i in ids]
        return np.array(rows,dtype=np.float32)


def structurally_valid(frame):
    """No catalog lookup or publication-time filtering, including implicit joins."""
    if frame.impression_id.duplicated().any() or frame.impression_time.isna().any(): raise ValueError('Bad request keys')
    for row in frame.itertuples(index=False):
        ids=set(map(int,row.article_ids_inview)); clicked=set(map(int,row.article_ids_clicked))
        if len(ids)!=len(row.article_ids_inview) or not clicked<=ids or not 0<len(clicked)<len(ids):
            raise ValueError('Invalid recorded ranking request')
    return frame.sort_values(['impression_time','impression_id'],kind='stable').reset_index(drop=True)


def build_event_matrix(frame,state):
    groups=frame.article_ids_inview.map(len).to_numpy(dtype=np.int32)
    x=np.empty((groups.sum(),len(STRICT_FEATURES)),dtype=np.float32); y=np.zeros(len(x),dtype=np.int8); items=np.empty(len(x),dtype=np.int32)
    start=0
    for j,row in enumerate(frame.itertuples(index=False)):
        ids=list(map(int,row.article_ids_inview)); end=start+len(ids); clicked=set(map(int,row.article_ids_clicked))
        x[start:end]=state.matrix(row.user_id,row.impression_time,row.device_type,ids)
        y[start:end]=[i in clicked for i in ids]; items[start:end]=ids; start=end
        if (j+1)%100000==0: print('strict events',j+1,flush=True)
    return x,y,items,groups


def mask_columns(matrix,columns):
    changed=np.array(matrix,copy=True)
    changed[:,[FEATURES.index(c) for c in columns]]=np.nan
    return changed


def permute_metadata(articles,seed):
    """Swap metadata coherently across article IDs within publication-date strata.

    Publication and modification timestamps stay attached to original IDs. This is
    synthetic information destruction, not a recovered historical snapshot.
    """
    result=articles.copy(deep=True); rng=np.random.default_rng(seed)
    fields=['category','premium','topics','ner_clusters']; positions=np.arange(len(articles))
    mapping=positions.copy()
    for indexes in articles.groupby(articles.published_time.dt.strftime('%Y-%m-%d')).indices.values():
        indexes=np.asarray(indexes); mapping[indexes]=rng.permutation(indexes)
    for column in fields: result[column]=articles.iloc[mapping][column].to_numpy()
    return result,{'seed':seed,'articles':len(articles),'remapped_ids':int((mapping!=positions).sum()),'mapping':mapping}
