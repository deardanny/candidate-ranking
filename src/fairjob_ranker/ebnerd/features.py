"""Shared offline/online features from fixed, strictly preceding history snapshots.

Content/category features explicitly use the released (not point-in-time verified)
article snapshot. Publication-time features carry a separate stability assumption.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import sha256

BEHAVIOR = ['hour_sin','hour_cos','dayofweek','device','candidate_count',
            'history_pop','history_pop_recent','history_first_seen_age','history_item_unknown']
USER = ['user_history_size','user_last_event_age','user_candidate_clicks','user_candidate_last_age']
FRESH = ['publication_age','publication_age_rank','publication_age_gap']
META = ['category','premium','category_prior']
PERSONAL = ['category_affinity','recent_category_affinity','topic_affinity','topic_max_affinity',
            'entity_affinity','entity_max_affinity']
FEATURES = BEHAVIOR+USER+FRESH+META+PERSONAL
VARIANTS = {'behavior':BEHAVIOR, 'behavior_history':BEHAVIOR+USER,
            'freshness':BEHAVIOR+FRESH, 'metadata':BEHAVIOR+FRESH+META,
            'personalized':FEATURES,
            'personalized_no_freshness':BEHAVIOR+USER+META+PERSONAL}
ARTICLE_COLUMNS = ['article_id','published_time','last_modified_time','category','premium','topics','ner_clusters']


def seconds(value):
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is not None:
        raise ValueError('Use a nonmissing timestamp in the timezone-unspecified source clock')
    return int(t.value//1_000_000_000)


def log_age(now,then):
    return np.log1p(max(0.,now-then)/3600.) if then is not None else 0.


class FeatureState:
    def __init__(self, articles, history, available_from):
        self.available_from=seconds(available_from)
        if articles.article_id.duplicated().any() or history.user_id.duplicated().any():
            raise ValueError('Duplicate catalog/history keys')
        self.articles={int(r.article_id):{'published':seconds(r.published_time),
            'modified':seconds(r.last_modified_time),'category':int(r.category),'premium':int(r.premium),
            'topics':tuple(r.topics),'entities':tuple(r.ner_clusters)} for r in articles.itertuples(index=False)}
        self.users={}; self.pop=Counter(); self.recent_pop=Counter(); self.first={}; self.category_pop=Counter()
        self.latest_event=-1
        for r in history.itertuples(index=False):
            ids=list(map(int,r.article_id_fixed)); raw_ts=np.asarray(r.impression_time_fixed,dtype='datetime64[s]')
            if np.isnat(raw_ts).any(): raise ValueError('Missing history timestamp')
            ts=raw_ts.astype(np.int64)
            if len(ids)!=len(ts) or (len(ts) and (np.diff(ts)<0).any()): raise ValueError('History arrays not aligned/sorted')
            if len(ts) and (ts>=self.available_from).any(): raise ValueError('History is not strictly before snapshot availability')
            counts=Counter(ids); last={}; cats=Counter(); recent=Counter(); topics=Counter(); entities=Counter()
            for k,(i,t) in enumerate(zip(ids,ts)):
                if i not in self.articles: raise ValueError('History article absent from catalog')
                t=int(t); a=self.articles[i]; last[i]=t
                self.pop[i]+=1; self.first[i]=min(self.first.get(i,t),t)
                self.recent_pop[i]+=int(t>=self.available_from-86400)
                self.category_pop[a['category']]+=1; cats[a['category']]+=1
                if k>=len(ids)-20: recent[a['category']]+=1
                topics.update(set(a['topics'])); entities.update(set(a['entities']))
            latest=int(ts[-1]) if len(ts) else -1
            self.latest_event=max(self.latest_event,latest)
            self.users[int(r.user_id)]={'n':len(ids),'latest':latest,'counts':counts,'last':last,
                'categories':cats,'recent_categories':recent,'topics':topics,'entities':entities}
        self.total=max(1,sum(self.pop.values()))
        self.version=None; self.directory=None

    @classmethod
    def load(cls,directory):
        directory=Path(directory); meta=json.loads((directory/'state.json').read_text())
        for name,digest in meta['sha256'].items():
            if sha256(directory/name)!=digest: raise ValueError('State checksum mismatch')
        obj=cls(pd.read_parquet(directory/'articles.parquet'),pd.read_parquet(directory/'history.parquet'),meta['available_from'])
        obj.version=sha256(directory/'state.json'); obj.directory=directory
        return obj

    @staticmethod
    def save(directory,articles,history,available_from):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
        articles[ARTICLE_COLUMNS].to_parquet(directory/'articles.parquet',index=False)
        history[['user_id','article_id_fixed','impression_time_fixed']].to_parquet(directory/'history.parquet',index=False)
        meta={'schema':'ebnerd-state-v1','available_from':str(available_from),
            'content_snapshot_point_in_time_verified':False,
            'sha256':{n:sha256(directory/n) for n in ['articles.parquet','history.parquet']}}
        (directory/'state.json').write_text(json.dumps(meta,indent=2)+'\n')

    def matrix(self, user_id, request_time, device, candidate_ids, *, history_user_id=None):
        now=seconds(request_time)
        if now<self.available_from or self.latest_event>=now:
            raise ValueError('Future history snapshot for request')
        ids=list(map(int,candidate_ids))
        if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty/duplicate candidates')
        if any(i not in self.articles for i in ids): raise ValueError('Missing catalog candidate')
        if any(self.articles[i]['published']>now for i in ids): raise ValueError('Unpublished candidate')
        user=self.users.get(int(user_id if history_user_id is None else history_user_id))
        n=user['n'] if user else 0; denom=max(1,n)
        t=pd.Timestamp(request_time); hour=t.hour+t.minute/60
        context=[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),t.dayofweek,int(device),len(ids)]
        ages=np.array([(now-self.articles[i]['published'])/3600 for i in ids])
        age_ranks=np.array([(ages<a).sum()/max(1,len(ids)-1) for a in ages])
        rows=[]
        for k,i in enumerate(ids):
            a=self.articles[i]; cat=a['category']
            personal=[]
            for name,keys in [('topics',a['topics']),('entities',a['entities'])]:
                values=[user[name].get(x,0)/denom for x in set(keys)] if user else []
                personal.extend([float(np.mean(values)) if values else 0.,max(values,default=0.)])
            rows.append(context+[
                np.log1p(self.pop[i]),np.log1p(self.recent_pop[i]),log_age(now,self.first.get(i)),int(i not in self.first),
                np.log1p(n),log_age(now,user['latest']) if user and n else 0.,
                np.log1p(user['counts'].get(i,0)) if user else 0.,log_age(now,user['last'].get(i)) if user else 0.,
                np.log1p(ages[k]),age_ranks[k],np.log1p(ages[k]-ages.min()),
                cat,a['premium'],self.category_pop[cat]/self.total,
                user['categories'].get(cat,0)/denom if user else 0.,
                user['recent_categories'].get(cat,0)/max(1,min(n,20)) if user else 0.,*personal])
        result=np.asarray(rows,dtype=np.float32)
        if result.shape!=(len(ids),len(FEATURES)) or not np.isfinite(result).all(): raise ValueError('Invalid feature matrix')
        return result


def clean_behaviors(path,articles):
    raw=pd.read_parquet(path,columns=['impression_id','user_id','impression_time','device_type','article_ids_inview','article_ids_clicked'])
    pub=articles.set_index('article_id').published_time.to_dict(); rejected=[]; keep=[]
    for r in raw.itertuples(index=False):
        ids=list(map(int,r.article_ids_inview)); labels=set(map(int,r.article_ids_clicked))
        reason=None
        if not ids or len(ids)!=len(set(ids)) or not labels<=set(ids): reason='invalid_candidate_list'
        elif any(i not in pub or pd.isna(pub[i]) or pub[i]>r.impression_time for i in ids): reason='publication_time'
        if reason: rejected.append({'impression_id':int(r.impression_id),'reason':reason})
        keep.append(reason is None)
    result=raw.loc[keep].sort_values(['impression_time','impression_id'],kind='stable').reset_index(drop=True)
    if result.impression_id.duplicated().any(): raise ValueError('Duplicate request IDs')
    return result,pd.DataFrame(rejected,columns=['impression_id','reason'])


def build_matrix(frame,state,progress=None):
    groups=frame.article_ids_inview.map(len).to_numpy(dtype=np.int32)
    x=np.empty((groups.sum(),len(FEATURES)),dtype=np.float32); labels=np.zeros(len(x),dtype=np.int8)
    items=np.empty(len(x),dtype=np.int32); offset=0
    for j,r in enumerate(frame.itertuples(index=False)):
        ids=list(map(int,r.article_ids_inview)); end=offset+len(ids); clicked=set(map(int,r.article_ids_clicked))
        x[offset:end]=state.matrix(r.user_id,r.impression_time,r.device_type,ids)
        labels[offset:end]=[int(i in clicked) for i in ids]; items[offset:end]=ids; offset=end
        if progress and (j+1)%50000==0: progress(f'features {j+1}/{len(frame)}')
    return x,labels,items,groups
