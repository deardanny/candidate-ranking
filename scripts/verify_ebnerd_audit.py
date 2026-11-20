"""Independent exploded-table counts, split integrity and publication-time sensitivity."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.download_ebnerd import digest


def main():
    raw=ROOT/'data/raw/ebnerd-small/extracted'
    out=ROOT/'reports/ebnerd-audit'
    report=json.loads((out/'metrics.json').read_text())
    manifest=json.loads((ROOT/'data/source/ebnerd/manifest.json').read_text())
    checks={}
    checks['download_hashes']=all(digest(ROOT/f['path'])==f['sha256'] for f in manifest['files'])
    a=pd.read_parquet(raw/'articles.parquet',columns=['article_id','published_time','last_modified_time'])
    b={s:pd.read_parquet(raw/s/'behaviors.parquet',columns=['impression_id','user_id','impression_time',
         'article_ids_inview','article_ids_clicked']) for s in ['train','validation']}
    h=pd.read_parquet(raw/'train/history.parquet',columns=['article_id_fixed'])
    known=set(b['train'].article_ids_inview.explode()) | set(b['train'].article_ids_clicked.explode()) | set(h.article_id_fixed.explode())
    assignments=pd.read_parquet(ROOT/'data/processed/ebnerd-audit-v1/request_splits.parquet')
    checks['one_assignment_per_request']=len(assignments)==sum(map(len,b.values())) and not assignments.impression_id.duplicated().any()
    checks['assignment_hash']=digest(ROOT/'data/processed/ebnerd-audit-v1/request_splits.parquet')==report['assignment_sha256']
    ranges=assignments.groupby('split').impression_time.agg(['min','max'])
    checks['strict_three_way_time_order']=bool(ranges.loc['train','max']<ranges.loc['validation','min'] and ranges.loc['validation','max']<ranges.loc['test','min'])
    extra={}
    for source,frame in b.items():
        expanded=frame[['impression_id','user_id','impression_time','article_ids_inview']].explode('article_ids_inview').rename(columns={'article_ids_inview':'article_id'})
        clicks=frame[['impression_id','article_ids_clicked']].explode('article_ids_clicked').drop_duplicates().rename(columns={'article_ids_clicked':'article_id'})
        expanded=expanded.merge(clicks.assign(clicked=True),on=['impression_id','article_id'],how='left',validate='one_to_one')
        expanded['clicked']=expanded.clicked.eq(True)
        expanded=expanded.merge(a,on='article_id',validate='many_to_one')
        expected=report['official_splits'][source]
        checks[source+'_candidate_rows']=len(expanded)==expected['candidate_rows']
        checks[source+'_clicked_rows']=int(expanded.clicked.sum())==expected['clicked_candidate_rows']
        checks[source+'_future_modified_count']=int((expanded.last_modified_time>expanded.impression_time).sum())==expected['modified_after_request_candidates']
        invalid=expanded.published_time>expanded.impression_time
        bad_requests=set(expanded.loc[invalid,'impression_id'])
        clean=expanded.loc[~expanded.impression_id.isin(bad_requests)].copy()
        extra[source]={'future_publication_requests':len(bad_requests),
            'future_publication_articles':int(expanded.loc[invalid,'article_id'].nunique()),
            'future_publication_max_gap_seconds':float((expanded.loc[invalid,'published_time']-expanded.loc[invalid,'impression_time']).dt.total_seconds().max()),
            'after_quarantining_whole_requests':{'requests':int(clean.impression_id.nunique()),
                'candidates':len(clean),'clicks':int(clean.clicked.sum())}}
        if source=='train':
            clean_known=set(clean.article_id) | set(h.article_id_fixed.explode())
        if source=='validation':
            expanded['cold']=~expanded.article_id.isin(known)
            c=expanded.loc[expanded.cold]
            target=report['temporal_splits']['test']['cold_cohorts']['unseen_training_inputs']
            checks['independent_cold_counts']=all([
                len(c)==target['candidate_rows'],int(c.clicked.sum())==target['clicked_candidate_rows'],
                c.article_id.nunique()==target['items'],c.impression_id.nunique()==target['requests'],
                c.loc[c.clicked,'impression_id'].nunique()==target['clicked_requests']])
            clicked=c.loc[c.clicked].copy()
            clicked['source_day_start_07']=(clicked.impression_time-pd.Timedelta(hours=7)).dt.strftime('%Y-%m-%d')
            extra['cold_clicks_per_day']=clicked.groupby('source_day_start_07').agg(
                clicked_candidates=('article_id','size'),items=('article_id','nunique'),users=('user_id','nunique')).to_dict('index')
            distribution=clicked.article_id.value_counts()
            extra['cold_clicked_distinct_items']=len(distribution)
            extra['cold_click_top5_items_share']=float(distribution.head(5).sum()/len(clicked))
            extra['test_users_unseen_in_train']=len(set(frame.user_id)-set(b['train'].user_id))
            clean_cold=clean.loc[~clean.article_id.isin(clean_known)]
            extra[source]['after_quarantining_whole_requests']['cold_items']=int(clean_cold.article_id.nunique())
            extra[source]['after_quarantining_whole_requests']['cold_clicked_candidates']=int(clean_cold.clicked.sum())
            extra[source]['after_quarantining_whole_requests']['definition']='Training and evaluation publication-anomaly requests quarantined; train history retained.'
        del expanded
    extra['article_modification_range']={k:str(v) for k,v in a.last_modified_time.agg(['min','max']).items()}
    extra['articles_modified_on_2023_06_29']=int(a.last_modified_time.dt.strftime('%Y-%m-%d').eq('2023-06-29').sum())
    checks['histories_pass']=all(all(x[k]==0 for k in ['duplicate_user_rows','unaligned_rows','null_timestamp_events',
        'not_strictly_before_first_request','unknown_catalog_events','request_users_without_history']) for x in report['histories'].values())
    verification={'passed':all(checks.values()),'checks':checks,'sensitivity':extra,
        'note':'Tests cover audit mechanics; this does not certify article snapshots or online causal effects.',
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/'scripts/download_ebnerd.py',ROOT/'scripts/audit_ebnerd.py',
            ROOT/'scripts/verify_ebnerd_audit.py',ROOT/'src/fairjob_ranker/ebnerd/audit.py',
            ROOT/'tests/test_ebnerd_audit.py',out/'metrics.json']}}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    if not verification['passed']: raise SystemExit('Audit verification failed')


if __name__=='__main__': main()
