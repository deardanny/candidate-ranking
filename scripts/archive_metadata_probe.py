"""Bounded read-only public-URL snapshot search; no user data leaves the machine.

This is an availability probe, not historical metadata certification. Public HTML
does not reproduce EB-NeRD's proprietary topic/entity extraction pipeline.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/ebnerd-metadata-audit'
OUT=ROOT/'reports/ebnerd-metadata-audit'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fetch',action='store_true')
args=parser.parse_args()
sample=DATA/'archive_probe_sample.json'
if not args.fetch:
    articles=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/articles.parquet',columns=['article_id','url','published_time','last_modified_time'])
    requests=pd.read_parquet(ROOT/'data/processed/ebnerd-v1/validation_requests.parquet')
    first={}; count=Counter()
    for row in requests.itertuples(index=False):
        for article in row.article_ids_inview:
            item=int(article); first.setdefault(item,row.impression_time); count[item]+=1
    articles=articles[articles.article_id.isin(first)].copy()
    articles['views']=articles.article_id.map(count)
    articles['later_modified']=articles.last_modified_time>=pd.Timestamp('2023-06-30')
    selected=[]
    for flag in [False,True]:
        top=articles[articles.later_modified==flag].sort_values(['views','article_id'],ascending=[False,True]).head(6)
        for row in top.itertuples(index=False):
            selected.append({'article_id':int(row.article_id),'url':row.url,'published_time':str(row.published_time),
                             'first_impression':str(first[row.article_id]),'later_modified':flag})
    sample.write_text(json.dumps(selected,indent=2)+'\n')
    (OUT/'archive_probe_protocol.json').write_text(json.dumps({
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection':'Top six candidate-exposure counts per last-modified cohort, label-independent; tie by article_id.',
        'sample_size':len(selected),'endpoint':'https://archive.org/wayback/available',
        'query_time':'First evaluation-week impression of each item; do not accept a later archive snapshot.',
        'interpretation':'Missing/late archives cannot certify field stability. HTML snapshots cannot establish proprietary tag generation times.',
        'max_requests':12,'per_request_timeout_seconds':15},indent=2)+'\n')
    print('Prepared',len(selected),'public URLs, no user IDs.')
else:
    records=[]
    for row in json.loads(sample.read_text()):
        timestamp=pd.Timestamp(row['first_impression']).strftime('%Y%m%d%H%M%S')
        url='https://archive.org/wayback/available?'+urllib.parse.urlencode({'url':row['url'],'timestamp':timestamp})
        result={**row,'requested_timestamp':timestamp}
        try:
            with urllib.request.urlopen(url,timeout=15) as response:
                payload=json.load(response)
            result['response']=payload
            closest=payload.get('archived_snapshots',{}).get('closest',{})
            stamp=closest.get('timestamp','')
            result['has_accessible_snapshot']=bool(closest.get('available'))
            result['at_or_before_first_impression']=bool(stamp and stamp<=timestamp and closest.get('available'))
        except Exception as exc:
            result['error_type']=type(exc).__name__; result['error']=str(exc)
        records.append(result)
        (DATA/'archive_probe_responses.json').write_text(json.dumps(records,indent=2)+'\n')
        print(len(records),'/',len(json.loads(sample.read_text())),result.get('error_type','ok'),flush=True)
    summary={'attempted':len(records),'successful_api_responses':sum('response' in r for r in records),
             'accessible_closest_snapshots':sum(r.get('has_accessible_snapshot',False) for r in records),
             'at_or_before_first_impression':sum(r.get('at_or_before_first_impression',False) for r in records),
             'errors':dict(Counter(r['error_type'] for r in records if 'error_type' in r)),
             'completed_at_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'Availability probe only; no reconstructed point-in-time metadata or proprietary extraction versions.'}
    (OUT/'archive_probe.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
