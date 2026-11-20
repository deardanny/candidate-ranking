"""Inspect the three pre-impression public snapshots from the fixed URL probe.

Raw licensed metadata and archived HTML stay in ignored local data directories.
Only aggregates are emitted. Failure to retrieve a snapshot is not evidence that
an article did or did not change.
"""
from collections import Counter
from datetime import datetime,timezone
import argparse
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import urllib.request
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/ebnerd-metadata-audit'; OUT=ROOT/'reports/ebnerd-metadata-audit'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cached',action='store_true',help='Parse previously downloaded HTML, no network request.')
args=parser.parse_args()

class ExtractMetadata(HTMLParser):
    def __init__(self):
        super().__init__(); self.meta={}; self.blocks=[]; self.in_json=False; self.current=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='meta':
            key=attrs.get('property',attrs.get('name',''))
            if key and 'content' in attrs: self.meta[key]=attrs['content']
        if tag=='script' and attrs.get('type')=='application/ld+json': self.in_json=True; self.current=[]
    def handle_data(self,data):
        if self.in_json: self.current.append(data)
    def handle_endtag(self,tag):
        if tag=='script' and self.in_json:
            self.in_json=False
            try:self.blocks.append(json.loads(''.join(self.current)))
            except ValueError:pass

def flat(value):
    if isinstance(value,list):
        for v in value: yield from flat(v)
    elif isinstance(value,dict):
        yield value
        if '@graph' in value: yield from flat(value['@graph'])

def normalized(value): return ' '.join(str(value).split())

records=json.loads((DATA/'archive_probe_responses.json').read_text())
receipt_path=DATA/'archive_page_comparison.json'
prior={r['article_id']:r for r in json.loads(receipt_path.read_text())} if args.cached and receipt_path.exists() else {}
articles=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/articles.parquet').set_index('article_id')
output=[]
for row in records:
    if not row.get('at_or_before_first_impression'):continue
    closest=row['response']['archived_snapshots']['closest']; url=closest['url'].replace('http://web.archive.org/','https://web.archive.org/',1)
    r={'article_id':row['article_id'],'archive_timestamp':closest['timestamp'],'url':url}
    try:
        cached=DATA/f"archive_{row['article_id']}.html"
        if args.cached:
            raw=cached.read_bytes();r['loaded_cached_html']=True
            r['final_url']=prior[row['article_id']]['final_url'];r['http_status']=prior[row['article_id']]['http_status']
        else:
            with urllib.request.urlopen(url,timeout=20) as response:
                raw=response.read(20_000_000); r['http_status']=response.status; r['final_url']=response.url
            cached.write_bytes(raw)
        final_stamp=re.search(r'/web/(\d{14})',r['final_url'])
        if final_stamp is None or final_stamp.group(1)!=closest['timestamp']:
            raise ValueError('Archive redirect changed the requested capture timestamp')
        r['archive_capture_timestamp_verified']=True
        parser=ExtractMetadata(); parser.feed(raw.decode('utf-8',errors='replace'))
        r['meta']=parser.meta
        r['article_json_ld']=[v for b in parser.blocks for v in flat(b) if 'Article' in str(v.get('@type',''))]
        title=parser.meta.get('og:title') or next((v.get('headline') for v in r['article_json_ld'] if v.get('headline')),None)
        r['has_article_title']=title is not None
        if title is not None:r['title_matches_released']=normalized(title)==normalized(articles.loc[row['article_id'],'title'])
        r['released_published_time']=str(articles.loc[row['article_id'],'published_time'])
        r['released_category_str']=articles.loc[row['article_id'],'category_str']
        r['released_title']=articles.loc[row['article_id'],'title']
    except Exception as exc:r['error_type']=type(exc).__name__;r['error']=str(exc)
    output.append(r);print('archived page',len(output),r.get('error_type','ok'),flush=True)
(DATA/'archive_page_comparison.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
summary={'attempted':len(output),'retrieved':sum('http_status' in r for r in output),
         'article_title_available':sum(r.get('has_article_title',False) for r in output),
         'released_title_matches':sum(r.get('title_matches_released',False) for r in output),
         'released_title_differs':sum(r.get('title_matches_released') is False for r in output),
         'errors':dict(Counter(r['error_type'] for r in output if 'error_type' in r)),
         'completed_at_utc':datetime.now(timezone.utc).isoformat(),
         'scope':'Small descriptive public-page comparison; titles are not model inputs, proprietary topics/entities and categorical IDs remain unversioned.'}
stamps=[]
for row in output:
    value=row.get('meta',{}).get('og:article:published_time')
    if value is not None:
        stamps.append(datetime.fromisoformat(value.replace('Z','+00:00')).replace(tzinfo=None)==datetime.fromisoformat(row['released_published_time']))
summary.update({'published_time_present':len(stamps),'published_time_matches_source_clock':sum(stamps),
    'timestamp_comparison_note':'Three Z-suffixed public timestamps match naive dataset values after stripping UTC suffix; no blanket source-timezone inference.',
    'cached_html_parsed':sum(r.get('loaded_cached_html',False) for r in output),
    'network_pages_fetched_this_run':sum('http_status' in r and not r.get('loaded_cached_html') for r in output),
    'archive_capture_timestamps_verified':sum(r.get('archive_capture_timestamp_verified',False) for r in output)})
(OUT/'archive_page_comparison.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
