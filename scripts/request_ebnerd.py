"""Send the first clean request containing a model-unseen article to local HTTP."""
import argparse
import json
from pathlib import Path
import httpx
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--url',default='http://127.0.0.1:8001'); args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
seen=set(json.loads((root/'models/ebnerd-v1/final-personalized/seen_items.json').read_text()))
frame=pd.read_parquet(root/'data/processed/ebnerd-v1/validation_requests.parquet',columns=['impression_id','user_id','impression_time','device_type','article_ids_inview'])
row=next(r for r in frame.itertuples(index=False) if any(int(i) not in seen for i in r.article_ids_inview))
body={'schema_version':'ebnerd-v1','request_id':'demo-'+str(row.impression_id),'user_id':int(row.user_id),
      'request_time':str(row.impression_time),'device_type':int(row.device_type),
      'candidate_ids':list(map(int,row.article_ids_inview)),'top_k':len(row.article_ids_inview)}
with httpx.Client(base_url=args.url,timeout=10,trust_env=False) as client:
    response=client.post('/rank',json=body); response.raise_for_status(); print(json.dumps(response.json(),indent=2))
