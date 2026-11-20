"""Compare the official full article bundle to small without publishing article text."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ebnerd-metadata-audit'
small_path=ROOT/'data/raw/ebnerd-small/extracted/articles.parquet'
large_path=ROOT/'data/raw/ebnerd-article-reference/extracted/articles.parquet'
archive=ROOT/'data/raw/ebnerd-article-reference/articles_large_only.zip'
small=pd.read_parquet(small_path).set_index('article_id');large=pd.read_parquet(large_path).set_index('article_id')
assert small.index.is_unique and large.index.is_unique
shared=small.index.intersection(large.index)

def normalize(value):
    if isinstance(value,(list,tuple,np.ndarray)):return [normalize(v) for v in value]
    if pd.isna(value):return None
    if isinstance(value,(pd.Timestamp,np.datetime64)):return str(pd.Timestamp(value))
    if isinstance(value,np.generic):return value.item()
    return value

mismatched={}
for col in small:
    # Canonical strings prevent pandas from coercing normalized None back to NaN,
    # whose NaN != NaN comparison would falsely report identical nulls as edits.
    canonical=lambda value:json.dumps(normalize(value),ensure_ascii=False,sort_keys=True)
    left=small.loc[shared,col].map(canonical);right=large.loc[shared,col].map(canonical)
    mismatched[col]=sum(a!=b for a,b in zip(left,right))
result={'source_url':'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip',
    'download_sha256':sha256(archive),'source_parquet_sha256':sha256(large_path),
    'small_articles':len(small),'large_articles':len(large),'shared_ids':len(shared),
    'column_mismatched_articles':mismatched,'compared_at_utc':datetime.now(timezone.utc).isoformat(),
    'interpretation':'A second official bundle is not automatically a historical snapshot.'}
path=OUT/'reference_comparison.json'
if path.exists():
    expected=json.loads(path.read_text())
    for key in ['download_sha256','source_parquet_sha256','small_articles','large_articles','shared_ids','column_mismatched_articles']:
        assert expected[key]==result[key],key
    (OUT/'reference_verification.json').write_text(json.dumps({'all_counts_hashes_and_field_differences_reproduced':True,'verified_at_utc':result['compared_at_utc']},indent=2)+'\n')
else:path.write_text(json.dumps(result,indent=2)+'\n')
print('Shared articles:',len(shared),'field mismatches:',sum(mismatched.values()))
