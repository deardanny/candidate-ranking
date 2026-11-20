"""Post-modification cohorts on unchanged full slates; no label-driven exclusions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.evaluation import paired_interval,METRICS

ROOT=Path(__file__).resolve().parents[1]; base=ROOT/'data/processed/ebnerd-v1'; out=ROOT/'reports/ebnerd-metadata-audit'
articles=pd.read_parquet(ROOT/'data/raw/ebnerd-small/extracted/articles.parquet',columns=['article_id','last_modified_time','published_time'])
frame=pd.read_parquet(base/'validation_requests.parquet'); ids=np.load(base/'test_items.npy'); groups=np.load(base/'test_groups.npy'); y=np.load(base/'test_y.npy'); offsets=np.r_[0,np.cumsum(groups)]
lookup=articles.set_index('article_id').last_modified_time
modified=lookup.reindex(ids).to_numpy(); late=modified>=np.datetime64('2023-06-30'); bulk=(modified>=np.datetime64('2023-06-29')) & ~late
anylate=np.add.reduceat(late,offsets[:-1])>0; clickedlate=np.add.reduceat(late*y,offsets[:-1])>0
allbulk=np.add.reduceat(bulk,offsets[:-1])==groups
models={name:pd.read_parquet(base/f'test-{name}_outcomes.parquet') for name in ['personalized','metadata','freshness','behavior']}
result={}
for name,mask in [('all_candidates_modified_June29',allbulk),('any_candidate_modified_later',anylate),('clicked_item_modified_later',clickedlate)]:
    result[name]={'requests':int(mask.sum()),'users':int(frame.loc[mask,'user_id'].nunique()),'models':{},'personalized_minus_metadata':{}}
    for model,values in models.items(): result[name]['models'][model]=values.loc[mask,METRICS].mean().to_dict()
    for metric in ['ndcg5','hit1','cold_hit1']:
        result[name]['personalized_minus_metadata'][metric]=paired_interval(frame.loc[mask,'user_id'],models['personalized'].loc[mask,metric]-models['metadata'].loc[mask,metric])
result['metadata_availability']={'candidate_rows':len(ids),'rows_with_last_modified_not_after_impression':int((modified<=np.repeat(frame.impression_time.to_numpy(),groups)).sum()),
    'late_modified_candidate_rows':int(late.sum()),'late_modified_clicked_rows':int((late*y).sum()),
    'caveat':'June29 is also after all impressions. Bulk timestamp cohorts are not a certified leakage-free control.'}
(out/'modification_cohorts.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
