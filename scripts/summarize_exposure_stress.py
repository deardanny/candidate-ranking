"""Preserve slow requests and queue propagation from the completed stress run."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ebnerd-metadata-audit';DATA=ROOT/'data/processed/ebnerd-metadata-audit'
x=np.load(ROOT/'runs/ebnerd-metadata-audit/exposure_stress_latencies.npy');indices=np.where(x[:,0]>50)[0]
groups=np.load(DATA/'strict_test_groups.npy')[-len(x):]
result={'requests':len(x),'misses':len(indices),'miss_rate':len(indices)/len(x),
    'misses_with_processing_over_50ms':int((x[indices,1]>50).sum()),'misses_with_queue_over_50ms':int((x[indices,2]>50).sum()),
    'outliers':[{'scheduled_index':int(i),'scheduled_elapsed_seconds':int(i)/50,'candidate_count':int(groups[i]),
                 'scheduled_to_complete_ms':float(x[i,0]),'processing_ms':float(x[i,1]),'scheduler_queue_ms':float(x[i,2])} for i in indices],
    'scope':'In-memory replay only. Processing stalls and queue delays are measured; no GC/OS tracing was captured, so root cause is not identified. No rerun to replace these observed misses.'}
(OUT/'exposure_stress_outliers.json').write_text(json.dumps(result,indent=2)+'\n');print('Preserved',len(indices),'deadline misses')
