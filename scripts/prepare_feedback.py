"""Prepare small development features without article tables or held-out labels."""
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from fairjob_ranker.ebnerd.feedback import FeedbackState, FEEDBACK_FEATURES
from fairjob_ranker.ebnerd.metadata_audit import STRICT_FEATURES
from fairjob_ranker.ebnerd.exposure_audit import EXPOSURE_FEATURES
from fairjob_ranker.io import sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/processed/ebnerd-metadata-audit'
DATA = ROOT/'data/processed/ebnerd-top1-feedback'
OUT = ROOT/'reports/ebnerd-top1-feedback'
if (OUT/'preparation.json').exists(): raise SystemExit('Already prepared')
frame = pd.read_parquet(BASE/'strict_train_requests.parquet')
state = FeedbackState(delay=3600)
base = np.load(BASE/'exposure_d3600_train_X.npy', mmap_mode='r')
x = np.lib.format.open_memmap(DATA/'train_X.npy', mode='w+', dtype='float32', shape=(len(base), base.shape[1]+len(FEEDBACK_FEATURES)))
x[:, :base.shape[1]] = base
offset = 0
times = frame.impression_time.to_numpy(dtype='datetime64[s]').astype('int64')
for j, (row, t) in enumerate(zip(frame.itertuples(index=False), times)):
    ids = row.article_ids_inview; end = offset+len(ids)
    x[offset:end, base.shape[1]:] = state.matrix(t, ids)
    state.observe(t, ids, row.article_ids_clicked)
    offset = end
    if (j+1)%50000 == 0: print('feedback features', j+1, flush=True)
assert offset == len(x) and np.isfinite(x).all()
x.flush()
features = STRICT_FEATURES+EXPOSURE_FEATURES+FEEDBACK_FEATURES
(OUT/'preparation.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(), 'features':features,
    'requests':len(frame), 'candidates':len(x), 'protocol_sha256':sha256(OUT/'protocol.json'),
    'matrix_sha256':sha256(DATA/'train_X.npy')},indent=2)+'\n')
print('Prepared response features', flush=True)
