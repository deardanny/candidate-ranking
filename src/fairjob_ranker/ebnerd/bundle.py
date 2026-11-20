"""News model bundles are independent of time-versioned catalogs/history."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from ..io import sha256
from .features import FEATURES,VARIANTS


class Bundle:
    def __init__(self,directory):
        self.directory=Path(directory); self.meta=json.loads((self.directory/'metadata.json').read_text())
        if self.meta['schema_version']!='ebnerd-v1' or self.meta['score_type']!='ranking_score': raise ValueError('Bundle schema mismatch')
        if set(self.meta['sha256'])!={'model.txt','features.json','seen_items.json'}: raise ValueError('Unexpected artifacts')
        for name,digest in self.meta['sha256'].items():
            if sha256(self.directory/name)!=digest: raise ValueError('Bundle checksum mismatch')
        self.features=json.loads((self.directory/'features.json').read_text())
        if self.features!=VARIANTS[self.meta['variant']]: raise ValueError('Feature contract mismatch')
        self.indices=[FEATURES.index(c) for c in self.features]
        self.model=lgb.Booster(model_file=str(self.directory/'model.txt'))
        if self.model.feature_name()!=self.features: raise ValueError('Model feature mismatch')
        self.seen=set(json.loads((self.directory/'seen_items.json').read_text()))
        self.digest=sha256(self.directory/'metadata.json')

    def predict(self,matrix):
        score=self.model.predict(np.asarray(matrix)[:,self.indices],num_threads=1)
        if score.shape!=(len(matrix),) or not np.isfinite(score).all(): raise ValueError('Invalid scores')
        return score
