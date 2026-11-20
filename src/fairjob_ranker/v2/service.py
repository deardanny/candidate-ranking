"""V2 scoring adapter sharing V1 request, feature, catalog and fallback contracts."""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ..catalog import AdCatalog, redis_client
from ..schema import CATEGORICAL, FEATURES, VERSION
from .models import load_bundle


class Ranker:
    def __init__(self, directory, redis_url=None):
        torch.set_num_threads(1)
        self.bundle = load_bundle(Path(directory))
        self.client = redis_client(redis_url, .005) if redis_url else None
        self.catalog = AdCatalog(Path(directory)/'ad_catalog.parquet',self.bundle.metadata['catalog_version'],self.client)
        self.catalog.probe()

    def close(self):
        if self.client is not None:
            self.client.close()

    def rank(self, request):
        start = perf_counter()
        records, skipped, reasons, hits = self.catalog.lookup(request.candidate_ids)
        if not records:
            raise ValueError('No known candidates remain')
        lookup = perf_counter()
        context = {'cat0':request.context.cat0,'cat1':request.context.cat1,**request.context.numeric}
        raw = pd.DataFrame([{**r,**context} for r in records])
        features = self.bundle.transformer.transform(raw[FEATURES])
        oov = int(features[CATEGORICAL].eq(0).to_numpy().sum())
        transformed = perf_counter()
        method, version = self.bundle.metadata['kind'],self.bundle.metadata['model_version']
        try:
            scores = np.asarray(self.bundle.predict_transformed(features))
            if scores.shape != (len(records),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
                raise ValueError('Invalid scores')
        except Exception:
            method = 'smoothed_ad_ctr'
            version = method+':'+self.bundle.metadata['artifact_sha256']['fallback.json'][:12]
            reasons.append('model_inference_error')
            scores = np.array([self.bundle.fallback['scores'].get(str(r['product_id']),self.bundle.fallback['train_ctr']) for r in records])
        inferred = perf_counter()
        rankings = [{'product_id':r['product_id'],'score':float(s),'position':i+1}
                    for i,(r,s) in enumerate(sorted(zip(records,scores),key=lambda x:(-float(x[1]),x[0]['product_id']))[:request.top_k])]
        end = perf_counter()
        return {'request_id':request.request_id,'schema_version':VERSION,'model_version':version,
                'attempted_model_version':self.bundle.metadata['model_version'],
                'feature_version':self.bundle.metadata['feature_version'],'catalog_version':self.bundle.metadata['catalog_version'],
                'scoring_method':method,'degraded':bool(reasons),'fallback_reason':','.join(sorted(set(reasons))) or None,
                'skipped_candidate_ids':skipped,'cache_hits':hits,'oov_categories':oov,'rankings':rankings,
                'timings_ms':{'lookup':(lookup-start)*1000,'transform':(transformed-lookup)*1000,
                              'inference':(inferred-transformed)*1000,'sort':(end-inferred)*1000,'service':(end-start)*1000}}
