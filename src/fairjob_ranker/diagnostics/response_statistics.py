"""User-cross-fitted, hierarchical response statistics with label-free transforms."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from ..io import read_json, sha256, write_json
from ..schema import AD_FEATURES
from .candidate_count_reproduction import user_bucket


CONTEXT_KEYS = ['cat0', 'cat1']
KEY_COLUMNS = ['product_id'] + CONTEXT_KEYS + AD_FEATURES
TABLE_KEYS = [(k,) for k in ['product_id'] + AD_FEATURES + CONTEXT_KEYS] + [
    (k, context) for context in CONTEXT_KEYS for k in ['product_id'] + AD_FEATURES]


def name(key):
    return '__'.join(key)


def statistic_columns(variant):
    if variant not in ['ad_stats', 'ad_context_stats']:
        raise ValueError('Unknown feature variant')
    keys = [(k,) for k in ['product_id'] + AD_FEATURES] if variant == 'ad_stats' else TABLE_KEYS
    return [f'rs__{name(key)}__{suffix}' for key in keys for suffix in ['p', 'log_n']]


@dataclass
class ResponseStatistics:
    tables: dict
    rows: int
    clicks: int
    product_smoothing: float = 20.
    category_smoothing: float = 100.

    @property
    def prior(self):
        return (self.clicks + .5) / (self.rows + 1.)

    @classmethod
    def fit(cls, source, product_smoothing=20., category_smoothing=100.):
        if source.empty or not source.click.isin([0, 1]).all():
            raise ValueError('Nonempty source with binary labels required')
        if product_smoothing <= 0 or category_smoothing <= 0:
            raise ValueError('Positive smoothing required')
        if source[KEY_COLUMNS].isna().any().any():
            raise ValueError('Missing lookup keys')
        tables = {key: source.groupby(list(key), sort=True).click.agg(n='size', c='sum') for key in TABLE_KEYS}
        return cls(tables, len(source), int(source.click.sum()), product_smoothing, category_smoothing)

    def counts(self, keys, key):
        index = pd.Index(keys[key[0]], name=key[0]) if len(key) == 1 else pd.MultiIndex.from_frame(keys[list(key)])
        values = self.tables[key].reindex(index).fillna(0.)[['n', 'c']].to_numpy(dtype=float)
        return values[:, 0], values[:, 1]

    def transform(self, keys):
        # Reject label-bearing input rather than relying on callers to trust column selection.
        if set(keys.columns) != set(KEY_COLUMNS) or keys.empty:
            raise ValueError('Transform requires exactly the label-free lookup keys')
        if keys.isna().any().any():
            raise ValueError('Missing lookup keys')
        rates, counts = {}, {}
        for field in AD_FEATURES + CONTEXT_KEYS:
            n, c = self.counts(keys, (field,))
            rates[(field,)] = (c+self.category_smoothing*self.prior)/(n+self.category_smoothing)
            counts[(field,)] = n
        n, c = self.counts(keys, ('product_id',))
        parent = np.mean([rates[(a,)] for a in AD_FEATURES], axis=0)
        rates[('product_id',)] = (c+self.product_smoothing*parent)/(n+self.product_smoothing)
        counts[('product_id',)] = n
        for context in CONTEXT_KEYS:
            for ad in AD_FEATURES:
                key = (ad, context)
                n, c = self.counts(keys, key)
                parent = expit(logit(rates[(ad,)])+logit(rates[(context,)])-logit(self.prior))
                rates[key] = (c+self.category_smoothing*parent)/(n+self.category_smoothing)
                counts[key] = n
            key = ('product_id', context)
            n, c = self.counts(keys, key)
            parent = np.mean([rates[(a, context)] for a in AD_FEATURES], axis=0)
            rates[key] = (c+self.product_smoothing*parent)/(n+self.product_smoothing)
            counts[key] = n
        result = pd.DataFrame({f'rs__{name(key)}__{suffix}': rates[key] if suffix == 'p' else np.log1p(counts[key])
                              for key in TABLE_KEYS for suffix in ['p', 'log_n']}, index=keys.index).astype('float32')
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Nonfinite response features')
        return result

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for key, table in self.tables.items():
            path = folder/(name(key)+'.parquet')
            table.to_parquet(path)
            hashes[path.name] = sha256(path)
        write_json(folder/'metadata.json', {'rows': self.rows, 'clicks': self.clicks,
            'product_smoothing': self.product_smoothing, 'category_smoothing': self.category_smoothing,
            'feature_order': statistic_columns('ad_context_stats'), 'sha256': hashes})

    @classmethod
    def load(cls, folder):
        metadata = read_json(folder/'metadata.json')
        if metadata['feature_order'] != statistic_columns('ad_context_stats'):
            raise ValueError('Statistic schema mismatch')
        for file, digest in metadata['sha256'].items():
            if sha256(folder/file) != digest:
                raise ValueError('Statistic table changed')
        return cls({key: pd.read_parquet(folder/(name(key)+'.parquet')) for key in TABLE_KEYS},
                   metadata['rows'], metadata['clicks'], metadata['product_smoothing'], metadata['category_smoothing'])


def encoding_assignments(frame, config, outer_fold):
    users = pd.DataFrame({'user_id': sorted(frame.user_id.unique())})
    users['encoding_fold'] = users.user_id.map(lambda u: user_bucket(u,
        config['encoding_salt']+f':outer-{outer_fold}', config['encoding_folds']))
    return users


def cross_fit_statistics(fit, config, outer_fold):
    if not fit.index.equals(pd.RangeIndex(len(fit))):
        raise ValueError('Cross-fitting requires a reset positional index')
    assignment = encoding_assignments(fit, config, outer_fold)
    buckets = fit.user_id.map(assignment.set_index('user_id').encoding_fold)
    result = np.full((len(fit), len(statistic_columns('ad_context_stats'))), np.nan, dtype=np.float32)
    audits = []
    for encoding_fold in range(config['encoding_folds']):
        target_mask = buckets.eq(encoding_fold)
        source, target = fit.loc[~target_mask], fit.loc[target_mask]
        if target.empty or source.empty:
            raise ValueError('Empty encoding partition')
        if set(source.user_id) & set(target.user_id) or set(source.impression_id) & set(target.impression_id):
            raise ValueError('Encoding user/request leakage')
        encoder = ResponseStatistics.fit(source, config['product_smoothing'], config['category_smoothing'])
        result[target_mask] = encoder.transform(target[KEY_COLUMNS]).to_numpy()
        audits.append({'encoding_fold': encoding_fold, 'source_rows': len(source), 'target_rows': len(target),
            'source_users': int(source.user_id.nunique()), 'target_users': int(target.user_id.nunique()),
            'source_clicks': encoder.clicks, 'global_prior': encoder.prior, 'overlap_users': 0, 'overlap_requests': 0})
    if not np.isfinite(result).all():
        raise ValueError('Incomplete out-of-user statistics')
    return pd.DataFrame(result, columns=statistic_columns('ad_context_stats')), assignment, audits
