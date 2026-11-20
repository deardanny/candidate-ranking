"""Full-rank stacked DCN-V2, group-uniform pairs, and verified V2 bundles."""
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..features import FeatureTransformer
from ..io import read_json, sha256
from ..schema import CATEGORICAL, FEATURES, NUMERICAL, VERSION
from ..torch_model import EmbeddingMLP, predict_tensors, to_tensors


class CrossLayer(nn.Module):
    """x_(l+1) = x_0 * (W_l x_l + b_l) + x_l, full matrix W."""
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, original, current):
        return original * self.linear(current) + current


class DeepCross(EmbeddingMLP):
    def __init__(self, category_sizes, embedding_dim=8, hidden_dims=None, dropout=0.1, cross_layers=2):
        super().__init__(category_sizes, embedding_dim, hidden_dims, dropout)
        width = len(category_sizes) * embedding_dim + len(NUMERICAL)
        self.cross = nn.ModuleList([CrossLayer(width) for _ in range(cross_layers)])

    def forward(self, categorical, numerical):
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        original = torch.cat([*embedded, numerical], dim=1)
        crossed = original
        for layer in self.cross:
            crossed = layer(original, crossed)
        return self.network(crossed).squeeze(1)


class PairSampler:
    """Sample groups uniformly, then one observed positive and negative per group."""
    def __init__(self, frame):
        self.groups = []
        indexed = frame.reset_index(drop=True)
        for _, group in indexed.groupby('impression_id', sort=True):
            positive = group.index[group.click.eq(1)].to_numpy()
            negative = group.index[group.click.eq(0)].to_numpy()
            if len(positive) and len(negative):
                self.groups.append((positive, negative))
        if not self.groups:
            raise ValueError('Pairwise learning needs groups with both labels')

    def sample(self, count, rng):
        chosen = rng.integers(len(self.groups), size=count)
        positive, negative = [], []
        for i in chosen:
            p, n = self.groups[i]
            positive.append(rng.choice(p))
            negative.append(rng.choice(n))
        return torch.tensor(positive), torch.tensor(negative)


@dataclass
class Bundle:
    model: object
    transformer: FeatureTransformer
    metadata: dict
    fallback: dict
    directory: Path

    def predict_transformed(self, transformed):
        if self.metadata['kind'] == 'lightgbm':
            return self.model.predict(transformed, num_threads=1)
        return predict_tensors(self.model, *to_tensors(transformed))

    def predict(self, raw):
        return self.predict_transformed(self.transformer.transform(raw))


def load_bundle(directory):
    directory = Path(directory)
    meta = read_json(directory / 'metadata.json')
    if meta['bundle_version'] != 2 or meta['schema_version'] != VERSION or meta['feature_order'] != FEATURES:
        raise ValueError('V2 bundle schema mismatch')
    kind = meta['kind']
    if kind not in {'lightgbm', 'mlp', 'dcn'}:
        raise ValueError('Unknown model kind')
    weights = 'model.txt' if kind == 'lightgbm' else 'weights.pt'
    required = {weights, 'features.json', 'ad_catalog.parquet', 'fallback.json'}
    if set(meta['artifact_sha256']) != required:
        raise ValueError('Unexpected bundle artifact list')
    for name in required:
        if sha256(directory / name) != meta['artifact_sha256'][name]:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    if meta['feature_version'] != meta['artifact_sha256']['features.json'] or meta['catalog_version'] != meta['artifact_sha256']['ad_catalog.parquet']:
        raise ValueError('Feature/catalog version mismatch')
    transformer = FeatureTransformer.load(directory / 'features.json')
    if kind == 'lightgbm':
        model = lgb.Booster(model_file=str(directory / weights))
        if model.feature_name() != FEATURES:
            raise ValueError('LightGBM feature order mismatch')
    else:
        arch = meta['architecture']
        if arch['category_sizes'] != [len(transformer.vocabularies[c]) + 1 for c in CATEGORICAL]:
            raise ValueError('Vocabulary mismatch')
        model = (DeepCross if kind == 'dcn' else EmbeddingMLP)(**arch)
        model.load_state_dict(torch.load(directory / weights, weights_only=True, map_location='cpu'))
        model.eval()
    fallback = read_json(directory / 'fallback.json')
    values = np.array([fallback['train_ctr'], *fallback['scores'].values()])
    if fallback['fitted_on'] != 'inner_train' or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid train-only fallback')
    return Bundle(model, transformer, meta, fallback, directory)
