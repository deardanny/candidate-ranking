"""Frozen popularity and contextual models fitted only on synthetic training users."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..io import sha256, write_json
from .world import FEATURES, choose, features, people, rng_for, visit


def training_rows(config, offers, cohort, count):
    by_id = {a['item_id']: a for a in offers}
    rows, labels, ids = [], [], []
    for p in people(config['seed'], count, cohort):
        seen = {}
        for index in range(config['training_visits']):
            ctx, candidates = visit(config['seed'], p, index, offers, training=True)
            rng = rng_for(config['seed'], 'random-training-policy', p['user_id'], index)
            shown = [by_id[str(i)] for i in rng.choice(candidates, 3, replace=False)]
            chosen = choose(config['seed'], p, index, ctx, shown, seen)
            for ad in shown:
                rows.append(features(ctx, ad)); labels.append(int(ad['item_id'] == chosen)); ids.append(ad['item_id'])
                seen[ad['item_id']] = seen.get(ad['item_id'], 0) + 1
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), ids


def train(config, offers, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    x, y, ids = training_rows(config, offers, 'train', config['training_users'])
    vx, vy, _ = training_rows(config, offers, 'validation', config['validation_users'])
    prior = float(y.mean())
    popularity = {}
    ids = np.array(ids)
    for ad in offers:
        mask = ids == ad['item_id']
        popularity[ad['item_id']] = float((y[mask].sum() + 20 * prior) / (mask.sum() + 20))
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
              'num_leaves': 15, 'learning_rate': .05, 'min_data_in_leaf': 100,
              'num_threads': 2, 'seed': config['seed'], 'deterministic': True,
              'force_col_wise': True}
    model = lgb.train(params, lgb.Dataset(x, label=y, feature_name=FEATURES),
                      num_boost_round=160, valid_sets=[lgb.Dataset(vx, label=vy, feature_name=FEATURES)],
                      callbacks=[lgb.early_stopping(15, verbose=False)])
    model.save_model(str(out/'model.txt'))
    write_json(out/'popularity.json', popularity)
    write_json(out/'metadata.json', {
        'synthetic': True, 'training_users': config['training_users'],
        'validation_users': config['validation_users'], 'training_examples': len(y),
        'validation_examples': len(vy), 'training_display_click_fraction': prior,
        'feature_names': FEATURES, 'rounds': model.best_iteration,
        'validation_logloss': model.best_score['valid_0']['binary_logloss'],
        'parameters': params,
        'provenance': 'Generated synthetic clicks under a random display policy; no evaluation users or outcomes.',
        'score_type': 'uncalibrated synthetic display-click prediction, used only for ordering',
        'sha256': {name: sha256(out/name) for name in ('model.txt', 'popularity.json')}})


class Policies:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory/'metadata.json').read_text())
        for name, expected in self.metadata['sha256'].items():
            if sha256(directory/name) != expected:
                raise ValueError('Model artifact changed')
        if self.metadata['feature_names'] != FEATURES:
            raise ValueError('Feature schema mismatch')
        self.model = lgb.Booster(model_file=str(directory/'model.txt'))
        if self.model.feature_name() != FEATURES:
            raise ValueError('Booster feature mismatch')
        self.popularity = json.loads((directory/'popularity.json').read_text())
        self.versions = {'popularity': 'synthetic-popularity-' + sha256(directory/'popularity.json')[:12],
                         'lightgbm': 'synthetic-lightgbm-' + sha256(directory/'model.txt')[:12]}

    def score(self, policy, matrix, ids):
        if policy == 'popularity':
            return np.array([self.popularity[i] for i in ids])
        if policy != 'lightgbm':
            raise ValueError('Unknown policy')
        scores = np.asarray(self.model.predict(np.asarray(matrix), num_threads=1))
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError('Invalid prediction')
        return scores
