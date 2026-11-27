"""Fictional campus coupon world: curiosity clicks can differ from usable offers."""
import math
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np

from ..io import sha256, write_json
from .world import rng_for

CATEGORIES = ('coffee', 'lunch', 'study')
FEATURES = ['interest_match', 'discount', 'price', 'price_budget_ratio', 'walk_minutes',
            'service_minutes', 'round_trip_minutes', 'time_ratio', 'minutes_until_class',
            'budget', 'coffee', 'lunch', 'study']
NAMES = [
    'North Gate Coffee', 'Library Coffee', 'Maple Espresso', 'Station Beans',
    'Garden Coffee', 'West End Cafe', 'Campus Roast', 'Bridge Coffee',
    'Market Bento', 'Library Deli', 'Maple Noodles', 'Station Bowl',
    'Garden Kitchen', 'West End Wraps', 'Campus Rice', 'Bridge Sandwich',
    'North Gate Study', 'Library Lounge', 'Maple Desk', 'Station Study',
    'Garden Desk', 'West End Workroom', 'Campus Corner', 'Bridge Workspace']


def offers(seed):
    rng = rng_for(seed, 'campus-catalog')
    result = []
    for i, name in enumerate(NAMES):
        category = i // 8
        original_price = float(round(rng.uniform((4, 9, 6)[category], (8, 16, 12)[category]), 2))
        discount = float(rng.choice([.1, .2, .3, .5]))
        walk = int(rng.integers(2, 17))
        service = int(rng.integers(*[(2, 6), (6, 15), (20, 41)][category]))
        if i == 0:
            original_price, discount, walk, service = 6., .5, 15, 3
        elif i == 1:
            original_price, discount, walk, service = 5., .1, 3, 3
        result.append({'item_id': f'campus-{i:02d}', 'merchant': name,
            'category': category, 'product': ('Americano', 'Lunch set', 'Study seat + drink')[category],
            'original_price': original_price, 'discount': discount,
            'price': round(original_price * (1-discount), 2), 'walk_minutes': walk,
            'service_minutes': service, 'round_trip_minutes': 2*walk+service,
            'sponsored': True, 'fictional': True, 'campaign_eligible': True})
    return result


def people(seed, n, cohort):
    rng = rng_for(seed, 'campus-personas', cohort)
    return [{'user_id': f'{cohort}-{i:05d}',
             'public': {'interest': int(rng.integers(0, 3)), 'budget': float(rng.uniform(4, 14))},
             'private': {'affinity': rng.normal(0, .35, 3).tolist(),
                         'no_need': float(rng.uniform(.05, .25)),
                         'discount_curiosity': float(rng.uniform(.65, 1.35))}}
            for i in range(n)]


def visit(seed, p, index, ads, training=False):
    rng = rng_for(seed, 'campus-visit', p['user_id'], index)
    ctx = dict(p['public'])
    if rng.random() < .25:
        ctx['interest'] = int(rng.integers(0, 3))
    ctx['budget'] *= float(rng.uniform(.85, 1.15))
    ctx['minutes_until_class'] = float(rng.integers(15, 91 if training else 46))
    # Six goal-matching and two other offers: retrieval is identical for both policies.
    matching = [a['item_id'] for a in ads if a['category'] == ctx['interest']]
    other = [a['item_id'] for a in ads if a['category'] != ctx['interest']]
    ids = list(rng.choice(matching, 6, replace=False)) + list(rng.choice(other, 2, replace=False))
    rng.shuffle(ids)
    return ctx, [str(i) for i in ids]


def features(ctx, ad):
    return [float(ctx['interest'] == ad['category']), ad['discount'], ad['price'],
            ad['price']/ctx['budget'], float(ad['walk_minutes']), float(ad['service_minutes']),
            float(ad['round_trip_minutes']), ad['round_trip_minutes']/ctx['minutes_until_class'],
            ctx['minutes_until_class'], ctx['budget']] + [float(ad['category'] == c) for c in range(3)]


def respond(seed, person, index, ctx, ad, travel_constraint=True):
    """One ad; at most one detail click and one subsequent coupon redemption.

    Policy names, predictions and group assignments are not inputs. All behavioral
    coefficients are assumptions authored for this stress test, not fitted humans.
    """
    rng = rng_for(seed, 'campus-response', person['user_id'], index)
    no_need_draw, click_draw, redeem_draw = rng.random(3)
    if no_need_draw < person['private']['no_need']:
        return {'clicked': False, 'redeemed': False, 'virtual_redemption_delay_minutes': None}
    f = features(ctx, ad)
    affinity = person['private']['affinity'][ad['category']]
    click_logit = (-2.3 + 1.05*f[0] + 3.8*ad['discount']*person['private']['discount_curiosity']
                   + affinity - .25*max(0, f[3]-1))
    clicked = bool(click_draw < 1/(1+math.exp(-click_logit)))
    # A discount can induce curiosity even when the round trip cannot fit before class.
    feasible = not travel_constraint or ad['round_trip_minutes'] <= ctx['minutes_until_class']
    redemption_logit = .8 + .7*f[0] + affinity - 2.8*max(0, f[3]-1)
    redeemed = bool(clicked and feasible and redeem_draw < 1/(1+math.exp(-redemption_logit)))
    delay = float(ad['walk_minutes'] + ad['service_minutes']) if redeemed else None
    return {'clicked': clicked, 'redeemed': redeemed, 'virtual_redemption_delay_minutes': delay}


def training_rows(config, ads, cohort, count):
    by_id = {a['item_id']: a for a in ads}
    x, clicks, coupons = [], [], []
    for p in people(config['seed'], count, cohort):
        for i in range(config['training_visits']):
            ctx, candidates = visit(config['seed'], p, i, ads, training=True)
            ad = by_id[str(rng_for(config['seed'], 'campus-random-policy', p['user_id'], i).choice(candidates))]
            result = respond(config['seed'], p, i, ctx, ad)
            x.append(features(ctx, ad)); clicks.append(result['clicked']); coupons.append(result['redeemed'])
    return np.asarray(x, dtype=np.float32), {'click': np.array(clicks, int), 'coupon': np.array(coupons, int)}


def train(config, ads, out):
    out = Path(out); out.mkdir(exist_ok=False)
    x, y = training_rows(config, ads, 'train', config['training_users'])
    vx, vy = training_rows(config, ads, 'validation', config['validation_users'])
    metadata = {'synthetic': True, 'features': FEATURES, 'training_rows': len(x),
                'validation_rows': len(vx), 'training_users': config['training_users'],
                'validation_users': config['validation_users'], 'models': {}}
    for target in ('click', 'coupon'):
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
                  'num_leaves': 15, 'learning_rate': .05, 'min_data_in_leaf': 80,
                  'num_threads': 2, 'seed': config['seed'], 'deterministic': True, 'force_col_wise': True}
        m = lgb.train(params, lgb.Dataset(x, label=y[target], feature_name=FEATURES),
            num_boost_round=180, valid_sets=[lgb.Dataset(vx, label=vy[target], feature_name=FEATURES)],
            callbacks=[lgb.early_stopping(15, verbose=False)])
        m.save_model(str(out/f'{target}.txt'))
        metadata['models'][target] = {'sha256': sha256(out/f'{target}.txt'), 'rounds': m.best_iteration,
            'validation_logloss': m.best_score['valid_0']['binary_logloss'], 'params': params,
            'training_rate': float(y[target].mean()),
            'target': 'click per impression' if target == 'click' else 'click AND redemption per impression; all fully observed exposures, including non-clicks'}
    write_json(out/'metadata.json', metadata)


class Policies:
    def __init__(self, path):
        path = Path(path)
        metadata = json.loads((path/'metadata.json').read_text())
        if metadata['features'] != FEATURES:
            raise ValueError('Feature contract mismatch')
        self.models, self.versions = {}, {}
        for target in ('click', 'coupon'):
            h = sha256(path/f'{target}.txt')
            if h != metadata['models'][target]['sha256']:
                raise ValueError('Model checksum mismatch')
            self.models[target] = lgb.Booster(model_file=str(path/f'{target}.txt'))
            if self.models[target].feature_name() != FEATURES:
                raise ValueError('Model feature names mismatch')
            self.versions[target] = f'synthetic-campus-{target}-{h[:12]}'

    def score(self, target, ctx, ads):
        result = self.models[target].predict(np.array([features(ctx, ad) for ad in ads]), num_threads=1)
        if result.shape != (len(ads),) or not np.isfinite(result).all():
            raise ValueError('Invalid scores')
        return result
