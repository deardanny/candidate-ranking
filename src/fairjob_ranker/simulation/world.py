"""Seeded invented consumers, offers and a policy-blind response generator.

No model scores, experiment arm or learned labels enter the response function.
The shared invented world still favors models able to learn its assumptions.
"""
import hashlib
import json
import math

import numpy as np

CATEGORIES = ('coffee', 'workspace', 'fitness', 'music', 'meal', 'books')
FEATURES = ['interest_match', 'price_budget_ratio', 'distance_limit_ratio',
            'duration_time_ratio', 'price', 'distance', 'duration', 'budget',
            'max_distance', 'available_minutes'] + [f'ad_{c}' for c in CATEGORIES]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rng_for(*parts):
    return np.random.default_rng(int(digest(list(parts))[:16], 16))


def catalog(seed):
    rng = rng_for(seed, 'catalog')
    offers = []
    for i in range(42):
        cat = i % 6
        offers.append({'item_id': f'ad-{i:03d}', 'category': cat,
                       'title': f'Fictional {CATEGORIES[cat]} offer {i + 1}',
                       'price': float(rng.uniform(2, 16)),
                       'distance': float(rng.uniform(1, 18)),
                       'duration': float(rng.uniform(2, 100)), 'cold': i >= 36})
    return offers


def people(seed, count, cohort):
    rng = rng_for(seed, 'people', cohort)
    result = []
    for i in range(count):
        interest = int(rng.integers(0, 6))
        result.append({
            'user_id': f'{cohort}-{i:05d}',
            'observed': {'interest': interest, 'budget': float(rng.uniform(3, 18)),
                         'max_distance': float(rng.uniform(4, 20)),
                         'available_minutes': float(rng.uniform(15, 120))},
            'private': {'affinity': rng.normal(0, .45, 6).tolist(),
                        'price_sensitivity': float(rng.uniform(.7, 1.5)),
                        'no_need_probability': float(rng.uniform(.05, .35)),
                        'patience': float(rng.uniform(.7, 1.0))}})
    return result


def visit(seed, person, index, offers, training=False):
    rng = rng_for(seed, 'visit', person['user_id'], index)
    context = dict(person['observed'])
    if rng.random() < .25:
        context['interest'] = int(rng.integers(0, 6))
    context['budget'] *= float(rng.uniform(.8, 1.2))
    context['available_minutes'] *= float(rng.uniform(.6, 1.1))
    old = [a['item_id'] for a in offers if not a['cold']]
    new = [a['item_id'] for a in offers if a['cold']]
    if not training and index >= 4:
        ids = list(rng.choice(old, 6, replace=False)) + list(rng.choice(new, 2, replace=False))
        rng.shuffle(ids)
    else:
        ids = list(rng.choice(old, 8, replace=False))
    return context, [str(x) for x in ids]


def features(context, offer):
    return [float(context['interest'] == offer['category']),
            offer['price'] / context['budget'], offer['distance'] / context['max_distance'],
            offer['duration'] / context['available_minutes'],
            offer['price'], offer['distance'], offer['duration'], context['budget'],
            context['max_distance'], context['available_minutes']] + [float(offer['category'] == i) for i in range(6)]


def choose(seed, person, visit_index, context, displayed, seen, mode='preferences'):
    """At most one synthetic click after top-to-bottom examination of <=3 ads.

    Random numbers are keyed by user and visit, not arm or experiment. The null
    mode ignores identities/features entirely, making it a useful negative control.
    """
    if mode not in ('preferences', 'item_independent'):
        raise ValueError('Unknown response model')
    rng = rng_for(seed, 'response', person['user_id'], visit_index)
    no_need = rng.random() < person['private']['no_need_probability']
    examination = rng.random(3)
    click_draw = rng.random(3)
    if no_need:
        return None
    for position, ad in enumerate(displayed):
        if position >= 3:
            raise ValueError('Simulator supports three displayed positions')
        if examination[position] > (1.0, .65, .35)[position] * person['private']['patience']:
            continue
        if mode == 'item_independent':
            probability = .12
        else:
            f = features(context, ad)
            appeal = float(rng_for(seed, 'appeal', ad['item_id']).normal(0, .35))
            utility = (-2.1 + 2.4 * f[0] + person['private']['affinity'][ad['category']] + appeal
                       - 2.0 * person['private']['price_sensitivity'] * max(0, f[1] - 1)
                       - 1.2 * max(0, f[2] - 1) - .8 * max(0, f[3] - 1)
                       - .22 * seen.get(ad['item_id'], 0))
            probability = 1 / (1 + math.exp(-utility))
        if click_draw[position] < probability:
            return ad['item_id']
    return None
