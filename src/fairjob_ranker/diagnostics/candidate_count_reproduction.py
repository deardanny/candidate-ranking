"""Candidate-count diagnostics and nested, user-disjoint refits on development data."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import AD_FEATURES, CATEGORICAL, FEATURES
from .business_validation import bootstrap_request_means, interval, topk_weights


META = ['user_id', 'impression_id', 'product_id', 'click', 'displayrandom', 'rank']


def user_bucket(user, salt, modulus):
    return int.from_bytes(hashlib.sha256(f'{salt}:{int(user)}'.encode()).digest()[:8], 'big') % modulus


def user_assignments(frame, config):
    users = pd.DataFrame({'user_id': sorted(frame.user_id.unique())})
    users['outer_fold'] = users.user_id.map(lambda u: user_bucket(u, config['outer_salt'], config['outer_folds']))
    for fold in range(config['outer_folds']):
        users[f'inner_fit_{fold}'] = users.user_id.map(
            lambda u: user_bucket(u, config['inner_salt']+f':fold-{fold}', 100) < config['inner_train_percent'])
    return users


def vector_outcomes(frame, scores):
    """Vectorized, tie-expected top-k outcomes on every multi-candidate request."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError('Aligned finite scores required')
    if not frame.click.isin([0, 1]).all() or frame.duplicated(['impression_id', 'product_id']).any():
        raise ValueError('Binary labels and unique candidates required')
    work = frame[META].reset_index(drop=True).assign(score=scores)
    grouping = work.groupby('impression_id', sort=True)
    if grouping[['user_id', 'displayrandom']].nunique().gt(1).any().any():
        raise ValueError('Request crosses user or display flag')
    stats = grouping.agg(user_id=('user_id', 'first'), displayrandom=('displayrandom', 'first'),
                         candidates=('click', 'size'), clicks=('click', 'sum'),
                         min_rank=('rank', 'min'), max_rank=('rank', 'max'), rank_count=('rank', 'nunique'))
    stats['any_click'] = stats.clicks.gt(0).astype(float)
    stats['rank_contiguous'] = stats.min_rank.eq(1) & stats.max_rank.eq(stats.candidates) & stats.rank_count.eq(stats.candidates)
    low = grouping.score.rank(method='min', ascending=False)
    high = grouping.score.rank(method='max', ascending=False)
    for k in [1, 3]:
        weights = ((k-low+1)/(high-low+1)).clip(0, 1)
        stats[f'capture{k}'] = (weights*work.click).groupby(work.impression_id).sum()
        stats[f'uniform{k}'] = stats.clicks*np.minimum(k, stats.candidates)/stats.candidates
    return stats.loc[stats.candidates.ge(2)].reset_index(), work.assign(top1_weight=((1-low+1)/(high-low+1)).clip(0, 1))


def seed_mean_outcomes(frame, predictions, seeds):
    groups, credits = None, np.zeros(len(frame))
    for variant in ['full', 'ad_only']:
        columns = []
        for seed in seeds:
            outcome, rows = vector_outcomes(frame, predictions[f'{variant}-s{seed}'])
            if groups is None:
                groups = outcome.drop(columns=['capture1', 'capture3'])
            np.testing.assert_array_equal(groups.impression_id, outcome.impression_id)
            for k in [1, 3]:
                groups[f'{variant}_s{seed}_{k}'] = outcome[f'capture{k}'].to_numpy()
            columns.append(f'{variant}_s{seed}_1')
            if variant == 'full':
                credits += rows.top1_weight.to_numpy()*rows.click.to_numpy()/len(seeds)
        for k in [1, 3]:
            groups[f'{variant}{k}'] = groups[[f'{variant}_s{s}_{k}' for s in seeds]].mean(axis=1)
    return groups, credits


def cohort_summary(groups, config):
    if groups.empty:
        return {'requests': 0}
    columns = ['full1', 'ad_only1', 'uniform1', 'full3', 'ad_only3', 'uniform3', 'any_click', 'clicks']
    point, samples = bootstrap_request_means(groups, columns, config['bootstrap_repeats'], config['bootstrap_seed'])
    idx = {name: i for i, name in enumerate(columns)}
    result = {'requests': len(groups), 'users': int(groups.user_id.nunique()), 'observed_clicks': int(groups.clicks.sum()),
              'requests_with_clicks': int(groups.any_click.sum()), 'mean_candidates': float(groups.candidates.mean()),
              'randomized_fraction': float(groups.displayrandom.mean()), 'rank_contiguous_fraction': float(groups.rank_contiguous.mean()),
              'click_request_prevalence': float(groups.any_click.mean()), 'comparisons': {},
              'top1_per_1000': {name: float(point[idx[name]]*1000) for name in ['full1', 'ad_only1', 'uniform1']},
              'per_seed_full_minus_uniform_top1': {str(s): float((groups[f'full_s{s}_1']-groups.uniform1).mean()*1000) for s in config['seeds']}}
    for k in [1, 3]:
        for a, b in [('full', 'uniform'), ('full', 'ad_only'), ('ad_only', 'uniform')]:
            ai, bi = idx[f'{a}{k}'], idx[f'{b}{k}']
            result['comparisons'][f'{a}-{b}-top{k}'] = interval(point[ai]-point[bi], samples[:, ai]-samples[:, bi], 1000)
    positive = groups.loc[groups.any_click.eq(1)]
    result['conditional_on_any_click'] = {
        'full_top1': float(positive.full1.mean()) if len(positive) else None,
        'uniform_top1': float(positive.uniform1.mean()) if len(positive) else None,
        'gap_top1': float((positive.full1-positive.uniform1).mean()) if len(positive) else None}
    with np.errstate(divide='ignore', invalid='ignore'):
        headroom = (point[idx['full1']]-point[idx['uniform1']])/(point[idx['any_click']]-point[idx['uniform1']])
        boot_headroom = (samples[:, idx['full1']]-samples[:, idx['uniform1']])/(samples[:, idx['any_click']]-samples[:, idx['uniform1']])
    result['fraction_of_label_oracle_headroom_used'] = interval(headroom, boot_headroom)
    return result


def summarize(groups, config):
    large = groups.candidates.ge(config['large_min_candidates'])
    random = groups.displayrandom.eq(1)
    masks = {'all_large': large, 'all_small': ~large, 'randomized_large': random & large,
             'randomized_small': random & ~large, 'all_2_to_5': groups.candidates.le(5),
             'all_6_to_10': groups.candidates.between(6, 10),
             'contiguous_large': large & groups.rank_contiguous}
    reports = {name: cohort_summary(groups.loc[mask], config) for name, mask in masks.items()}
    interactions = {}
    for name, mask in [('all', np.ones(len(groups), bool)), ('randomized', random)]:
        part = groups.loc[mask].copy()
        big = part.candidates.ge(config['large_min_candidates'])
        part['large_n'], part['small_n'] = big.astype(float), (~big).astype(float)
        delta = part.full1-part.uniform1
        part['large_delta'], part['small_delta'] = delta*big, delta*(~big)
        point, draws = bootstrap_request_means(part, ['large_n', 'small_n', 'large_delta', 'small_delta'],
                                               config['bootstrap_repeats'], config['bootstrap_seed'])
        with np.errstate(divide='ignore', invalid='ignore'):
            estimate = point[2]/point[0]-point[3]/point[1]
            samples = draws[:, 2]/draws[:, 0]-draws[:, 3]/draws[:, 1]
        interactions[name] = interval(estimate, samples, 1000)
    ql, qs = reports['all_large']['click_request_prevalence'], reports['all_small']['click_request_prevalence']
    dl = reports['all_large']['conditional_on_any_click']['gap_top1']
    ds = reports['all_small']['conditional_on_any_click']['gap_top1']
    decomposition = {'prevalence_component_per1000': 1000*(ql-qs)*(dl+ds)/2,
                     'conditional_gap_component_per1000': 1000*(dl-ds)*(ql+qs)/2,
                     'total_large_minus_small_per1000': 1000*(ql*dl-qs*ds),
                     'common_click_prevalence': float(groups.any_click.mean()),
                     'common_prevalence_large_effect_per1000': float(1000*groups.any_click.mean()*dl),
                     'common_prevalence_small_effect_per1000': float(1000*groups.any_click.mean()*ds),
                     'interpretation': 'Exact symmetric arithmetic decomposition, not causal attribution.'}
    np.testing.assert_allclose(decomposition['prevalence_component_per1000']+decomposition['conditional_gap_component_per1000'],
                               decomposition['total_large_minus_small_per1000'], atol=1e-12)
    return {'cohorts': reports, 'large_minus_small_interaction': interactions, 'decomposition': decomposition}


def concentration(groups, frame, credits, config):
    large = groups.loc[groups.candidates.ge(config['large_min_candidates'])].copy()
    large['delta'] = large.full1-large.uniform1
    user = large.groupby('user_id').delta.agg(['sum', 'size'])
    total = large.delta.sum()
    loo = 1000*(total-user['sum'])/(len(large)-user['size'])
    positive_user = user['sum'].clip(lower=0).sort_values(ascending=False)
    excluded = user['sum'].nlargest(5).index
    rest = large.loc[~large.user_id.isin(excluded)]
    work = frame[META].copy().assign(credit=credits)
    work = work.loc[work.impression_id.isin(large.impression_id)]
    product = work.groupby('product_id').credit.sum().sort_values(ascending=False)
    return {'users_with_positive_net_delta': int(user['sum'].gt(1e-12).sum()),
            'users_with_negative_net_delta': int(user['sum'].lt(-1e-12).sum()),
            'top5_share_of_positive_user_net_contributions': float(positive_user.head(5).sum()/positive_user.sum()),
            'leave_one_user_out_effect_per1000_min': float(loo.min()), 'leave_one_user_out_effect_per1000_max': float(loo.max()),
            'excluding_five_most_positive_users_effect_per1000': float((rest.full1-rest.uniform1).mean()*1000),
            'full_selected_click_credit': float(product.sum()), 'clicked_products': int(work.loc[work.click.eq(1)].product_id.nunique()),
            'products_with_selected_click_credit': int(product.gt(0).sum()),
            'top5_product_share_of_selected_click_credit': float(product.head(5).sum()/product.sum()),
            'share_selected_click_credit_at_original_rank1': float(work.loc[work['rank'].eq(1), 'credit'].sum()/product.sum()),
            'share_all_observed_clicks_at_original_rank1': float(work.loc[work['rank'].eq(1), 'click'].sum()/work.click.sum()),
            'note': 'Influence diagnostics chosen using observed outcomes; not an alternative unbiased success test.'}


def thinning(frame, predictions, config):
    """Label-blind candidate thinning; retain original scores and outcomes."""
    settings = config['diagnostic_thinning']
    rng = np.random.default_rng(settings['seed'])
    size, repeats = settings['retained_candidates'], settings['repeats']
    rows = []
    all_labels = frame.click.to_numpy()
    all_scores = predictions[[f'full-s{s}' for s in config['seeds']]].to_numpy()
    for impression, ix in frame.groupby('impression_id', sort=True).indices.items():
        if len(ix) < config['large_min_candidates']:
            continue
        y, p = all_labels[ix], all_scores[ix]
        any_click, full, uniform = [], [], []
        for _ in range(repeats):
            selected = rng.choice(len(ix), size=size, replace=False)
            labels = y[selected]
            any_click.append(float(labels.sum() > 0))
            uniform.append(float(labels.mean()))
            full.append(float(np.mean([topk_weights(p[selected, s], 1) @ labels for s in range(len(config['seeds']))])))
        rows.append({'impression_id': int(impression), 'user_id': int(frame.iloc[ix[0]].user_id),
                     'any_click': float(np.mean(any_click)), 'full1': float(np.mean(full)), 'uniform1': float(np.mean(uniform))})
    result = pd.DataFrame(rows)
    q = result.any_click.mean()
    return {'requests': len(result), 'retained_candidates': size, 'draws_per_request': repeats,
            'click_request_prevalence': float(q), 'full_top1_per1000': float(result.full1.mean()*1000),
            'uniform_top1_per1000': float(result.uniform1.mean()*1000),
            'delta_per1000': float((result.full1-result.uniform1).mean()*1000),
            'conditional_full_top1': float(result.full1.mean()/q),
            'conditional_uniform_top1': float(result.uniform1.mean()/q),
            'note': 'Same original large requests; random subsets are drawn without labels or scores. Original click labels stay fixed. Monte Carlo diagnostic, not live effects or an extra independent sample.'}


def split_frames(frame, assignments, fold):
    if assignments.user_id.duplicated().any() or set(assignments.user_id) != set(frame.user_id):
        raise ValueError('User assignments must cover the pool exactly once')
    mapped = assignments.set_index('user_id')
    held = frame.user_id.map(mapped.outer_fold).eq(fold)
    fitted = ~held & frame.user_id.map(mapped[f'inner_fit_{fold}'])
    validated = ~held & ~frame.user_id.map(mapped[f'inner_fit_{fold}'])
    parts = [frame.loc[m].reset_index(drop=True) for m in [fitted, validated, held]]
    if any(p.empty for p in parts):
        raise ValueError('Every nested split must contain rows')
    for key in ['user_id', 'impression_id']:
        sets = [set(p[key]) for p in parts]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Nested split leakage')
    return parts


def run(config_path, output, models, tracking_uri=None):
    for path in [output, models]:
        if path.exists():
            raise FileExistsError(path)
    config = read_json(config_path)
    if config['large_min_candidates'] != 11 or config['outer_folds'] != 3 or config['variants'] != ['full', 'ad_only']:
        raise ValueError('Fixed replication contract changed')
    baseline_path = Path(config['baseline_config'])
    baseline = read_json(baseline_path)
    data_dir = Path(config['data_dir'])
    manifest = read_json(data_dir/'manifest.json')
    for name in ['train.parquet', 'validation.parquet']:
        if sha256(data_dir/name) != manifest['files'][name]['sha256']:
            raise ValueError('Data changed: '+name)
    parent_protocol = Path(config['parent_business_metrics']).parent/'protocol.json'
    parent = read_json(parent_protocol)
    existing_models = parent['model_sha256']
    for name, digest in existing_models.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Existing model changed: '+name)
    input_paths = [config_path, baseline_path, Path(__file__), data_dir/'train.parquet', data_dir/'validation.parquet',
                   Path(config['discovery_predictions']), Path(config['parent_business_metrics']),
                   Path('src/fairjob_ranker/diagnostics/business_validation.py'), Path('src/fairjob_ranker/features.py')]
    hashes = {str(path): sha256(path) for path in input_paths}
    output.mkdir(parents=True)
    models.mkdir(parents=True)
    write_json(output/'protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
               'input_sha256': hashes, 'existing_model_sha256': existing_models, 'original_test_read': False,
               'fresh_external_data': False, 'new_user_partitions_and_models': True})

    discovery = pd.read_parquet(data_dir/'validation.parquet', columns=META).reset_index(drop=True)
    old_predictions = pd.read_parquet(config['discovery_predictions'])
    pd.testing.assert_frame_equal(discovery, old_predictions[META].reset_index(drop=True))
    discovery_groups, credits = seed_mean_outcomes(discovery, old_predictions, config['seeds'])
    discovery_report = summarize(discovery_groups, config)
    discovery_report['concentration'] = concentration(discovery_groups, discovery, credits, config)
    discovery_report['label_blind_thinning'] = thinning(discovery, old_predictions, config)
    write_json(output/'discovery.json', discovery_report)
    discovery_groups.to_parquet(output/'discovery_outcomes.parquet', index=False)
    print('Discovery decomposition and label-blind thinning complete.', flush=True)

    pool = pd.read_parquet(data_dir/'train.parquet')
    if set(pool.user_id) & set(discovery.user_id) or set(pool.impression_id) & set(discovery.impression_id):
        raise ValueError('Discovery users overlap replication pool')
    assignments = user_assignments(pool, config)
    assignments.to_parquet(output/'user_assignments.parquet', index=False)
    if set(assignments.outer_fold) != set(range(config['outer_folds'])):
        raise ValueError('Missing outer fold')
    mlflow = None
    if tracking_uri:
        import mlflow as tracking
        mlflow = tracking
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['name'])
    fold_reports, oof_groups, oof_predictions, audits, runs = {}, [], [], [], []
    for fold in range(config['outer_folds']):
        fit, inner, held = split_frames(pool, assignments, fold)
        folder = models/f'fold-{fold}'
        folder.mkdir()
        transformer = FeatureTransformer.fit(fit)
        transformer.save(folder/'features.json')
        xf, xi = transformer.transform(fit), transformer.transform(inner)
        audit = {'fold': fold, 'fit_rows': len(fit), 'inner_rows': len(inner), 'held_rows': len(held),
                 'fit_users': int(fit.user_id.nunique()), 'inner_users': int(inner.user_id.nunique()),
                 'held_users': int(held.user_id.nunique()), 'overlap_users': 0, 'overlap_impressions': 0,
                 'transformer_sha256': sha256(folder/'features.json'), 'outer_labels_used_for_early_stopping': False,
                 'discovery_user_overlap': 0}
        write_json(folder/'split_audit.json', audit)
        audits.append(audit)
        fold_runs = []
        for variant in config['variants']:
            columns = FEATURES if variant == 'full' else AD_FEATURES
            cats = [c for c in CATEGORICAL if c in columns]
            for seed in config['seeds']:
                params = {**baseline['params'], **{k: seed for k in ['seed', 'feature_fraction_seed', 'bagging_seed', 'data_random_seed']}}
                history = {}
                started = perf_counter()
                booster = lgb.train(params, lgb.Dataset(xf[columns], label=fit.click, categorical_feature=cats),
                                    num_boost_round=baseline['num_boost_round'],
                                    valid_sets=[lgb.Dataset(xi[columns], label=inner.click, categorical_feature=cats)],
                                    valid_names=['inner'], callbacks=[lgb.early_stopping(baseline['early_stopping_rounds'], verbose=False),
                                                                    lgb.record_evaluation(history)])
                name = f'{variant}-s{seed}'
                model_path = folder/(name+'.txt')
                booster.save_model(str(model_path), num_iteration=booster.best_iteration)
                reloaded = lgb.Booster(model_file=str(model_path))
                np.testing.assert_array_equal(booster.predict(xi[columns].iloc[:1000], num_threads=4),
                                              reloaded.predict(xi[columns].iloc[:1000], num_threads=4))
                row = {'fold': fold, 'variant': variant, 'seed': seed, 'model': str(model_path),
                       'model_sha256': sha256(model_path), 'best_iteration': booster.best_iteration,
                       'inner_logloss': float(booster.best_score['inner']['binary_logloss']),
                       'training_seconds': perf_counter()-started}
                write_json(folder/(name+'-history.json'), history)
                if mlflow:
                    with mlflow.start_run(run_name=f'fold-{fold}-{name}') as tracked:
                        row['mlflow_run_id'] = tracked.info.run_id
                        mlflow.log_params({'fold': fold, 'variant': variant, 'seed': seed, 'fit_users': audit['fit_users'],
                                           'held_users': audit['held_users'], 'protocol_sha256': sha256(output/'protocol.json')})
                        mlflow.log_metrics({'best_iteration': row['best_iteration'], 'inner_logloss': row['inner_logloss']})
                        mlflow.log_artifact(str(model_path))
                        mlflow.log_artifact(str(folder/'split_audit.json'))
                fold_runs.append(row)
                runs.append(row)
                print(f"Fold {fold} {name}: trees={row['best_iteration']}, inner LogLoss={row['inner_logloss']:.6f}", flush=True)
                write_json(output/'progress.json', {'trained_models': len(runs), 'fold': fold, 'stage': 'fit'})
        write_json(folder/'frozen_models.json', {'runs': fold_runs, 'outer_scored': False,
                                               'transformer_sha256': audit['transformer_sha256']})
        # All six checkpoints are frozen before held-out features are scored.
        xh = FeatureTransformer.load(folder/'features.json').transform(held)
        prediction = held[META].copy()
        for row in fold_runs:
            columns = FEATURES if row['variant'] == 'full' else AD_FEATURES
            booster = lgb.Booster(model_file=row['model'])
            prediction[f"{row['variant']}-s{row['seed']}"] = booster.predict(xh[columns], num_threads=4)
        groups, _ = seed_mean_outcomes(held, prediction, config['seeds'])
        groups['outer_fold'] = fold
        prediction['outer_fold'] = fold
        oof_groups.append(groups)
        oof_predictions.append(prediction)
        prediction.to_parquet(output/f'fold-{fold}-predictions.parquet', index=False)
        fold_reports[str(fold)] = summarize(groups, config)
        write_json(output/f'fold-{fold}-metrics.json', fold_reports[str(fold)])
        delta = fold_reports[str(fold)]['cohorts']['all_large']['comparisons']['full-uniform-top1']
        print(f"Fold {fold} held-out n>=11: {delta['estimate']:+.3f} /1000, CI={delta['ci95']}", flush=True)
        if mlflow:
            for row in fold_runs:
                with mlflow.start_run(run_id=row['mlflow_run_id']):
                    mlflow.log_metrics({'fold_seed_mean_large_top1_delta_per1000': delta['estimate']})
    combined = pd.concat(oof_groups, ignore_index=True).sort_values('impression_id').reset_index(drop=True)
    predictions = pd.concat(oof_predictions, ignore_index=True)
    if predictions.impression_id.nunique() != pool.impression_id.nunique() or len(predictions) != len(pool):
        raise ValueError('Incomplete OOF coverage')
    if predictions.duplicated(['impression_id', 'product_id']).any() or predictions.groupby('user_id').outer_fold.nunique().gt(1).any():
        raise ValueError('Repeated OOF user or row')
    pd.testing.assert_frame_equal(pool[META].sort_values(['impression_id', 'product_id']).reset_index(drop=True),
                                  predictions[META].sort_values(['impression_id', 'product_id']).reset_index(drop=True))
    combined.to_parquet(output/'oof_outcomes.parquet', index=False)
    pooled = summarize(combined, config)
    delta = pooled['cohorts']['all_large']['comparisons']['full-uniform-top1']
    fold_positive = [fold_reports[str(f)]['cohorts']['all_large']['comparisons']['full-uniform-top1']['estimate'] > 0
                     for f in range(config['outer_folds'])]
    result = {'config': config, 'protocol_sha256': sha256(output/'protocol.json'), 'runs': runs, 'split_audits': audits,
              'folds': fold_reports, 'pooled_oof': pooled, 'original_test_read': False, 'tracking_uri': tracking_uri,
              'replication_pool': {'rows': len(pool), 'users': int(pool.user_id.nunique()), 'requests': int(pool.impression_id.nunique()),
                                   'discovery_user_overlap': 0},
              'all_fold_large_effects_positive': bool(all(fold_positive)),
              'large_effect_replication_supported': bool(all(fold_positive) and delta['ci95'][0] > 0),
              'larger_effect_than_small_supported': bool(pooled['large_minus_small_interaction']['all']['ci95'][0] > 0),
              'checks': ['all_nested_users_and_impressions_disjoint', 'discovery_users_excluded', 'checkpoints_frozen_before_outer_scoring',
                         'all_original_train_rows_evaluated_once', 'model_reload_agreement']}
    for name, digest in hashes.items():
        assert sha256(Path(name)) == digest
    for name, digest in existing_models.items():
        assert sha256(Path(name)) == digest
    for row in runs:
        assert sha256(Path(row['model'])) == row['model_sha256']
    result['checks'].append('sources_data_and_existing_models_unchanged')
    write_json(output/'metrics.json', result)
    (output/'progress.json').unlink()
    print('Completed candidate-count reproduction:', output, flush=True)
    return result
