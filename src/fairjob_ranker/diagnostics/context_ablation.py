"""Separate across-context click prediction from within-impression ad ranking."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..evaluation import click_metrics, rank_groups
from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import AD_FEATURES, CATEGORICAL, CONTEXT, FEATURES


FEATURE_SETS = {'context_only': CONTEXT, 'ad_only': AD_FEATURES, 'full': FEATURES}
RANK_METRICS = ['ndcg_tie_averaged', 'pair_accuracy', 'top1_click_tie_averaged']


def expected_ndcg(labels, scores, k=5):
    """Exact expected DCG under a uniform permutation inside each score tie."""
    labels, scores = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError('Aligned nonempty 1-D labels and scores required')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Binary labels and finite scores required')
    if k < 1:
        raise ValueError('Positive k required')
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    ordered_scores, ordered_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    discounts = np.zeros(len(labels))
    discounts[:min(k, len(labels))] = 1 / np.log2(np.arange(min(k, len(labels))) + 2)
    dcg = sum(ordered_labels[a:b].mean() * discounts[a:b].sum() for a, b in zip(starts, ends))
    ideal = discounts[:min(int(labels.sum()), k)].sum()
    return float(dcg / ideal)


def diagnostic_groups(frame, scores):
    """Label-based metrics on the same eligible impressions as the V1 evaluator."""
    scores = np.asarray(scores, dtype=float)
    legacy = rank_groups(frame, scores)
    if scores.ndim != 1 or len(scores) != len(frame):
        raise ValueError('One score per row required')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('Probabilities required')
    work = frame[['impression_id', 'user_id', 'displayrandom', 'click']].copy()
    work['score'] = scores
    work = work.loc[work.impression_id.isin(legacy.impression_id)]
    rows = []
    for impression, group in work.groupby('impression_id', sort=True):
        y, p = group.click.to_numpy(), group.score.to_numpy()
        differences = p[y == 1, None] - p[y == 0][None, :]
        top = p == p.max()
        rows.append({'impression_id': int(impression), 'user_id': int(group.user_id.iloc[0]),
                     'displayrandom': int(group.displayrandom.iloc[0]),
                     'candidates': len(group), 'clicks': int(y.sum()),
                     'ndcg_tie_averaged': expected_ndcg(y, p),
                     'random_ndcg': expected_ndcg(y, np.zeros(len(y))),
                     'pair_accuracy': float(((differences > 0) + .5 * (differences == 0)).mean()),
                     'pair_tie_rate': float((differences == 0).mean()),
                     'positive_negative_pairs': int(differences.size),
                     'top1_click_tie_averaged': float(y[top].mean()),
                     'score_span': float(np.ptp(p))})
    columns = ['impression_id', 'user_id', 'displayrandom', 'candidates', 'clicks',
               'ndcg_tie_averaged', 'random_ndcg', 'pair_accuracy', 'pair_tie_rate',
               'positive_negative_pairs', 'top1_click_tie_averaged', 'score_span']
    result = pd.DataFrame(rows, columns=columns)
    return result.merge(legacy[['impression_id', 'ndcg_at_5', 'mrr']], on='impression_id',
                        validate='one_to_one').rename(columns={'ndcg_at_5': 'legacy_ndcg_id_ties', 'mrr': 'legacy_mrr_id_ties'})


def span_summary(frame, scores, near_tie_threshold):
    work = frame[['impression_id']].assign(score=np.asarray(scores))
    stats = work.groupby('impression_id').score.agg(['size', 'min', 'max', 'nunique'])
    stats = stats.loc[stats['size'] >= 2]
    spans = stats['max'] - stats['min']
    return {'multi_candidate_impressions': len(stats),
            'all_scores_tied_fraction': float((spans == 0).mean()) if len(stats) else None,
            'near_tied_fraction': float((spans <= near_tie_threshold).mean()) if len(stats) else None,
            'nonzero_span_fraction': float((spans > 0).mean()) if len(stats) else None,
            'mean_span': float(spans.mean()) if len(stats) else None,
            'median_span': float(spans.median()) if len(stats) else None,
            'p90_span': float(spans.quantile(.9)) if len(stats) else None,
            'mean_distinct_scores': float(stats['nunique'].mean()) if len(stats) else None}


def evaluate_diagnostics(frame, scores, near_tie_threshold=1e-12):
    groups = diagnostic_groups(frame, scores)
    report = {}
    for cohort in ['all', 'randomized']:
        mask = np.ones(len(frame), dtype=bool) if cohort == 'all' else frame.displayrandom.eq(1).to_numpy()
        part = frame.loc[mask]
        gs = groups if cohort == 'all' else groups.loc[groups.displayrandom == 1]
        report[cohort] = {
            'click': click_metrics(part.click.to_numpy(), np.asarray(scores)[mask]),
            'ranking': {'eligible_impressions': len(gs), 'eligible_users': int(gs.user_id.nunique()),
                        **{c: float(gs[c].mean()) if len(gs) else None for c in
                           RANK_METRICS + ['random_ndcg', 'pair_tie_rate', 'legacy_ndcg_id_ties', 'legacy_mrr_id_ties']}},
            'score_discrimination': span_summary(part, np.asarray(scores)[mask], near_tie_threshold)}
    return report, groups


def feature_audit(frame, transformed):
    context_variations = frame.groupby('impression_id')[CONTEXT].nunique().gt(1).sum()
    ad_variations = frame.groupby('product_id')[AD_FEATURES].nunique().gt(1).sum()
    if context_variations.any() or ad_variations.any():
        raise ValueError('Observed feature roles violate the audited context/catalog contract')
    counts = frame.groupby('impression_id').size()
    multiple = counts[counts >= 2].index
    eligible = frame.groupby('impression_id').click.agg(['sum', 'size'])
    eligible_ids = eligible.loc[(eligible['size'] >= 2) & (eligible['sum'] > 0) & (eligible['sum'] < eligible['size'])].index
    collisions = {}
    for label, values in [('raw', frame[AD_FEATURES]), ('transformed', transformed[AD_FEATURES])]:
        values = values.copy()
        values['impression_id'] = frame.impression_id.to_numpy()
        unique_vectors = values.drop_duplicates().groupby('impression_id').size()
        collisions[label] = {
            'multi_impressions_all_ad_vectors_identical_fraction': float(unique_vectors.reindex(multiple).eq(1).mean()),
            'eligible_impressions_all_ad_vectors_identical_fraction': float(unique_vectors.reindex(eligible_ids).eq(1).mean())}
    user_varying = frame.groupby('user_id')[CONTEXT].nunique().gt(1).any(axis=1)
    return {'rows': len(frame), 'users': int(frame.user_id.nunique()), 'impressions': int(frame.impression_id.nunique()),
            'context_variations_within_impression': context_variations.to_dict(),
            'ad_variations_within_product': ad_variations.to_dict(),
            'users_with_context_changing_across_impressions': int(user_varying.sum()),
            'candidate_feature_collisions': collisions,
            'categorical_oov_rates': {c: float(transformed[c].eq(0).mean()) for c in CATEGORICAL}}


def bootstrap_mean_by_user(user_ids, values, repeats, seed):
    """Row/impression-weighted mean; whole users are the resampling units."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(user_ids) or repeats < 1 or not np.isfinite(values).all():
        raise ValueError('Finite aligned values and positive bootstrap repeats required')
    if not len(values):
        return None
    per_user = pd.DataFrame({'user_id': np.asarray(user_ids), 'v': values}).groupby('user_id').v.agg(['sum', 'size'])
    sums, counts = per_user['sum'].to_numpy(), per_user['size'].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        ids = rng.integers(0, len(per_user), len(per_user))
        samples[i] = sums[ids].sum() / counts[ids].sum()
    return {'mean_difference': float(values.mean()), 'ci95': [float(v) for v in np.quantile(samples, [.025, .975])],
            'users': len(per_user), 'observations': len(values)}


def logloss_rows(y, scores):
    p = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def paired_comparison(frame, candidate_groups, reference_groups, candidate_scores, reference_scores, repeats, seed):
    """Average metrics across paired seeds before cluster resampling, not ensemble predictions."""
    keys = ['impression_id', 'user_id', 'displayrandom']
    deltas = []
    for a, b in zip(candidate_groups, reference_groups, strict=True):
        paired = a.merge(b, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
        if len(paired) != len(a) or len(paired) != len(b):
            raise ValueError('Unmatched ranking cohorts')
        out = paired[keys].copy()
        for metric in RANK_METRICS:
            out[metric] = paired[metric+'_a'] - paired[metric+'_b']
        deltas.append(out)
    averaged = pd.concat(deltas).groupby(keys, as_index=False)[RANK_METRICS].mean()
    y = frame.click.to_numpy()
    ll = np.mean([logloss_rows(y, a) - logloss_rows(y, b)
                  for a, b in zip(candidate_scores, reference_scores, strict=True)], axis=0)
    report = {'unit': 'user', 'seed_count': len(deltas), 'bootstrap_repeats': repeats, 'bootstrap_seed': seed,
              'aggregation': 'Equal seed mean of paired metrics, then whole-user resampling; not a score ensemble',
              'all': {}, 'randomized': {}}
    for cohort in ['all', 'randomized']:
        gs = averaged if cohort == 'all' else averaged.loc[averaged.displayrandom == 1]
        for metric in RANK_METRICS:
            report[cohort][metric] = bootstrap_mean_by_user(gs.user_id, gs[metric], repeats, seed)
        mask = np.ones(len(frame), dtype=bool) if cohort == 'all' else frame.displayrandom.eq(1).to_numpy()
        report[cohort]['logloss'] = bootstrap_mean_by_user(frame.loc[mask, 'user_id'], ll[mask], repeats, seed)
    return report


def run_experiment(data_dir, config_path, model_root, output, tracking_uri=None):
    from ..v2.training import development_split
    import mlflow

    for path in [model_root, output]:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    config = read_json(config_path)
    lgb_config = read_json(Path(config['lightgbm_config']))
    manifest = read_json(data_dir/'manifest.json')
    if set(config['variants']) != set(FEATURE_SETS) or config['representative_seed'] not in config['seeds']:
        raise ValueError('Expected the three diagnostic feature sets and a fixed representative seed')
    original = pd.read_parquet(data_dir/'train.parquet')
    if sha256(data_dir/'train.parquet') != manifest['files']['train.parquet']['sha256']:
        raise ValueError('Training checksum mismatch')
    train, inner = development_split(original, config['development_split_salt'], config['inner_train_percent'])
    transformer = FeatureTransformer.fit(train)
    xt, xi = transformer.transform(train), transformer.transform(inner)
    audit = {'inner_train': feature_audit(train, xt), 'inner_validation': feature_audit(inner, xi)}
    model_root.mkdir(parents=True)
    output.mkdir(parents=True)
    transformer.save(model_root/'features.json')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['experiment_name'])
    preregistered = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                    'config_sha256': sha256(config_path), 'lightgbm_config': lgb_config,
                    'lightgbm_config_sha256': sha256(Path(config['lightgbm_config'])),
                    'dataset_manifest_sha256': sha256(data_dir/'manifest.json'),
                    'source_sha256': sha256(Path(__file__)), 'test_read': False,
                    'feature_sets': FEATURE_SETS, 'feature_audit': audit,
                    'environment': {name: version(name) for name in ['lightgbm', 'numpy', 'pandas', 'scikit-learn']}}
    write_json(output/'preregistered.json', preregistered)
    rows, boosters, predictions, groups = [], {}, {'inner': {}, 'outer': {}}, {'inner': {}, 'outer': {}}
    baseline_ctr = float(train.click.mean())
    for variant in config['variants']:
        columns = FEATURE_SETS[variant]
        categories = [c for c in columns if c in CATEGORICAL]
        for seed in config['seeds']:
            name = f'{variant}-s{seed}'
            params = {**lgb_config['params'], **{key: seed for key in ['seed', 'feature_fraction_seed', 'bagging_seed', 'data_random_seed']}}
            history = {}
            started = perf_counter()
            booster = lgb.train(params, lgb.Dataset(xt[columns], label=train.click, categorical_feature=categories),
                                num_boost_round=lgb_config['num_boost_round'],
                                valid_sets=[lgb.Dataset(xi[columns], label=inner.click, categorical_feature=categories)],
                                valid_names=['inner_validation'],
                                callbacks=[lgb.early_stopping(lgb_config['early_stopping_rounds'], verbose=False),
                                           lgb.record_evaluation(history)])
            seconds = perf_counter() - started
            directory = model_root/name
            directory.mkdir()
            booster.save_model(str(directory/'model.txt'), num_iteration=booster.best_iteration)
            loaded = lgb.Booster(model_file=str(directory/'model.txt'))
            scores = booster.predict(xi[columns], num_threads=params['num_threads'])
            reloaded = loaded.predict(xi[columns], num_threads=params['num_threads'])
            np.testing.assert_array_equal(scores, reloaded)
            if loaded.feature_name() != columns:
                raise ValueError('Saved feature contract mismatch')
            metrics, ranked = evaluate_diagnostics(inner, scores, config['near_tie_threshold'])
            row = {'name': name, 'variant': variant, 'seed': seed, 'features': columns,
                   'best_iteration': booster.best_iteration, 'training_seconds': seconds,
                   'params': params, 'model': str(directory/'model.txt'),
                   'model_sha256': sha256(directory/'model.txt'), 'transformer_sha256': sha256(model_root/'features.json'),
                   'inner': metrics, 'reload_max_abs_error': float(np.max(np.abs(scores-reloaded))),
                   'feature_gain': dict(zip(columns, map(float, booster.feature_importance(importance_type='gain'))))}
            write_json(directory/'metadata.json', row)
            write_json(directory/'training_history.json', history)
            if tracking_uri:
                with mlflow.start_run(run_name=name) as run:
                    row['mlflow_run_id'] = run.info.run_id
                    mlflow.log_params({**params, 'variant': variant, 'feature_count': len(columns),
                                       'config_sha256': preregistered['config_sha256']})
                    mlflow.log_metrics({'inner_logloss': metrics['all']['click']['logloss'],
                                        'inner_pair_accuracy': metrics['all']['ranking']['pair_accuracy'],
                                        'inner_ndcg_tie_averaged': metrics['all']['ranking']['ndcg_tie_averaged'],
                                        'best_iteration': booster.best_iteration})
                    mlflow.log_artifact(str(output/'preregistered.json'))
                    mlflow.log_artifact(str(model_root/'features.json'))
                    mlflow.log_artifacts(str(directory), artifact_path='model')
            rows.append(row)
            boosters[name] = loaded
            predictions['inner'][name], groups['inner'][name] = scores, ranked
            write_json(output/'progress.json', {'completed_models': len(rows), 'outer_read': False})
            print(f'{name}: trees={booster.best_iteration}, inner LogLoss={metrics["all"]["click"]["logloss"]:.6f}, pair={metrics["all"]["ranking"]["pair_accuracy"]:.4f}', flush=True)
    frozen = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'models': {r['name']: r['model_sha256'] for r in rows},
              'representative_seed': config['representative_seed'], 'no_seed_selection': True, 'outer_loaded': False}
    write_json(output/'frozen_models.json', frozen)
    # Outer labels are only read after every model and checkpoint is fixed.
    if sha256(data_dir/'validation.parquet') != manifest['files']['validation.parquet']['sha256']:
        raise ValueError('Validation checksum mismatch')
    outer = pd.read_parquet(data_dir/'validation.parquet')
    for key in ['user_id', 'impression_id']:
        if set(original[key]) & set(outer[key]):
            raise ValueError('Train/outer leakage')
    xo = transformer.transform(outer)
    audit['outer_validation'] = feature_audit(outer, xo)
    # Verify the catalog role across train/outer as well as within each split.
    if pd.concat([original[['product_id']+AD_FEATURES], outer[['product_id']+AD_FEATURES]]).groupby('product_id')[AD_FEATURES].nunique().gt(1).any().any():
        raise ValueError('Ad features changed across splits')
    for row in rows:
        name = row['name']
        scores = boosters[name].predict(xo[row['features']], num_threads=row['params']['num_threads'])
        row['outer'], ranked = evaluate_diagnostics(outer, scores, config['near_tie_threshold'])
        predictions['outer'][name], groups['outer'][name] = scores, ranked
        if tracking_uri:
            with mlflow.start_run(run_id=row['mlflow_run_id']):
                mlflow.log_metrics({'outer_logloss': row['outer']['all']['click']['logloss'],
                                    'outer_pair_accuracy': row['outer']['all']['ranking']['pair_accuracy'],
                                    'outer_ndcg_tie_averaged': row['outer']['all']['ranking']['ndcg_tie_averaged']})
                mlflow.log_dict(row, 'result.json')
    diagnostic_results = {}
    for split, frame in [('inner', inner), ('outer', outer)]:
        diagnostic_results[split] = {}
        constant = np.full(len(frame), baseline_ctr)
        constant_metrics, constant_groups = evaluate_diagnostics(frame, constant, config['near_tie_threshold'])
        predictions[split]['constant'] = constant
        groups[split]['constant'] = constant_groups
        diagnostic_results[split]['constant'] = constant_metrics
        for seed in config['seeds']:
            name = f'full-s{seed}'
            pooled = frame[['impression_id']].assign(score=predictions[split][name]).groupby('impression_id').score.transform('mean').to_numpy()
            pooled_name = f'full_group_mean-s{seed}'
            metrics, ranked = evaluate_diagnostics(frame, pooled, config['near_tie_threshold'])
            diagnostic_results[split][pooled_name] = metrics
            predictions[split][pooled_name], groups[split][pooled_name] = pooled, ranked
        table = frame[['user_id', 'impression_id', 'product_id', 'click', 'displayrandom', 'rank']].copy()
        for name, scores in predictions[split].items():
            table[name] = scores
        table.to_parquet(output/f'{split}_predictions.parquet', index=False)
        pd.concat([g.assign(model=name) for name, g in groups[split].items()]).to_parquet(output/f'{split}_ranking.parquet', index=False)
    comparisons = {}
    for split, frame in [('inner', inner), ('outer', outer)]:
        comparisons[split] = {}
        for candidate, reference in [('full', 'context_only'), ('ad_only', 'context_only'), ('full', 'ad_only'),
                                     ('full', 'full_group_mean'), ('context_only', 'constant')]:
            comparison_name = candidate+' - '+reference
            comparisons[split][comparison_name] = {}
            for scope, seeds in [('seed_mean', config['seeds']), ('representative_seed', [config['representative_seed']])]:
                names_a = [f'{candidate}-s{s}' for s in seeds]
                names_b = [f'{reference}-s{s}' if reference != 'constant' else 'constant' for s in seeds]
                comparisons[split][comparison_name][scope] = paired_comparison(
                    frame, [groups[split][n] for n in names_a], [groups[split][n] for n in names_b],
                    [predictions[split][n] for n in names_a], [predictions[split][n] for n in names_b],
                    config['bootstrap_repeats'], config['bootstrap_seed'])
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'preregistered_sha256': sha256(output/'preregistered.json'), 'test_read': False,
              'train_ctr': baseline_ctr, 'tracking_uri': tracking_uri, 'runs': rows,
              'diagnostic_controls': diagnostic_results, 'comparisons': comparisons, 'feature_audit': audit,
              'existing_champion_changed': False}
    # Check the exact full-feature seed-42 replication against the existing V2 model.
    existing = Path('models/v2/lightgbm-s42/model.txt')
    if existing.exists() and data_dir.resolve() == Path('data/processed/fairjob-v1').resolve():
        old = lgb.Booster(model_file=str(existing))
        previous = old.predict(xo[FEATURES], num_threads=4)
        difference = float(np.max(np.abs(previous-predictions['outer']['full-s42'])))
        result['existing_v2_replication'] = {'model_sha256': sha256(existing), 'max_abs_prediction_error': difference}
        np.testing.assert_allclose(previous, predictions['outer']['full-s42'], rtol=0, atol=1e-12)
    seed = config['representative_seed']
    context = groups['outer'][f'context_only-s{seed}'].set_index('impression_id')
    full = groups['outer'][f'full-s{seed}'].set_index('impression_id')
    eligible = full.loc[full.candidates.between(3, 8) & full.score_span.gt(0)].copy()
    eligible['ndcg_change'] = eligible.ndcg_tie_averaged-context.ndcg_tie_averaged
    examples = []
    for label, ascending in [('largest_gain', False), ('largest_loss', True)]:
        if eligible.empty:
            continue
        selected = eligible.reset_index().sort_values(['ndcg_change', 'impression_id'], ascending=[ascending, True]).iloc[0]
        impression = int(selected.impression_id)
        mask = outer.impression_id.eq(impression).to_numpy()
        table = outer.loc[mask, ['product_id', 'click', 'rank']+AD_FEATURES].copy()
        for variant in config['variants']:
            table[variant+'_score'] = predictions['outer'][f'{variant}-s{seed}'][mask]
        table = table.sort_values(['full_score', 'product_id'], ascending=[False, True])
        examples.append({'selection': label, 'impression_id': impression, 'displayrandom': int(selected.displayrandom),
                         'ndcg_change': float(selected.ndcg_change),
                         'context_ndcg': float(context.loc[impression, 'ndcg_tie_averaged']),
                         'full_ndcg': float(full.loc[impression, 'ndcg_tie_averaged']),
                         'rows': table.to_dict('records'), 'scope': 'Selected extreme for explanation, not representative evidence'})
    write_json(output/'examples.json', examples)
    write_json(output/'feature_audit.json', audit)
    write_json(output/'metrics.json', result)
    (output/'progress.json').unlink()
    print(f'Completed {len(rows)} models; saved {output}', flush=True)
    return result
