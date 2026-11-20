"""A fixed, matched user-split experiment on checkpoint and ranking objectives."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import CATEGORICAL, FEATURES
from .business_validation import bootstrap_request_means, interval
from .candidate_count_reproduction import META, split_frames, vector_outcomes


class Top1Metric:
    """Fast tie-expected top1, retaining zero-click requests in its denominator."""

    def __init__(self, frame):
        self.codes, ids = pd.factorize(frame.impression_id, sort=False)
        self.n = len(ids)
        self.multi = np.bincount(self.codes, minlength=self.n) >= 2
        self.labels = frame.click.to_numpy(dtype=float)
        if not self.multi.any() or not np.isin(self.labels, [0, 1]).all():
            raise ValueError('Binary labels and multi-candidate requests required')

    def value(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != self.labels.shape or not np.isfinite(scores).all():
            raise ValueError('Aligned finite scores required')
        maximum = np.full(self.n, -np.inf)
        np.maximum.at(maximum, self.codes, scores)
        ties = scores == maximum[self.codes]
        counts = np.bincount(self.codes, weights=ties, minlength=self.n)
        clicks = np.bincount(self.codes, weights=ties*self.labels, minlength=self.n)
        return float((clicks/counts)[self.multi].mean())

    def __call__(self, predictions, dataset):
        if not np.array_equal(dataset.get_label(), self.labels):
            raise ValueError('Evaluation labels no longer align')
        return 'tie_top1', self.value(predictions), True


def grouped_frame(frame, seed=None):
    # Training tie order is independent of labels, historical rank and display flag.
    work = frame if seed is None else frame.iloc[np.random.default_rng(seed).permutation(len(frame))]
    work = work.sort_values('impression_id', kind='stable').reset_index(drop=True)
    sizes = work.groupby('impression_id', sort=False).size().to_numpy(dtype=np.int32)
    assert sizes.sum() == len(work)
    assert np.array_equal(np.repeat(work.impression_id.unique(), sizes), work.impression_id)
    return work, sizes


def balanced_query_weights(frame):
    groups = frame.groupby('impression_id').agg(user=('user_id', 'first'), n=('click', 'size'), c=('click', 'sum'))
    informative = groups.c.gt(0) & groups.c.lt(groups.n)
    selected = groups.loc[informative]
    counts = selected.groupby('user').size()
    if counts.empty:
        raise ValueError('No informative training queries')
    query_weights = 1 / selected.user.map(counts)
    query_weights /= query_weights.mean()
    all_weights = pd.Series(1., index=groups.index)
    all_weights.loc[selected.index] = query_weights
    weights = frame.impression_id.map(all_weights).to_numpy()
    per_user = query_weights.groupby(selected.user).sum()
    np.testing.assert_allclose(per_user, len(selected)/len(counts), rtol=1e-12)
    audit = {'informative_queries': len(selected), 'informative_users': len(counts),
             'informative_rows': int(selected.n.sum()), 'all_queries': len(groups),
             'query_weight_min': float(query_weights.min()), 'query_weight_max': float(query_weights.max()),
             'informative_query_weight_mean': float(query_weights.mean()),
             'per_user_query_weight_sum_min': float(per_user.min()), 'per_user_query_weight_sum_max': float(per_user.max())}
    return weights, audit


def outcomes(frame, predictions, config):
    groups = None
    for variant in ['baseline'] + config['variants']:
        for seed in config['seeds']:
            result, _ = vector_outcomes(frame, predictions[f'{variant}-s{seed}'])
            if groups is None:
                groups = result.drop(columns=['capture1', 'capture3'])
            np.testing.assert_array_equal(groups.impression_id, result.impression_id)
            for k in [1, 3]:
                groups[f'{variant}_s{seed}_{k}'] = result[f'capture{k}'].to_numpy()
        for k in [1, 3]:
            groups[f'{variant}{k}'] = groups[[f'{variant}_s{s}_{k}' for s in config['seeds']]].mean(axis=1)
    return groups


def summarize_cohort(groups, config):
    names = ['baseline'] + config['variants'] + ['uniform']
    columns = [f'{n}{k}' for n in names for k in [1, 3]]
    point, samples = bootstrap_request_means(groups, columns, config['bootstrap_repeats'], config['bootstrap_seed'])
    index = {name: i for i, name in enumerate(columns)}
    comparisons = [(v, 'baseline') for v in config['variants']] + [
        ('rank_top1', 'bce_top1'), ('rank_user_top1', 'rank_top1'), ('rank_top1', 'uniform'), ('baseline', 'uniform')]
    results = {}
    for a, b in comparisons:
        for k in [1, 3]:
            ai, bi = index[f'{a}{k}'], index[f'{b}{k}']
            delta = groups[f'{a}{k}']-groups[f'{b}{k}']
            results[f'{a}-{b}-top{k}'] = {
                **interval(point[ai]-point[bi], samples[:, ai]-samples[:, bi], 1000),
                'positive_requests': int(delta.gt(1e-12).sum()), 'negative_requests': int(delta.lt(-1e-12).sum())}
    # Same bootstrap engine now sees one row per user: equal-user descriptive utility.
    macro = groups.groupby('user_id')[columns].mean().reset_index()
    mp, ms = bootstrap_request_means(macro, columns, config['bootstrap_repeats'], config['bootstrap_seed'])
    user_macro = {v: interval(mp[index[v+'1']]-mp[index['baseline1']],
                              ms[:, index[v+'1']]-ms[:, index['baseline1']], 1000) for v in config['variants']}
    return {'requests': len(groups), 'users': int(groups.user_id.nunique()),
            'observed_clicks': int(groups.clicks.sum()), 'positive_requests': int(groups.any_click.sum()),
            'zero_click_requests': int(groups.any_click.eq(0).sum()),
            'top1_per1000': {n: float(point[index[n+'1']]*1000) for n in names},
            'top3_per1000': {n: float(point[index[n+'3']]*1000) for n in names},
            'comparisons': results, 'user_macro_minus_baseline_per1000': user_macro,
            'per_seed_delta_per1000': {v: {str(s): float((groups[f'{v}_s{s}_1']-groups[f'baseline_s{s}_1']).mean()*1000)
                                                 for s in config['seeds']} for v in config['variants']}}


def summarize(groups, config):
    return {name: summarize_cohort(groups.loc[mask], config) for name, mask in {
        'all_multi': np.ones(len(groups), dtype=bool), 'randomized_multi': groups.displayrandom.eq(1),
        'all_large': groups.candidates.ge(11), 'randomized_large': groups.candidates.ge(11) & groups.displayrandom.eq(1)
    }.items()}


def run(config_path, output, models, tracking_uri=None):
    if output.exists() or models.exists():
        raise FileExistsError('Use new output/model directories for a new protocol')
    config, baseline = read_json(config_path), read_json(Path(read_json(config_path)['baseline_config']))
    parent = Path(config['parent_report'])
    parent_metrics, parent_protocol = read_json(parent/'metrics.json'), read_json(parent/'protocol.json')
    if config['variants'] != ['bce_top1', 'rank_top1', 'rank_user_top1'] or config['primary_variant'] != 'rank_top1':
        raise ValueError('Fixed experiment arms changed')
    for path, digest in {**parent_protocol['input_sha256'], **parent_protocol['existing_model_sha256']}.items():
        # No original validation file is read; its previously verified hash is not rechecked here.
        if Path(path).name != 'validation.parquet' and sha256(Path(path)) != digest:
            raise ValueError('Frozen parent input changed: '+path)
    baseline_runs = [r for r in parent_metrics['runs'] if r['variant'] == 'full']
    frozen = {r['model']: r['model_sha256'] for r in parent_metrics['runs']}
    for fold in range(config['outer_folds']):
        for path in [parent/f'fold-{fold}-predictions.parquet', Path(config['parent_models'])/f'fold-{fold}'/'features.json']:
            frozen[str(path)] = sha256(path)
    for path, digest in frozen.items():
        assert sha256(Path(path)) == digest
    paths = [config_path, Path(config['baseline_config']), Path(__file__), Path('scripts/ranking_objective_reproduction.py'),
             Path('src/fairjob_ranker/diagnostics/candidate_count_reproduction.py'), Path('src/fairjob_ranker/diagnostics/business_validation.py'),
             Path('src/fairjob_ranker/features.py'), parent/'user_assignments.parquet', parent/'metrics.json', parent/'protocol.json',
             Path(config['data_dir'])/'train.parquet']
    hashes = {str(p): sha256(p) for p in paths}
    output.mkdir(parents=True)
    models.mkdir(parents=True)
    write_json(output/'protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'input_sha256': hashes, 'existing_artifact_sha256': frozen, 'original_test_read': False,
        'original_validation_read': False, 'reused_development_users': True})
    pool = pd.read_parquet(Path(config['data_dir'])/'train.parquet')
    assignments = pd.read_parquet(parent/'user_assignments.parquet')
    tracking = None
    if tracking_uri:
        import mlflow as tracking
        tracking.set_tracking_uri(tracking_uri)
        tracking.set_experiment(config['name'])
    runs, audits = [], []
    for fold in range(config['outer_folds']):
        fit, inner, held = split_frames(pool, assignments, fold)
        folder = models/f'fold-{fold}'
        folder.mkdir()
        transformer_path = Path(config['parent_models'])/f'fold-{fold}'/'features.json'
        transformer = FeatureTransformer.load(transformer_path)
        xf, xi = transformer.transform(fit), transformer.transform(inner)
        _, wa = balanced_query_weights(fit)
        audit = {**wa, 'fold': fold, 'fit_rows': len(fit), 'inner_rows': len(inner), 'held_rows': len(held),
                 'fit_users': int(fit.user_id.nunique()), 'inner_users': int(inner.user_id.nunique()), 'held_users': int(held.user_id.nunique()),
                 'transformer': str(transformer_path), 'transformer_sha256': sha256(transformer_path),
                 'user_and_impression_overlap': 0, 'outer_used_for_selection': False}
        write_json(folder/'split_audit.json', audit)
        audits.append(audit)
        inner_rank, inner_sizes = grouped_frame(inner)
        xi_rank = transformer.transform(inner_rank)
        for seed in config['seeds']:
            fit_rank, fit_sizes = grouped_frame(fit, seed)
            xf_rank = transformer.transform(fit_rank)
            weights, _ = balanced_query_weights(fit_rank)
            for variant in config['variants']:
                is_rank = variant != 'bce_top1'
                train_frame, eval_frame = (fit_rank, inner_rank) if is_rank else (fit, inner)
                train_x, eval_x = (xf_rank, xi_rank) if is_rank else (xf, xi)
                params = {**baseline['params'], 'metric': 'None',
                          **{k: seed for k in ['seed', 'feature_fraction_seed', 'bagging_seed', 'data_random_seed']}}
                if is_rank:
                    params.update(config['ranking_params'])
                train = lgb.Dataset(train_x[FEATURES], label=train_frame.click, categorical_feature=CATEGORICAL,
                    group=fit_sizes if is_rank else None, weight=weights if variant == 'rank_user_top1' else None)
                valid = lgb.Dataset(eval_x[FEATURES], label=eval_frame.click, categorical_feature=CATEGORICAL,
                    group=inner_sizes if is_rank else None, reference=train)
                metric, history, started = Top1Metric(eval_frame), {}, perf_counter()
                booster = lgb.train(params, train, num_boost_round=baseline['num_boost_round'], valid_sets=[valid],
                    valid_names=['inner'], feval=metric, callbacks=[lgb.early_stopping(baseline['early_stopping_rounds'],
                    first_metric_only=True, verbose=False), lgb.record_evaluation(history)])
                path = folder/f'{variant}-s{seed}.txt'
                booster.save_model(str(path), num_iteration=booster.best_iteration)
                restored = lgb.Booster(model_file=str(path))
                np.testing.assert_array_equal(booster.predict(eval_x.iloc[:1000], num_threads=4),
                                              restored.predict(eval_x.iloc[:1000], num_threads=4))
                prefix_error = None
                if not is_rank:
                    old = next(r for r in baseline_runs if r['fold'] == fold and r['seed'] == seed)
                    old_model = lgb.Booster(model_file=old['model'])
                    length = min(old_model.num_trees(), restored.num_trees())
                    prefix_error = float(np.max(np.abs(old_model.predict(eval_x.iloc[:1000], num_iteration=length, num_threads=4)
                                        -restored.predict(eval_x.iloc[:1000], num_iteration=length, num_threads=4))))
                    assert prefix_error < 1e-12, 'Selection-only control changed training trajectory'
                row = {'fold': fold, 'variant': variant, 'seed': seed, 'model': str(path), 'model_sha256': sha256(path),
                       'params': params, 'best_iteration': int(booster.best_iteration), 'evaluated_rounds': len(history['inner']['tie_top1']),
                       'inner_top1_per1000': float(booster.best_score['inner']['tie_top1']*1000),
                       'training_seconds': perf_counter()-started, 'bce_prefix_max_error': prefix_error,
                       'score_type': 'raw_ranking_score' if is_rank else 'binary_probability'}
                write_json(folder/f'{variant}-s{seed}-history.json', history)
                if tracking:
                    with tracking.start_run(run_name=f'fold-{fold}-{variant}-s{seed}') as tracked:
                        row['mlflow_run_id'] = tracked.info.run_id
                        tracking.log_params({'fold': fold, 'variant': variant, 'seed': seed, 'objective': params['objective'],
                            'checkpoint_metric': 'tie_top1', 'protocol_sha256': sha256(output/'protocol.json')})
                        tracking.log_metrics({k: row[k] for k in ['best_iteration', 'inner_top1_per1000']})
                        tracking.log_artifact(str(path))
                        tracking.log_artifact(str(folder/'split_audit.json'))
                runs.append(row)
                write_json(output/'progress.json', {'stage': 'fit_only', 'trained_models': len(runs), 'outer_scored': False})
                print(f"Trained {len(runs)}/27: fold{fold} {variant} s{seed}, trees={row['best_iteration']}, inner top1={row['inner_top1_per1000']:.3f}/1000", flush=True)
    # All new checkpoints are frozen across ALL folds before scoring any outer fold.
    write_json(models/'frozen_models.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'runs': runs, 'outer_scored': False, 'protocol_sha256': sha256(output/'protocol.json')})
    print('All 27 checkpoints frozen; beginning matched outer evaluation.', flush=True)
    folds, combined = {}, []
    for fold in range(config['outer_folds']):
        _, _, held = split_frames(pool, assignments, fold)
        transformer = FeatureTransformer.load(Path(audits[fold]['transformer']))
        xh = transformer.transform(held)
        old = pd.read_parquet(parent/f'fold-{fold}-predictions.parquet')
        pd.testing.assert_frame_equal(held[META], old[META].reset_index(drop=True))
        prediction = held[META].copy()
        for seed in config['seeds']:
            prediction[f'baseline-s{seed}'] = old[f'full-s{seed}'].to_numpy()
        for row in [r for r in runs if r['fold'] == fold]:
            booster = lgb.Booster(model_file=row['model'])
            prediction[f"{row['variant']}-s{row['seed']}"] = booster.predict(xh[FEATURES], num_threads=4)
        prediction['outer_fold'] = fold
        prediction.to_parquet(output/f'fold-{fold}-predictions.parquet', index=False)
        groups = outcomes(held, prediction, config)
        groups['outer_fold'] = fold
        combined.append(groups)
        folds[str(fold)] = summarize(groups, config)
        write_json(output/f'fold-{fold}-metrics.json', folds[str(fold)])
        delta = folds[str(fold)]['randomized_multi']['comparisons']['rank_top1-baseline-top1']
        print(f"Fold{fold} randomized primary delta={delta['estimate']:+.3f}/1000 CI={delta['ci95']}", flush=True)
    groups = pd.concat(combined).sort_values('impression_id').reset_index(drop=True)
    assert not groups.impression_id.duplicated().any()
    assert groups.groupby('user_id').outer_fold.nunique().eq(1).all()
    expected = pool.groupby('impression_id').size()
    assert set(groups.impression_id) == set(expected[expected >= 2].index)
    groups.to_parquet(output/'oof_outcomes.parquet', index=False)
    pooled = summarize(groups, config)
    primary = pooled['randomized_multi']['comparisons']['rank_top1-baseline-top1']
    guardrail = pooled['all_multi']['comparisons']['rank_top1-baseline-top1']
    all_positive = all(folds[str(f)]['randomized_multi']['comparisons']['rank_top1-baseline-top1']['estimate'] > 0 for f in range(3))
    result = {'config': config, 'runs': runs, 'split_audits': audits, 'folds': folds, 'pooled_oof': pooled,
              'protocol_sha256': sha256(output/'protocol.json'), 'frozen_models_sha256': sha256(models/'frozen_models.json'),
              'tracking_uri': tracking_uri, 'original_test_read': False, 'original_validation_read': False,
              'primary_decision': {'all_randomized_fold_points_positive': bool(all_positive),
                                   'pooled_randomized_lower_bound_positive': bool(primary['ci95'][0] > 0),
                                   'pooled_all_display_lower_bound_positive': bool(guardrail['ci95'][0] > 0),
                                   'supported': bool(all_positive and primary['ci95'][0] > 0 and guardrail['ci95'][0] > 0)},
              'oof_rows': len(pool), 'oof_users': int(pool.user_id.nunique()), 'oof_multi_requests': len(groups)}
    for path, digest in {**hashes, **frozen, **{r['model']: r['model_sha256'] for r in runs}}.items():
        assert sha256(Path(path)) == digest, path
    write_json(output/'metrics.json', result)
    (output/'progress.json').unlink()
    print('Completed:', output, result['primary_decision'], flush=True)
    return result
