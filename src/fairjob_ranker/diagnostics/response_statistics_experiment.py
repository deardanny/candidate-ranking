"""Matched user-split evaluation of response features and unseen-ad selection."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import CATEGORICAL, FEATURES
from .business_validation import bootstrap_request_means, interval
from .candidate_count_reproduction import META, split_frames
from .ranking_objective_reproduction import outcomes
from .response_statistics import KEY_COLUMNS, ResponseStatistics, cross_fit_statistics, statistic_columns


def cold_status(frame, fit, inner):
    return pd.DataFrame({'fit_unseen': ~frame.product_id.isin(fit.product_id),
        'strict_unseen': ~frame.product_id.isin(pd.concat([fit.product_id, inner.product_id]))}, index=frame.index)


def selection_outcomes(predictions, config):
    groups = outcomes(predictions, predictions, config).set_index('impression_id')
    rows = predictions.copy()
    rows['cold_click'] = rows.click*rows.strict_unseen
    request_stats = rows.groupby('impression_id').agg(cold_candidates=('strict_unseen', 'sum'),
        fit_unseen_candidates=('fit_unseen', 'sum'), cold_clicks=('cold_click', 'sum'))
    groups = groups.join(request_stats)
    for variant in ['baseline'] + config['variants']:
        for seed in config['seeds']:
            score = rows[f'{variant}-s{seed}']
            maximum = score.groupby(rows.impression_id).transform('max')
            ties = score.eq(maximum)
            weights = ties / ties.groupby(rows.impression_id).transform('sum')
            cold_capture = (weights*rows.cold_click).groupby(rows.impression_id).sum()
            cold_selected = (weights*rows.strict_unseen).groupby(rows.impression_id).sum()
            groups[f'{variant}_s{seed}_cold1'] = cold_capture
            groups[f'{variant}_s{seed}_cold_selected1'] = cold_selected
        groups[f'{variant}_cold1'] = groups[[f'{variant}_s{s}_cold1' for s in config['seeds']]].mean(axis=1)
        groups[f'{variant}_cold_selected1'] = groups[[f'{variant}_s{s}_cold_selected1' for s in config['seeds']]].mean(axis=1)
    groups['uniform_cold1'] = groups.cold_clicks / groups.candidates
    groups['uniform_cold_selected1'] = groups.cold_candidates / groups.candidates
    return groups.reset_index()


def cohort_summary(groups, config):
    if groups.empty:
        return {'requests': 0, 'users': 0}
    names = ['baseline'] + config['variants'] + ['uniform']
    columns = [n+suffix for n in names for suffix in ['1', '3', '_cold1', '_cold_selected1']]
    point, draws = bootstrap_request_means(groups, columns, config['bootstrap_repeats'], config['bootstrap_seed'])
    ix = {c: i for i, c in enumerate(columns)}
    comparisons = {}
    for a, b in [('ad_stats', 'baseline'), ('ad_context_stats', 'baseline'), ('ad_context_stats', 'ad_stats')]:
        for suffix, metric in [('1', 'top1'), ('3', 'top3'), ('_cold1', 'cold_top1'), ('_cold_selected1', 'cold_selected')]:
            i, j = ix[a+suffix], ix[b+suffix]
            comparisons[f'{a}-{b}-{metric}'] = interval(point[i]-point[j], draws[:, i]-draws[:, j], 1000)
    return {'requests': len(groups), 'users': int(groups.user_id.nunique()), 'observed_clicks': int(groups.clicks.sum()),
        'requests_with_clicks': int(groups.any_click.sum()), 'zero_click_requests': int(groups.any_click.eq(0).sum()),
        'cold_candidates': int(groups.cold_candidates.sum()), 'cold_clicks': int(groups.cold_clicks.sum()),
        'requests_with_cold_clicks': int(groups.cold_clicks.gt(0).sum()),
        'top1_per1000': {n: float(point[ix[n+'1']]*1000) for n in names},
        'cold_top1_per1000': {n: float(point[ix[n+'_cold1']]*1000) for n in names},
        'cold_selected_fraction': {n: float(point[ix[n+'_cold_selected1']]) for n in names},
        'comparisons': comparisons,
        'per_seed_top1_minus_baseline': {n: {str(s): float((groups[f'{n}_s{s}_1']-groups[f'baseline_s{s}_1']).mean()*1000)
                                                for s in config['seeds']} for n in config['variants']}}


def summarize(groups, config):
    has_cold = groups.cold_candidates.gt(0)
    all_cold = groups.cold_candidates.eq(groups.candidates)
    random = groups.displayrandom.eq(1)
    masks = {'all_multi': np.ones(len(groups), bool), 'randomized_multi': random,
        'strict_has_cold': has_cold, 'randomized_strict_has_cold': random & has_cold,
        'strict_all_cold': all_cold, 'randomized_strict_all_cold': random & all_cold,
        'strict_mixed': has_cold & ~all_cold, 'no_strict_cold': ~has_cold,
        'fit_all_unseen': groups.fit_unseen_candidates.eq(groups.candidates)}
    return {label: cohort_summary(groups.loc[mask], config) for label, mask in masks.items()}


def probability_summary(predictions, config):
    result = {}
    for label, mask in {'all_rows': np.ones(len(predictions), bool), 'fit_seen_rows': ~predictions.fit_unseen,
        'fit_unseen_rows': predictions.fit_unseen, 'strict_unseen_rows': predictions.strict_unseen}.items():
        part = predictions.loc[mask]
        results = {}
        for name in ['baseline'] + config['variants']:
            ll, auc = [], []
            for seed in config['seeds']:
                p = part[f'{name}-s{seed}'].to_numpy()
                ll.append(float(log_loss(part.click, p, labels=[0, 1])))
                if part.click.nunique() == 2:
                    auc.append(float(roc_auc_score(part.click, p)))
            results[name] = {'logloss': float(np.mean(ll)), 'auc': float(np.mean(auc)) if auc else None}
        result[label] = {'rows': len(part), 'users': int(part.user_id.nunique()),
                         'products': int(part.product_id.nunique()), 'clicks': int(part.click.sum()), 'models': results}
    return result


def run(config_path, output, models, tracking_uri=None):
    if output.exists() or models.exists():
        raise FileExistsError('Refuse to overwrite experiment artifacts')
    config = read_json(config_path)
    parent = Path(config['parent_report'])
    parent_metrics = read_json(parent/'metrics.json')
    baseline = read_json(Path(config['baseline_config']))
    paths = [config_path, Path(config['baseline_config']), Path(__file__),
        Path('src/fairjob_ranker/diagnostics/response_statistics.py'), Path('scripts/response_statistics.py'),
        Path('src/fairjob_ranker/diagnostics/candidate_count_reproduction.py'),
        Path('src/fairjob_ranker/diagnostics/ranking_objective_reproduction.py'),
        Path('src/fairjob_ranker/diagnostics/business_validation.py'), Path('src/fairjob_ranker/features.py'),
        parent/'user_assignments.parquet', parent/'metrics.json', Path(config['data_dir'])/'train.parquet']
    hashes = {str(p): sha256(p) for p in paths}
    manifest = read_json(Path(config['data_dir'])/'manifest.json')
    assert hashes[str(Path(config['data_dir'])/'train.parquet')] == manifest['files']['train.parquet']['sha256']
    previous = read_json(Path('reports/ranking-objective-reproduction/protocol.json'))
    frozen = {**previous['existing_artifact_sha256']}
    prior_runs = read_json(Path('reports/ranking-objective-reproduction/metrics.json'))['runs']
    frozen.update({r['model']: r['model_sha256'] for r in prior_runs})
    for path, digest in frozen.items():
        assert sha256(Path(path)) == digest, path
    output.mkdir(parents=True)
    models.mkdir(parents=True)
    write_json(output/'protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'input_sha256': hashes, 'existing_artifact_sha256': frozen, 'original_test_read': False,
        'original_validation_read': False, 'outer_outcomes_seen_before_training': False,
        'development_pool_previously_used': True})
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
        response_fit, encoding_assignment, encoding_audit = cross_fit_statistics(fit, config, fold)
        response_fit.to_parquet(folder/'fit-response-oof.parquet', index=False)
        encoding_assignment.to_parquet(folder/'encoding-users.parquet', index=False)
        encoder = ResponseStatistics.fit(fit, config['product_smoothing'], config['category_smoothing'])
        encoder.save(folder/'response-store')
        response_inner = encoder.transform(inner[KEY_COLUMNS])
        xf = pd.concat([transformer.transform(fit), response_fit], axis=1)
        xi = pd.concat([transformer.transform(inner), response_inner], axis=1)
        assert len(xf) == len(fit) and len(xi) == len(inner)
        audit = {'fold': fold, 'fit_rows': len(fit), 'inner_rows': len(inner), 'held_rows': len(held),
            'fit_users': int(fit.user_id.nunique()), 'inner_users': int(inner.user_id.nunique()), 'held_users': int(held.user_id.nunique()),
            'transformer': str(transformer_path), 'transformer_sha256': sha256(transformer_path),
            'oof_stats_sha256': sha256(folder/'fit-response-oof.parquet'),
            'encoding_users_sha256': sha256(folder/'encoding-users.parquet'),
            'response_store_metadata_sha256': sha256(folder/'response-store/metadata.json'),
            'encoding_audits': encoding_audit, 'statistics_source': 'fit_only',
            'user_and_request_overlap': 0, 'model_fit_used_cross_fitted_statistics': True}
        write_json(folder/'split_audit.json', audit)
        audits.append(audit)
        for variant in config['variants']:
            columns = FEATURES + statistic_columns(variant)
            assert not set(columns) & {'click', 'user_id', 'product_id', 'rank', 'displayrandom', 'protected_attribute', 'senior'}
            for seed in config['seeds']:
                params = {**baseline['params'], **{k: seed for k in ['seed', 'feature_fraction_seed', 'bagging_seed', 'data_random_seed']}}
                history, started = {}, perf_counter()
                train = lgb.Dataset(xf[columns], label=fit.click, categorical_feature=CATEGORICAL)
                valid = lgb.Dataset(xi[columns], label=inner.click, categorical_feature=CATEGORICAL, reference=train)
                booster = lgb.train(params, train, num_boost_round=baseline['num_boost_round'], valid_sets=[valid],
                    valid_names=['inner'], callbacks=[lgb.early_stopping(baseline['early_stopping_rounds'], verbose=False), lgb.record_evaluation(history)])
                path = folder/f'{variant}-s{seed}.txt'
                booster.save_model(str(path), num_iteration=booster.best_iteration)
                restored = lgb.Booster(model_file=str(path))
                np.testing.assert_array_equal(booster.predict(xi[columns].iloc[:1000], num_threads=4),
                                              restored.predict(xi[columns].iloc[:1000], num_threads=4))
                row = {'fold': fold, 'variant': variant, 'seed': seed, 'feature_count': len(columns), 'model': str(path),
                    'model_sha256': sha256(path), 'best_iteration': int(booster.best_iteration),
                    'inner_logloss': float(booster.best_score['inner']['binary_logloss']), 'training_seconds': perf_counter()-started}
                write_json(folder/f'{variant}-s{seed}-history.json', history)
                if tracking:
                    with tracking.start_run(run_name=f'fold-{fold}-{variant}-s{seed}') as tracked:
                        row['mlflow_run_id'] = tracked.info.run_id
                        tracking.log_params({'fold': fold, 'variant': variant, 'seed': seed, 'features': len(columns),
                            'encoding_split_unit': 'user', 'protocol_sha256': sha256(output/'protocol.json')})
                        tracking.log_metrics({'best_iteration': row['best_iteration'], 'inner_logloss': row['inner_logloss']})
                        tracking.log_artifact(str(path))
                        tracking.log_artifact(str(folder/'split_audit.json'))
                runs.append(row)
                write_json(output/'progress.json', {'trained_models': len(runs), 'outer_scored': False})
                print(f"Trained {len(runs)}/18: fold{fold} {variant} s{seed}, trees={row['best_iteration']}, inner LL={row['inner_logloss']:.6f}", flush=True)
    write_json(models/'frozen_models.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': runs,
        'split_audits': audits, 'outer_scored': False, 'protocol_sha256': sha256(output/'protocol.json')})
    print('All18 models and fit-only stores frozen. Starting outer scoring.', flush=True)
    folds, combined, prediction_parts = {}, [], []
    for fold in range(config['outer_folds']):
        fit, inner, held = split_frames(pool, assignments, fold)
        folder = models/f'fold-{fold}'
        transformer = FeatureTransformer.load(Path(audits[fold]['transformer']))
        encoder = ResponseStatistics.load(folder/'response-store')
        xh = pd.concat([transformer.transform(held), encoder.transform(held[KEY_COLUMNS])], axis=1)
        prediction = pd.concat([held[META], cold_status(held, fit, inner)], axis=1)
        old = pd.read_parquet(parent/f'fold-{fold}-predictions.parquet')
        pd.testing.assert_frame_equal(held[META], old[META].reset_index(drop=True))
        for seed in config['seeds']:
            prediction[f'baseline-s{seed}'] = old[f'full-s{seed}'].to_numpy()
        for row in [r for r in runs if r['fold'] == fold]:
            booster = lgb.Booster(model_file=row['model'])
            prediction[f"{row['variant']}-s{row['seed']}"] = booster.predict(xh[FEATURES+statistic_columns(row['variant'])], num_threads=4)
        prediction['outer_fold'] = fold
        prediction.to_parquet(output/f'fold-{fold}-predictions.parquet', index=False)
        groups = selection_outcomes(prediction, config)
        groups['outer_fold'] = fold
        combined.append(groups)
        prediction_parts.append(prediction)
        folds[str(fold)] = summarize(groups, config)
        write_json(output/f'fold-{fold}-metrics.json', folds[str(fold)])
        delta = folds[str(fold)]['randomized_multi']['comparisons']['ad_context_stats-baseline-top1']
        print(f"Fold{fold} randomized primary delta={delta['estimate']:+.3f}/1000 CI={delta['ci95']}", flush=True)
    groups = pd.concat(combined).sort_values('impression_id').reset_index(drop=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    assert len(predictions) == len(pool) and not predictions.duplicated(['impression_id', 'product_id']).any()
    assert predictions.groupby('user_id').outer_fold.nunique().eq(1).all()
    pd.testing.assert_frame_equal(pool[META].sort_values(['impression_id', 'product_id']).reset_index(drop=True),
                                  predictions[META].sort_values(['impression_id', 'product_id']).reset_index(drop=True))
    groups.to_parquet(output/'oof_outcomes.parquet', index=False)
    pooled = summarize(groups, config)
    primary = pooled['randomized_multi']['comparisons']['ad_context_stats-baseline-top1']
    guard = pooled['all_multi']['comparisons']['ad_context_stats-baseline-top1']
    positive = all(folds[str(f)]['randomized_multi']['comparisons']['ad_context_stats-baseline-top1']['estimate'] > 0 for f in range(3))
    cold = pooled['strict_has_cold']['comparisons']['ad_context_stats-baseline-cold_top1']
    cold_random = pooled['randomized_strict_has_cold']['comparisons']['ad_context_stats-baseline-cold_top1']
    cold_positive = all(folds[str(f)]['strict_has_cold']['comparisons']['ad_context_stats-baseline-cold_top1']['estimate'] > 0 for f in range(3))
    result = {'config': config, 'runs': runs, 'split_audits': audits, 'folds': folds, 'pooled_oof': pooled,
        'probability_metrics': probability_summary(predictions, config), 'tracking_uri': tracking_uri,
        'protocol_sha256': sha256(output/'protocol.json'), 'frozen_models_sha256': sha256(models/'frozen_models.json'),
        'original_test_read': False, 'original_validation_read': False, 'oof_rows': len(predictions),
        'oof_users': int(predictions.user_id.nunique()), 'oof_multi_requests': len(groups),
        'primary_decision': {'all_randomized_fold_points_positive': bool(positive),
            'pooled_randomized_lower_bound_positive': bool(primary['ci95'][0] > 0),
            'pooled_all_display_lower_bound_positive': bool(guard['ci95'][0] > 0),
            'supported': bool(positive and primary['ci95'][0] > 0 and guard['ci95'][0] > 0)},
        'cold_secondary_decision': {'all_fold_cold_capture_points_positive': bool(cold_positive),
            'pooled_cold_capture_lower_bound_positive': bool(cold['ci95'][0] > 0),
            'pooled_randomized_cold_capture_lower_bound_positive': bool(cold_random['ci95'][0] > 0),
            'supported': bool(cold_positive and cold['ci95'][0] > 0 and cold_random['ci95'][0] > 0)}}
    for path, digest in {**hashes, **frozen, **{r['model']: r['model_sha256'] for r in runs}}.items():
        assert sha256(Path(path)) == digest, path
    write_json(output/'metrics.json', result)
    (output/'progress.json').unlink()
    print('Completed response statistics.', result['primary_decision'], result['cold_secondary_decision'], flush=True)
    return result
