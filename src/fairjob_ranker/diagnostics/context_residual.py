"""Frozen context probability plus cross-fitted ad corrections."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from ..evaluation import click_metrics
from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import AD_FEATURES, CATEGORICAL, CONTEXT, FEATURES
from .context_ablation import evaluate_diagnostics, paired_comparison


POLICIES = ['ad_offset', 'ad_mean_preserved', 'full_rank_graft']


def user_folds(frame, folds, salt):
    if folds < 2:
        raise ValueError('At least two folds required')
    mapping = {int(u): int.from_bytes(hashlib.sha256(f'{salt}:{int(u)}'.encode()).digest()[:8], 'big') % folds
               for u in frame.user_id.unique()}
    result = frame.user_id.map(mapping).to_numpy(dtype=int)
    if set(result) != set(range(folds)):
        raise ValueError('Every fold must contain users')
    return result


def group_mean(values, codes, counts):
    return np.bincount(codes, weights=values, minlength=len(counts)) / counts


def combine_scores(context_probability, correction_raw, impression_ids, alpha, preserve_mean, iterations=60):
    """Mean-preserving mode solves an intercept for the complete candidate set.

    It preserves probability mean, not just the mean raw score. Ranking is
    monotone in the donor raw score for alpha > 0; alpha=0 returns exact ties.
    """
    p = np.asarray(context_probability, dtype=float)
    r = np.asarray(correction_raw, dtype=float)
    ids = np.asarray(impression_ids)
    if p.ndim != 1 or r.shape != p.shape or ids.shape != p.shape or not len(p):
        raise ValueError('Aligned nonempty one-dimensional arrays required')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError('Finite raw scores and strictly interior probabilities required')
    if not np.isfinite(alpha) or alpha < 0 or iterations < 1:
        raise ValueError('Nonnegative finite alpha and positive iterations required')
    _, codes = np.unique(ids, return_inverse=True)
    counts = np.bincount(codes)
    target = group_mean(p, codes, counts)
    if np.max(np.abs(p-target[codes])) > 1e-12:
        raise ValueError('Context probability must be constant within each impression')
    if alpha == 0:
        return p.copy()
    if not preserve_mean:
        return expit(logit(p) + alpha*r)
    centered = alpha*(r-group_mean(r, codes, counts)[codes])
    lower = np.full(len(counts), np.inf)
    upper = np.full(len(counts), -np.inf)
    np.minimum.at(lower, codes, centered)
    np.maximum.at(upper, codes, centered)
    # These bounds straddle the target: every row lies below/above it.
    lo, hi = logit(target)-upper, logit(target)-lower
    for _ in range(iterations):
        middle = (lo+hi)/2
        means = group_mean(expit(middle[codes]+centered), codes, counts)
        smaller = means < target
        lo = np.where(smaller, middle, lo)
        hi = np.where(smaller, hi, middle)
    result = expit(((lo+hi)/2)[codes]+centered)
    if np.max(np.abs(group_mean(result, codes, counts)-target)) > 1e-10:
        raise ValueError('Mean normalization did not converge')
    return result


def drift_summary(frame, context, scores):
    data = pd.DataFrame({'impression': frame.impression_id.to_numpy(), 'base': context, 'score': scores})
    means = data.groupby('impression')[['base', 'score']].mean()
    difference = means.score-means.base
    return {'max_absolute_impression_mean_change': float(difference.abs().max()),
            'mean_absolute_impression_mean_change': float(difference.abs().mean()),
            'mean_signed_impression_mean_change': float(difference.mean())}


def fit_correction(xtrain, ytrain, oof_raw, xinner, yinner, base_raw_inner, config):
    history = {}
    booster = lgb.train(config['params'],
                        lgb.Dataset(xtrain, label=ytrain, init_score=oof_raw, categorical_feature=AD_FEATURES),
                        num_boost_round=config['num_boost_round'],
                        valid_sets=[lgb.Dataset(xinner, label=yinner, init_score=base_raw_inner, categorical_feature=AD_FEATURES)],
                        valid_names=['inner_validation'],
                        callbacks=[lgb.early_stopping(config['early_stopping_rounds'], verbose=False), lgb.record_evaluation(history)])
    # init_score is a Dataset offset; the saved correction model does not
    # supply the context model's score at prediction time. Add it explicitly.
    reconstructed = expit(base_raw_inner+booster.predict(xinner, raw_score=True, num_threads=config['params']['num_threads']))
    measured = click_metrics(np.asarray(yinner), reconstructed)['logloss']
    np.testing.assert_allclose(measured, booster.best_score['inner_validation']['binary_logloss'], rtol=0, atol=1e-9)
    return booster, history, measured


def cross_fit_context(train, assignments, seed, parent, config, destination):
    """Each user's offset is predicted by a model that never saw that user."""
    scores = np.full(len(train), np.nan)
    coverage = np.zeros(len(train), dtype=int)
    audits = []
    for fold in range(config['oof_folds']):
        hold = assignments == fold
        fitted, held = train.loc[~hold], train.loc[hold]
        overlap = set(fitted.user_id) & set(held.user_id)
        if overlap:
            raise ValueError('OOF user leakage')
        transformer = FeatureTransformer.fit(fitted)
        xt = transformer.transform(fitted)[CONTEXT]
        xh = transformer.transform(held)[CONTEXT]
        model = lgb.train(parent['params'], lgb.Dataset(xt, label=fitted.click, categorical_feature=['cat0', 'cat1']),
                          num_boost_round=parent['best_iteration'])
        raw = model.predict(xh, raw_score=True, num_threads=parent['params']['num_threads'])
        scores[hold] = raw
        coverage[hold] += 1
        folder = destination/f'seed-{seed}-fold-{fold}'
        folder.mkdir(parents=True)
        model.save_model(str(folder/'model.txt'))
        transformer.save(folder/'features.json')
        audit = {'seed': seed, 'fold': fold, 'fitted_rows': len(fitted), 'held_rows': len(held),
                 'fitted_users': int(fitted.user_id.nunique()), 'held_users': int(held.user_id.nunique()),
                 'overlap_users': len(overlap), 'fixed_trees': parent['best_iteration'],
                 'held_labels_used_for_fit_or_stopping': False, 'model_sha256': sha256(folder/'model.txt'),
                 'transformer_sha256': sha256(folder/'features.json')}
        write_json(folder/'audit.json', audit)
        audits.append(audit)
        print(f'OOF context seed={seed}, fold={fold}: {len(fitted):,} train / {len(held):,} held rows', flush=True)
    if not (coverage == 1).all() or not np.isfinite(scores).all():
        raise ValueError('Incomplete OOF scores')
    return scores, audits


def predict_bundle(directory, frame, seed, policy):
    """Reload the complete diagnostic pipeline, including frozen offsets."""
    directory = Path(directory)
    metadata = read_json(directory/'metadata.json')
    for path, digest in metadata['artifact_sha256'].items():
        if sha256(directory/path) != digest:
            raise ValueError('Bundle artifact changed: '+path)
    if policy not in POLICIES:
        raise ValueError('Unknown residual policy')
    transformed = FeatureTransformer.load(directory/'features.json').transform(frame)
    context = lgb.Booster(model_file=str(directory/f'context-s{seed}.txt'))
    probability = context.predict(transformed[CONTEXT], num_threads=4)
    donor_path = f'full-s{seed}.txt' if policy == 'full_rank_graft' else f'ad-correction-s{seed}.txt'
    donor_columns = FEATURES if policy == 'full_rank_graft' else AD_FEATURES
    donor = lgb.Booster(model_file=str(directory/donor_path))
    raw = donor.predict(transformed[donor_columns], raw_score=True, num_threads=4)
    return combine_scores(probability, raw, frame.impression_id.to_numpy(), metadata['selection'][policy]['alpha'],
                          preserve_mean=policy != 'ad_offset', iterations=metadata['normalization_iterations'])


def run_experiment(data_dir, config_path, model_root, output, tracking_uri=None):
    import mlflow
    from ..v2.training import development_split

    for directory in [model_root, output]:
        if directory.exists():
            raise FileExistsError(directory)
    config = read_json(config_path)
    if config['policies'] != POLICIES or 0. not in config['alpha_grid']:
        raise ValueError('Expected the registered policies and exact alpha=0 fallback')
    parents = read_json(Path(config['parent_experiment']))
    parent_prereg = read_json(Path(config['parent_experiment']).parent/'preregistered.json')
    parent_models = Path(config['parent_models'])
    manifest = read_json(data_dir/'manifest.json')
    if sha256(data_dir/'manifest.json') != parent_prereg['dataset_manifest_sha256']:
        raise ValueError('Parent experiment uses different data')
    for key in ['development_split_salt', 'inner_train_percent', 'seeds']:
        if parents['config'][key] != config[key]:
            raise ValueError('Parent split/seed contract changed: '+key)
    if sha256(data_dir/'train.parquet') != manifest['files']['train.parquet']['sha256']:
        raise ValueError('Training file changed')
    original = pd.read_parquet(data_dir/'train.parquet')
    train, inner = development_split(original, config['development_split_salt'], config['inner_train_percent'])
    transformer = FeatureTransformer.load(parent_models/'features.json')
    xt, xi = transformer.transform(train), transformer.transform(inner)
    assignments = user_folds(train, config['oof_folds'], config['oof_salt'])
    parent_rows = {r['name']: r for r in parents['runs']}
    lgb_config = read_json(Path(config['lightgbm_config']))
    model_root.mkdir(parents=True)
    output.mkdir(parents=True)
    shutil.copyfile(parent_models/'features.json', model_root/'features.json')
    for seed in config['seeds']:
        for name, prefix in [('context_only', 'context'), ('full', 'full')]:
            row = parent_rows[f'{name}-s{seed}']
            source = Path(row['model'])
            if sha256(source) != row['model_sha256']:
                raise ValueError('Parent model changed')
            if sha256(parent_models/'features.json') != row['transformer_sha256']:
                raise ValueError('Parent transformer changed')
            shutil.copyfile(source, model_root/f'{prefix}-s{seed}.txt')
    prereg = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'config_sha256': sha256(config_path), 'source_sha256': sha256(Path(__file__)),
              'lgb_config': lgb_config, 'parent_metrics_sha256': sha256(Path(config['parent_experiment'])),
              'train_sha256': sha256(data_dir/'train.parquet'), 'test_read': False,
              'inner_train_rows': len(train), 'inner_validation_rows': len(inner),
              'inner_train_users': int(train.user_id.nunique()), 'inner_validation_users': int(inner.user_id.nunique()),
              'frozen_parent_hashes': {p.name: sha256(p) for p in sorted(model_root.glob('*.txt'))},
              'transformer_sha256': sha256(model_root/'features.json')}
    write_json(output/'preregistered.json', prereg)
    pd.DataFrame({'user_id': train.user_id, 'fold': assignments}).drop_duplicates().to_parquet(output/'oof_users.parquet', index=False)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['experiment_name'])
    rows, oof_audits, grid = [], [], []
    predictions, groups = {'inner': {}, 'outer': {}}, {'inner': {}, 'outer': {}}
    donor_raw = {}
    for seed in config['seeds']:
        began = perf_counter()
        context = lgb.Booster(model_file=str(model_root/f'context-s{seed}.txt'))
        full = lgb.Booster(model_file=str(model_root/f'full-s{seed}.txt'))
        base_raw = context.predict(xi[CONTEXT], raw_score=True, num_threads=4)
        base_p = context.predict(xi[CONTEXT], num_threads=4)
        full_p = full.predict(xi[FEATURES], num_threads=4)
        for name, score in [('context', base_p), ('full', full_p)]:
            predictions['inner'][f'{name}-s{seed}'] = score
        oof, audit = cross_fit_context(train, assignments, seed, parent_rows[f'context_only-s{seed}'], config, model_root/'oof')
        oof_audits.extend(audit)
        pd.DataFrame({'user_id': train.user_id, 'impression_id': train.impression_id, 'product_id': train.product_id,
                      'fold': assignments, 'context_raw_oof': oof}).to_parquet(output/f'oof-s{seed}.parquet', index=False)
        fit_config = {**lgb_config, 'params': {**lgb_config['params'], **{k: seed for k in ['seed', 'feature_fraction_seed', 'bagging_seed', 'data_random_seed']}}}
        correction, history, offset_loss = fit_correction(xt[AD_FEATURES], train.click, oof,
                                                         xi[AD_FEATURES], inner.click, base_raw, fit_config)
        correction.save_model(str(model_root/f'ad-correction-s{seed}.txt'), num_iteration=correction.best_iteration)
        write_json(model_root/f'history-s{seed}.json', history)
        saved = lgb.Booster(model_file=str(model_root/f'ad-correction-s{seed}.txt'))
        raw = saved.predict(xi[AD_FEATURES], raw_score=True, num_threads=4)
        np.testing.assert_array_equal(raw, correction.predict(xi[AD_FEATURES], raw_score=True, num_threads=4))
        donor_raw[seed] = {'ad': raw, 'full': full.predict(xi[FEATURES], raw_score=True, num_threads=4)}
        row = {'seed': seed, 'correction_trees': correction.best_iteration, 'training_seconds': perf_counter()-began,
               'reconstructed_inner_logloss_alpha1': offset_loss,
               'model_sha256': sha256(model_root/f'ad-correction-s{seed}.txt'), 'reload_raw_max_error': 0.}
        if tracking_uri:
            with mlflow.start_run(run_name=f'ad-correction-s{seed}') as run:
                row['mlflow_run_id'] = run.info.run_id
                mlflow.log_params({'seed': seed, 'oof_folds': config['oof_folds'], 'correction_features': 11,
                                   'config_sha256': prereg['config_sha256'], 'fixed_context_trees': parent_rows[f'context_only-s{seed}']['best_iteration']})
                mlflow.log_metrics({'correction_trees': correction.best_iteration, 'inner_offset_logloss': offset_loss})
                mlflow.log_artifact(str(model_root/f'ad-correction-s{seed}.txt'))
                mlflow.log_artifact(str(output/'preregistered.json'))
        rows.append(row)
        for policy in POLICIES:
            donor = donor_raw[seed]['full' if policy == 'full_rank_graft' else 'ad']
            for alpha in config['alpha_grid']:
                score = combine_scores(base_p, donor, inner.impression_id.to_numpy(), alpha, policy != 'ad_offset', config['normalization_iterations'])
                # Selection uses LogLoss alone. Ranking is evaluated after the
                # single common strength for all seeds has been frozen.
                grid.append({'policy': policy, 'seed': seed, 'alpha': alpha,
                             'inner_logloss': click_metrics(inner.click.to_numpy(), score)['logloss']})
        write_json(output/'progress.json', {'trained_corrections': len(rows), 'outer_read': False})
        print(f'Correction seed={seed}, trees={correction.best_iteration}, inner offset LogLoss={offset_loss:.6f}', flush=True)
    selection = {}
    for policy in POLICIES:
        choices = [{'alpha': alpha, 'mean_inner_logloss': float(np.mean([r['inner_logloss'] for r in grid if r['policy'] == policy and r['alpha'] == alpha]))}
                   for alpha in config['alpha_grid']]
        selection[policy] = dict(min(choices, key=lambda r: (r['mean_inner_logloss'], r['alpha'])))
        selection[policy]['candidates'] = choices
    write_json(output/'inner_grid.json', grid)
    write_json(output/'selection.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'outer_read': False, 'selection': selection})
    metadata = {'selection': selection, 'normalization_iterations': config['normalization_iterations'],
                'artifact_sha256': {p.name: sha256(p) for p in sorted(model_root.glob('*')) if p.is_file()},
                'feature_contract': {'context': CONTEXT, 'correction': AD_FEATURES, 'full_rank_donor': FEATURES}}
    write_json(model_root/'metadata.json', metadata)
    print('Frozen strengths:', {k: v['alpha'] for k, v in selection.items()}, flush=True)
    # No original test access. Outer is opened only after every decision above.
    if sha256(data_dir/'validation.parquet') != manifest['files']['validation.parquet']['sha256']:
        raise ValueError('Outer data changed')
    outer = pd.read_parquet(data_dir/'validation.parquet')
    for key in ['user_id', 'impression_id']:
        if set(original[key]) & set(outer[key]):
            raise ValueError('Outer leakage')
    reports, mean_drift = {}, {}
    for split, frame in [('inner', inner), ('outer', outer)]:
        x = transformer.transform(frame)
        reports[split], mean_drift[split] = {}, {}
        for seed in config['seeds']:
            context = lgb.Booster(model_file=str(model_root/f'context-s{seed}.txt'))
            full = lgb.Booster(model_file=str(model_root/f'full-s{seed}.txt'))
            context_p = context.predict(x[CONTEXT], num_threads=4)
            predictions[split][f'context-s{seed}'] = context_p
            predictions[split][f'full-s{seed}'] = full.predict(x[FEATURES], num_threads=4)
            for policy in POLICIES:
                p = predict_bundle(model_root, frame, seed, policy)
                predictions[split][f'{policy}-s{seed}'] = p
                mean_drift[split][f'{policy}-s{seed}'] = drift_summary(frame, context_p, p)
            for name in ['context', 'full']+POLICIES:
                key = f'{name}-s{seed}'
                reports[split][key], groups[split][key] = evaluate_diagnostics(frame, predictions[split][key])
        frame[['user_id', 'impression_id', 'product_id', 'click', 'displayrandom', 'rank']].assign(**predictions[split]).to_parquet(output/f'{split}_predictions.parquet', index=False)
        pd.concat([g.assign(model=name) for name, g in groups[split].items()]).to_parquet(output/f'{split}_ranking.parquet', index=False)
    comparisons = {}
    for split, frame in [('inner', inner), ('outer', outer)]:
        comparisons[split] = {}
        for policy in POLICIES:
            for reference in ['context', 'full']:
                names = [f'{policy}-s{s}' for s in config['seeds']]
                refs = [f'{reference}-s{s}' for s in config['seeds']]
                comparisons[split][policy+' - '+reference] = paired_comparison(
                    frame, [groups[split][n] for n in names], [groups[split][n] for n in refs],
                    [predictions[split][n] for n in names], [predictions[split][n] for n in refs],
                    config['bootstrap_repeats'], config['bootstrap_seed'])
    if tracking_uri:
        for row in rows:
            seed = row['seed']
            with mlflow.start_run(run_id=row['mlflow_run_id']):
                for policy in POLICIES:
                    m = reports['outer'][f'{policy}-s{seed}']['all']
                    mlflow.log_metrics({f'{policy}_alpha': selection[policy]['alpha'],
                                        f'{policy}_outer_logloss': m['click']['logloss'],
                                        f'{policy}_outer_auc': m['click']['roc_auc'],
                                        f'{policy}_outer_ndcg': m['ranking']['ndcg_tie_averaged']})
                mlflow.log_dict(selection, 'selection.json')
                mlflow.log_dict(reports['outer'], 'outer_metrics.json')
    example = outer.loc[outer.impression_id == config['example_impression'], ['product_id', 'click', 'rank']].copy()
    mask = outer.impression_id.eq(config['example_impression']).to_numpy()
    for name in ['context', 'full']+POLICIES:
        example[name] = predictions['outer'][f'{name}-s{config["representative_seed"]}'][mask]
    write_json(output/'example.json', {'impression_id': config['example_impression'],
                                      'seed': config['representative_seed'], 'rows': example.to_dict('records'),
                                      'scope': 'Previously discussed impression, fixed before this experiment; not representative evidence'})
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'selection': selection,
              'test_read': False, 'existing_models_modified': False, 'tracking_uri': tracking_uri,
              'preregistered_sha256': sha256(output/'preregistered.json'), 'runs': rows,
              'oof_audits': oof_audits, 'metrics': reports, 'mean_drift': mean_drift,
              'comparisons': comparisons}
    write_json(output/'metrics.json', result)
    (output/'progress.json').unlink()
    print('Completed correction experiment:', output, flush=True)
    return result
