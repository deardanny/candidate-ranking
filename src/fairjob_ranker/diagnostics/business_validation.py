"""Observed-click selection utility; deliberately not an online policy-value estimator."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_json, sha256, write_json


def topk_weights(scores, k):
    """Marginal inclusion probabilities under uniform permutations of score ties."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Nonempty finite one-dimensional scores required')
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('Positive integer k required')
    take = min(k, len(scores))
    cutoff = np.partition(scores, len(scores)-take)[len(scores)-take]
    weights = (scores > cutoff).astype(float)
    tied = scores == cutoff
    weights[tied] = (take-weights.sum())/tied.sum()
    return weights


def request_outcomes(frame, scores, ks=(1, 3)):
    """Keep every multi-candidate request, including zero-click requests."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError('One finite score per row required')
    if not frame.click.isin([0, 1]).all():
        raise ValueError('Binary labels required')
    if frame.duplicated(['impression_id', 'product_id']).any():
        raise ValueError('Ambiguous candidate IDs')
    if frame.groupby('impression_id')[['user_id', 'displayrandom']].nunique().gt(1).any().any():
        raise ValueError('Each request must have one user and display flag')
    rows = []
    for impression, ix in frame.groupby('impression_id', sort=True).indices.items():
        if len(ix) < 2:
            continue
        part = frame.iloc[ix]
        y, p = part.click.to_numpy(), scores[ix]
        ordered = np.lexsort((part.product_id.to_numpy(), -p))
        row = {'impression_id': int(impression), 'user_id': int(part.user_id.iloc[0]),
               'displayrandom': int(part.displayrandom.iloc[0]), 'candidates': len(ix),
               'observed_clicks': int(y.sum()),
               'observed_rank_contiguous': bool(np.array_equal(np.sort(part['rank']), np.arange(1, len(ix)+1)))}
        for k in ks:
            row[f'captured_at_{k}'] = float(topk_weights(p, k) @ y)
            row[f'id_ties_at_{k}'] = float(y[ordered[:k]].sum())
            row[f'slots_at_{k}'] = min(k, len(ix))
            row[f'oracle_at_{k}'] = min(k, int(y.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_request_means(frame, columns, repeats, seed):
    """Paired whole-user resampling; retain every request of each drawn user."""
    if frame.empty or repeats < 1:
        raise ValueError('Nonempty cohort and positive repeats required')
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Finite outcomes required')
    per_user = frame.groupby('user_id')[columns].sum()
    counts = frame.groupby('user_id').size().reindex(per_user.index).to_numpy()
    sums = per_user.to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, len(columns)))
    # Bounded memory even when a user has many requests or many replicates.
    for start in range(0, repeats, 32):
        end = min(start+32, repeats)
        drawn = rng.integers(0, len(per_user), (end-start, len(per_user)))
        samples[start:end] = sums[drawn].sum(axis=1)/counts[drawn].sum(axis=1)[:, None]
    return values.mean(axis=0), samples


def interval(estimate, samples, scale=1.):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return {'estimate': float(estimate*scale) if np.isfinite(estimate) else None,
            'ci95': [float(v*scale) for v in np.quantile(finite, [.025, .975])] if len(finite) else None,
            'valid_bootstrap_replicates': len(finite)}


def summarize_cohort(frame, config):
    if frame.empty:
        return {'requests': 0, 'users': 0, 'policies': {}, 'comparisons': {}}
    names, ks = config['policies'], config['ks']
    value_cols = [f'{p}_at_{k}' for p in names for k in ks]
    extra_cols = ['observed_clicks']+[f'slots_at_{k}' for k in ks]+[f'oracle_at_{k}' for k in ks]
    columns = value_cols+extra_cols
    point, draws = bootstrap_request_means(frame, columns, config['bootstrap_repeats'], config['bootstrap_seed'])
    index = {c: i for i, c in enumerate(columns)}
    result = {'requests': len(frame), 'users': int(frame.user_id.nunique()),
              'observed_rows': int(frame.candidates.sum()), 'observed_clicks': int(frame.observed_clicks.sum()),
              'requests_with_clicks': int(frame.observed_clicks.gt(0).sum()),
              'zero_click_requests': int(frame.observed_clicks.eq(0).sum()),
              'observed_rank_contiguous_requests': int(frame.observed_rank_contiguous.sum()),
              'policies': {}, 'comparisons': {}, 'oracle_label_upper_bound': {}}
    for k in ks:
        oi = index[f'oracle_at_{k}']
        result['oracle_label_upper_bound'][str(k)] = float(point[oi]*1000)
    for name in names:
        result['policies'][name] = {}
        for k in ks:
            j, click_j, slot_j = index[f'{name}_at_{k}'], index['observed_clicks'], index[f'slots_at_{k}']
            with np.errstate(divide='ignore', invalid='ignore'):
                capture = point[j]/point[click_j]
                capture_samples = draws[:, j]/draws[:, click_j]
            result['policies'][name][str(k)] = {
                'expected_captured_clicks_in_cohort': float(frame[f'{name}_at_{k}'].sum()),
                'observed_clicked_ads_per_1000_requests': interval(point[j], draws[:, j], 1000),
                'fraction_of_observed_clicks_captured': interval(capture, capture_samples),
                'observed_clicked_fraction_of_selected_slots': interval(point[j]/point[slot_j], draws[:, j]/draws[:, slot_j])}
    for candidate, reference in config['comparisons']:
        comparison = {}
        for k in ks:
            a, b = index[f'{candidate}_at_{k}'], index[f'{reference}_at_{k}']
            difference = point[a]-point[b]
            with np.errstate(divide='ignore', invalid='ignore'):
                lift = difference/point[b]
                lift_samples = (draws[:, a]-draws[:, b])/draws[:, b]
            deltas = frame[f'{candidate}_at_{k}']-frame[f'{reference}_at_{k}']
            comparison[str(k)] = {
                'extra_observed_clicked_ads_per_1000_requests': interval(difference, draws[:, a]-draws[:, b], 1000),
                'relative_proxy_change': interval(lift, lift_samples),
                'positive_delta_requests': int((deltas > 1e-12).sum()),
                'negative_delta_requests': int((deltas < -1e-12).sum()),
                'zero_delta_requests': int((deltas.abs() <= 1e-12).sum())}
        result['comparisons'][f'{candidate} - {reference}'] = comparison
    # Service tie-rule sensitivity is fixed to seed 42, never selected by outcome.
    result['representative_seed'] = config['representative_seed']
    result['representative_seed_tie_sensitivity'] = {
        p: {str(k): {
            'tie_expected_per_1000': float(frame[f'{p}_s42_at_{k}'].mean()*1000),
            'product_id_ties_per_1000': float(frame[f'{p}_s42_id_at_{k}'].mean()*1000)} for k in ks}
        for p in names}
    result['per_seed_top1_per_1000'] = {
        p: {str(s): float(frame[f'{p}_s{s}_at_1'].mean()*1000) for s in config['seeds']} for p in names}
    return result


def cohort_masks(groups):
    randomized = groups.displayrandom.eq(1)
    return {'randomized_multi': randomized,
            'all_multi': np.ones(len(groups), dtype=bool),
            'randomized_observed_rank_contiguous': randomized & groups.observed_rank_contiguous,
            'all_multi_2_to_5': groups.candidates.between(2, 5),
            'all_multi_6_to_10': groups.candidates.between(6, 10),
            'all_multi_11_plus': groups.candidates.ge(11)}


def run(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = read_json(config_path)
    if config['representative_seed'] != 42 or config['ks'] != [1, 3]:
        raise ValueError('This version fixes seed 42 sensitivity and k=1,3')
    ablation = read_json(Path(config['ablation_metrics']))
    residual = read_json(Path(config['residual_metadata']))
    parent_models = Path(config['residual_metadata']).parent
    artifact_paths = {str(Path(r['model'])): r['model_sha256'] for r in ablation['runs']}
    artifact_paths.update({str(parent_models/name): digest for name, digest in residual['artifact_sha256'].items()})
    for name, digest in artifact_paths.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Parent artifact changed: '+name)
    validation = Path(config['validation'])
    if validation.name != 'validation.parquet':
        raise ValueError('Only development validation is permitted')
    manifest = read_json(validation.parent/'manifest.json')
    if sha256(validation) != manifest['files']['validation.parquet']['sha256']:
        raise ValueError('Validation changed')
    paths = [config_path, Path(__file__), Path(config['ablation_predictions']), Path(config['residual_predictions']),
             Path(config['ablation_metrics']), Path(config['residual_metadata']), validation]
    hashes = {str(path): sha256(path) for path in paths}
    output.mkdir(parents=True)
    write_json(output/'protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
               'input_sha256': hashes, 'model_sha256': artifact_paths, 'original_test_read': False,
               'note': 'New endpoints on previously used development data; not a fresh holdout or causal policy evaluation.'})

    meta = ['user_id', 'impression_id', 'product_id', 'click', 'displayrandom', 'rank']
    frame = pd.read_parquet(validation, columns=meta).reset_index(drop=True)
    pa = pd.read_parquet(config['ablation_predictions'])
    pr = pd.read_parquet(config['residual_predictions'])
    for parent in [pa, pr]:
        pd.testing.assert_frame_equal(frame, parent[meta].reset_index(drop=True))
    groups = None
    checks = []
    # Full graft and unnormalized ad correction are rank-equivalent controls.
    # Re-evaluate these to verify their top-k utility, not to count as extra wins.
    for seed in config['seeds']:
        for name in ['context', 'full']:
            column = 'context_only' if name == 'context' else name
            np.testing.assert_array_equal(pa[f'{column}-s{seed}'], pr[f'{name}-s{seed}'])
        score_map = {'uniform': pa[f'context_only-s{seed}'], 'ad_only': pa[f'ad_only-s{seed}'],
                     'full': pa[f'full-s{seed}'], 'ad_residual': pr[f'ad_mean_preserved-s{seed}'],
                     'graft_control': pr[f'full_rank_graft-s{seed}'], 'offset_control': pr[f'ad_offset-s{seed}']}
        seed_outcomes = {}
        for policy, scores in score_map.items():
            outcome = request_outcomes(frame, scores.to_numpy(), config['ks'])
            seed_outcomes[policy] = outcome
            if groups is None:
                groups = outcome.drop(columns=[f'{prefix}_at_{k}' for prefix in ['captured', 'id_ties'] for k in config['ks']])
            np.testing.assert_array_equal(groups.impression_id, outcome.impression_id)
            if policy in config['policies']:
                for k in config['ks']:
                    groups[f'{policy}_s{seed}_at_{k}'] = outcome[f'captured_at_{k}']
                    groups[f'{policy}_s{seed}_id_at_{k}'] = outcome[f'id_ties_at_{k}']
            print(f'Computed all-request top-k outcomes: {policy}, seed {seed}', flush=True)
        for k in config['ks']:
            np.testing.assert_allclose(seed_outcomes['uniform'][f'captured_at_{k}'],
                                       groups.observed_clicks*groups[f'slots_at_{k}']/groups.candidates, atol=1e-12, rtol=0)
            for a, b in [('full', 'graft_control'), ('ad_residual', 'offset_control')]:
                np.testing.assert_allclose(seed_outcomes[a][f'captured_at_{k}'], seed_outcomes[b][f'captured_at_{k}'], atol=1e-12, rtol=0)
        checks.append(f'seed_{seed}_uniform_and_rank_equivalent_controls_verified')
    for policy in config['policies']:
        for k in config['ks']:
            groups[f'{policy}_at_{k}'] = groups[[f'{policy}_s{s}_at_{k}' for s in config['seeds']]].mean(axis=1)
    groups.to_parquet(output/'request_outcomes.parquet', index=False)
    masks = cohort_masks(groups)
    if set(masks) != set(config['cohorts']):
        raise ValueError('Cohort contract changed')
    reports = {}
    for name in config['cohorts']:
        reports[name] = summarize_cohort(groups.loc[masks[name]], config)
        print(f'Bootstrap completed: {name}, requests={reports[name]["requests"]}', flush=True)
    primary = config['primary']
    primary_result = reports[primary['cohort']]['comparisons'][primary['candidate']+' - '+primary['reference']]['1']['extra_observed_clicked_ads_per_1000_requests']
    result = {'config': config, 'protocol_sha256': sha256(output/'protocol.json'),
              'cohorts': reports, 'primary_result': primary_result,
              'primary_proxy_success': bool(primary_result['ci95'][0] > 0),
              'original_test_read': False, 'retrained': False, 'online_lift_estimated': False,
              'original_validation_requests': int(frame.impression_id.nunique()),
              'excluded_single_candidate_requests': int(frame.impression_id.nunique()-len(groups)),
              'checks_passed': checks}
    for path, digest in hashes.items():
        assert sha256(Path(path)) == digest
    for path, digest in artifact_paths.items():
        assert sha256(Path(path)) == digest
    result['checks_passed'].append('parent_predictions_labels_models_and_sources_unchanged')
    write_json(output/'metrics.json', result)
    return result
