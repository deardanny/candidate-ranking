"""Audit complete-candidate normalization using frozen development models.

The independent-chunk path is deliberately wrong. It is a contract stress
test, not a reproduction of a bug in the existing production API.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fairjob_ranker.diagnostics.context_residual import combine_scores, group_mean
from fairjob_ranker.evaluation import click_metrics
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, CONTEXT, FEATURES


def distribution(values):
    x = np.asarray(values, dtype=float)
    return {'count': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'max': float(x.max())}


def relation_changes(indices, before, after, tolerance):
    """Label-blind pair order and tied-top-set changes, on multi-ad groups.

    A tolerance is used only for this contract check, not LogLoss or AUC.
    Separate strict reversals from loss of discrimination through new ties.
    """
    total = dict(impressions=0, pairs=0, reversed_pairs=0, collapsed_pairs=0,
                 separated_pairs=0, impressions_with_pair_changes=0,
                 impressions_with_top_set_changes=0)
    for index in indices:
        if len(index) < 2:
            continue
        a, b = before[index], after[index]
        left, right = np.triu_indices(len(index), 1)
        da, db = a[left]-a[right], b[left]-b[right]
        sa = np.where(np.abs(da) <= tolerance, 0, np.sign(da))
        sb = np.where(np.abs(db) <= tolerance, 0, np.sign(db))
        total['impressions'] += 1
        total['pairs'] += len(left)
        total['reversed_pairs'] += int(np.sum(sa*sb < 0))
        total['collapsed_pairs'] += int(np.sum((sa != 0) & (sb == 0)))
        total['separated_pairs'] += int(np.sum((sa == 0) & (sb != 0)))
        total['impressions_with_pair_changes'] += int(np.any(sa != sb))
        top_a, top_b = a >= a.max()-tolerance, b >= b.max()-tolerance
        total['impressions_with_top_set_changes'] += int(not np.array_equal(top_a, top_b))
    return total


def run(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = read_json(config_path)
    models = Path(config['parent_models'])
    metadata = read_json(models/'metadata.json')
    data_path = Path(config['data'])
    manifest = read_json(data_path.parent/'manifest.json')
    if data_path.name != 'validation.parquet':
        raise ValueError('This audit accepts development validation only')
    if sha256(data_path) != manifest['files'][data_path.name]['sha256']:
        raise ValueError('Validation checksum mismatch')
    for name, digest in metadata['artifact_sha256'].items():
        if sha256(models/name) != digest:
            raise ValueError('Frozen model artifact changed: '+name)
    output.mkdir(parents=True)
    prereg = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'config_sha256': sha256(config_path), 'source_sha256': sha256(Path(__file__)),
              'parent_metadata_sha256': sha256(models/'metadata.json'),
              'parent_predictions_sha256': sha256(Path(config['parent_predictions'])),
              'validation_sha256': sha256(data_path),
              'previously_used_development_data': True, 'original_test_read': False,
              'note': 'Rules fixed before this audit; this is not a new independent holdout.'}
    write_json(output/'protocol.json', prereg)

    frame = pd.read_parquet(data_path)
    saved = pd.read_parquet(config['parent_predictions'])
    keys = ['user_id', 'impression_id', 'product_id']
    pd.testing.assert_frame_equal(frame[keys].reset_index(drop=True), saved[keys].reset_index(drop=True))
    if frame.duplicated(['impression_id', 'product_id']).any():
        raise ValueError('Candidate IDs must be unique within a request')
    x = FeatureTransformer.load(models/'features.json').transform(frame)
    ids = frame.impression_id.to_numpy()
    _, codes = np.unique(ids, return_inverse=True)
    counts = np.bincount(codes)
    indices = list(frame.groupby('impression_id', sort=True).indices.values())
    # Scheduling/removal uses IDs only, never click or original display rank.
    ordinal = frame.groupby('impression_id').product_id.rank(method='first').to_numpy(dtype=int)-1
    chunk = ordinal % 2
    keep = (counts[codes] == 1) | (ordinal < counts[codes]-1)
    retained_indices = list(frame.loc[keep].reset_index(drop=True).groupby('impression_id').indices.values())
    changed_set_rows = keep & (counts[codes] > 1)
    example_mask = ids == config['example_impression']
    results, example_rows = [], []
    tolerance = config['comparison_tolerance']
    for seed in config['seeds']:
        context = lgb.Booster(model_file=str(models/f'context-s{seed}.txt'))
        base = context.predict(x[CONTEXT], num_threads=4)
        np.testing.assert_array_equal(base, saved[f'context-s{seed}'].to_numpy())
        for policy in config['policies']:
            donor_name = 'full' if policy == 'full_rank_graft' else 'ad-correction'
            columns = FEATURES if policy == 'full_rank_graft' else AD_FEATURES
            donor = lgb.Booster(model_file=str(models/f'{donor_name}-s{seed}.txt'))
            raw = donor.predict(x[columns], raw_score=True, num_threads=4)
            alpha = metadata['selection'][policy]['alpha']
            def normalize(p, r, group_ids):
                return combine_scores(p, r, group_ids, alpha, True, config['normalization_iterations'])
            exact = normalize(base, raw, ids)
            np.testing.assert_allclose(exact, saved[f'{policy}-s{seed}'], rtol=0, atol=tolerance)
            centered = expit(logit(base)+alpha*(raw-group_mean(raw, codes, counts)[codes]))
            bad_chunks, assembled_raw = np.empty(len(frame)), np.empty(len(frame))
            for part in (0, 1):
                mask = chunk == part
                bad_chunks[mask] = normalize(base[mask], raw[mask], ids[mask])
                assembled_raw[mask] = raw[mask]
            correct_chunks = normalize(base, assembled_raw, ids)
            np.testing.assert_array_equal(exact, correct_chunks)
            reduced = normalize(base[keep], raw[keep], ids[keep])
            center_order = relation_changes(indices, exact, centered, tolerance)
            retained_order = relation_changes(retained_indices, exact[keep], reduced, tolerance)
            assert center_order['reversed_pairs'] == 0
            assert retained_order['reversed_pairs'] == 0
            target = group_mean(base, codes, counts)
            exact_error = np.abs(group_mean(exact, codes, counts)-target)
            center_error = np.abs(group_mean(centered, codes, counts)-target)
            bad_mean_error = np.abs(group_mean(bad_chunks, codes, counts)-target)
            assert exact_error.max() < tolerance and bad_mean_error.max() < tolerance
            row = {'seed': seed, 'policy': policy, 'frozen_alpha': alpha,
                   'exact': {'click': click_metrics(frame.click.to_numpy(), exact),
                             'absolute_mean_drift': distribution(exact_error)},
                   'center_only': {'click': click_metrics(frame.click.to_numpy(), centered),
                                   'absolute_mean_drift': distribution(center_error),
                                   'relative_mean_drift': distribution(center_error/target),
                                   'order_check': center_order},
                   'independent_chunk_normalization': {
                       'absolute_mean_drift': distribution(bad_mean_error),
                       'absolute_row_score_change': distribution(np.abs(bad_chunks-exact)),
                       'order_check': relation_changes(indices, exact, bad_chunks, tolerance)},
                   'assemble_then_normalize': {'max_absolute_error': float(np.max(np.abs(correct_chunks-exact)))},
                   'remove_largest_id': {'removed_rows': int((~keep).sum()),
                       'absolute_retained_score_change': distribution(np.abs(reduced-exact[keep])[changed_set_rows[keep]]),
                       'retained_order_check': retained_order}}
            results.append(row)
            ex = frame.loc[example_mask, ['product_id', 'click']].copy()
            ex['seed'], ex['policy'] = seed, policy
            ex['chunk'] = chunk[example_mask]
            ex['context'], ex['raw'] = base[example_mask], raw[example_mask]
            ex['exact'], ex['center_only'] = exact[example_mask], centered[example_mask]
            ex['independent_chunks'] = bad_chunks[example_mask]
            ex['retained'] = keep[example_mask]
            ex['after_removal'] = np.nan
            retained_example = example_mask[keep]
            ex.loc[ex.retained, 'after_removal'] = reduced[retained_example]
            example_rows.extend(ex.astype(object).where(pd.notna(ex), None).to_dict('records'))
            print(f"{policy} seed={seed}: center drift max={center_error.max():.8g}; "
                  f"chunk pair-change requests={row['independent_chunk_normalization']['order_check']['impressions_with_pair_changes']}", flush=True)

    # Verify the frozen inputs still match after the complete read-only audit.
    assert sha256(models/'metadata.json') == prereg['parent_metadata_sha256']
    for name, digest in metadata['artifact_sha256'].items():
        assert sha256(models/name) == digest
    write_json(output/'metrics.json', {'protocol_sha256': sha256(output/'protocol.json'),
               'rows': len(frame), 'impressions': len(counts),
               'multi_candidate_impressions': int((counts > 1).sum()), 'results': results,
               'checks_passed': ['parent_predictions_reproduced', 'parent_artifacts_unchanged',
                                'assembled_raw_scores_match_exactly', 'mean_constraint',
                                'no_strict_reversal_after_centering_or_removal'],
               'original_test_read': False})
    write_json(output/'example.json', {'impression_id': config['example_impression'], 'rows': example_rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/candidate_contract.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/candidate-contract'))
    args = parser.parse_args()
    run(args.config, args.output)
