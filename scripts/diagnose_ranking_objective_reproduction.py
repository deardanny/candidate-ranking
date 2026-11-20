"""Post-result diagnostics; fixed models only, no new tuning or success tests."""
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fairjob_ranker.diagnostics.business_validation import bootstrap_request_means, interval
from fairjob_ranker.diagnostics.candidate_count_reproduction import split_frames
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, FEATURES


def score_statistics(frame, scores):
    codes, ids = pd.factorize(frame.impression_id, sort=False)
    n = len(ids)
    size = np.bincount(codes)
    clicks = np.bincount(codes, weights=frame.click)
    maximum, minimum = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(maximum, codes, scores)
    np.minimum.at(minimum, codes, scores)
    ties = scores == maximum[codes]
    tied_n = np.bincount(codes, weights=ties)
    captured = np.bincount(codes, weights=ties*frame.click.to_numpy()) / tied_n
    multi = size >= 2
    informative = multi & (clicks > 0) & (clicks < size)
    info_ids = ids[informative]
    info_frame = frame.loc[frame.impression_id.isin(info_ids)]
    return {'multi_requests': int(multi.sum()), 'informative_requests': int(informative.sum()),
            'informative_users': int(info_frame.user_id.nunique()),
            'informative_randomized_requests': int(info_frame.loc[info_frame.displayrandom.eq(1)].impression_id.nunique()),
            'multi_top1_per1000': float(captured[multi].mean()*1000),
            'informative_top1_fraction': float(captured[informative].mean()),
            'multi_score_variation_fraction': float((maximum > minimum)[multi].mean()),
            'informative_score_variation_fraction': float((maximum > minimum)[informative].mean()),
            'informative_top_score_tied_fraction': float((tied_n > 1)[informative].mean())}


def main():
    root = Path('reports/ranking-objective-reproduction')
    metrics = read_json(root/'metrics.json')
    config = metrics['config']
    parent = Path(config['parent_report'])
    previous = read_json(parent/'metrics.json')
    write_json(root/'diagnostic_protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'outcomes_already_inspected': True, 'confirmatory': False, 'retraining': False,
        'metrics_sha256': sha256(root/'metrics.json'), 'source_sha256': sha256(Path(__file__)),
        'purpose': 'Measure effective within-request label support, fit/inner/held ranking and score ties; describe random/nonrandom heterogeneity and user influence. No hyperparameter choice or model replacement.'})
    pool = pd.read_parquet(Path(config['data_dir'])/'train.parquet')
    assignments = pd.read_parquet(parent/'user_assignments.parquet')
    results = []
    for fold in range(3):
        fit, inner, held = split_frames(pool, assignments, fold)
        transformer = FeatureTransformer.load(Path(metrics['split_audits'][fold]['transformer']))
        runs = [r for r in metrics['runs'] if r['fold'] == fold] + [
            {**r, 'variant': 'baseline'} for r in previous['runs'] if r['fold'] == fold and r['variant'] == 'full']
        for part_name, frame in [('fit', fit), ('inner', inner), ('held', held)]:
            x = transformer.transform(frame)
            for row in runs:
                booster = lgb.Booster(model_file=row['model'])
                scores = booster.predict(x[FEATURES], num_threads=4)
                gains = booster.feature_importance(importance_type='gain')
                result = {'fold': fold, 'variant': row['variant'], 'seed': row['seed'], 'partition': part_name,
                          **score_statistics(frame, scores),
                          'ad_feature_gain_share': float(sum(g for f, g in zip(FEATURES, gains) if f in AD_FEATURES)/gains.sum())}
                results.append(result)
        print(f'Fixed-model fit/inner/held diagnostics complete for fold{fold}', flush=True)
    groups = pd.read_parquet(root/'oof_outcomes.parquet')
    primary = config['primary_variant']
    delta = groups[primary+'1']-groups.baseline1
    randomized = groups.displayrandom.eq(1)
    groups = groups.assign(random_n=randomized.astype(float), other_n=(~randomized).astype(float),
                           random_delta=delta*randomized, other_delta=delta*(~randomized))
    point, samples = bootstrap_request_means(groups, ['random_n', 'other_n', 'random_delta', 'other_delta'],
                                            config['bootstrap_repeats'], config['bootstrap_seed'])
    heterogeneity = interval(point[2]/point[0]-point[3]/point[1], samples[:, 2]/samples[:, 0]-samples[:, 3]/samples[:, 1], 1000)
    influence = {}
    for name, part in [('all_multi', groups), ('randomized_multi', groups.loc[randomized])]:
        d = part[primary+'1']-part.baseline1
        users = part.assign(delta=d).groupby('user_id').delta.agg(['sum', 'size'])
        losses = (-users['sum']).clip(lower=0).sort_values(ascending=False)
        excluded = users['sum'].nsmallest(5).index
        rest = part.loc[~part.user_id.isin(excluded)]
        loo = 1000*(d.sum()-users['sum'])/(len(part)-users['size'])
        influence[name] = {'positive_users': int(users['sum'].gt(1e-12).sum()), 'negative_users': int(users['sum'].lt(-1e-12).sum()),
            'top5_share_of_negative_user_mass': float(losses.head(5).sum()/losses.sum()),
            'excluding_five_most_negative_users_delta_per1000': float((rest[primary+'1']-rest.baseline1).mean()*1000),
            'leave_one_user_out_delta_min': float(loo.min()), 'leave_one_user_out_delta_max': float(loo.max())}
    write_json(root/'diagnostics.json', {'post_result_exploratory': True, 'no_retraining': True,
        'fixed_model_partition_statistics': results, 'random_minus_nonrandom_effect_interaction_per1000': heterogeneity,
        'influence': influence, 'interpretation': 'Support counts and descriptive gaps do not prove why training fails; objective, capacity/regularization, weak features, sampled labels, historical display and checkpoint noise remain possible causes.'})
    print('Post-result diagnostics saved.')


if __name__ == '__main__':
    main()
