"""Post-result diagnostic: score discovery users with the exact replication models.

This separates evaluation-cohort sensitivity from merely using smaller fit sets.
It is not an additional confirmation sample, and does not change the preregistered
replication decision, model checkpoints, or candidate-count cutoff.
"""
from datetime import datetime, timezone
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.tools/matplotlib').resolve()))
import lightgbm as lgb
import numpy as np
import pandas as pd

from fairjob_ranker.diagnostics.candidate_count_reproduction import META, seed_mean_outcomes, summarize
from fairjob_ranker.features import FeatureTransformer
from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, FEATURES


def main():
    root = Path('reports/candidate-count-reproduction')
    target = root/'matched_models_discovery.json'
    if target.exists():
        raise FileExistsError(target)
    main_result = read_json(root/'metrics.json')
    config = main_result['config']
    data_path = Path(config['data_dir'])/'validation.parquet'
    write_json(root/'matched_models_protocol.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'Post-result diagnostic, explicitly exploratory',
        'reason': 'Replication fit sets have fewer users than original discovery models. Hold the new trained models fixed and change only evaluation cohort.',
        'method': 'Evaluate all 18 frozen fold models on original discovery users; report per-fold-model results and mean per-request outcomes over all nine full models. Do not refit or select.',
        'main_result_sha256': sha256(root/'metrics.json'), 'source_sha256': sha256(Path(__file__)),
        'validation_sha256': sha256(data_path), 'original_test_read': False})
    frame = pd.read_parquet(data_path)
    fold_reports, outcomes = {}, []
    for fold in range(config['outer_folds']):
        runs = [r for r in main_result['runs'] if r['fold'] == fold]
        folder = Path(runs[0]['model']).parent
        transformer = FeatureTransformer.load(folder/'features.json')
        x = transformer.transform(frame)
        predictions = frame[META].copy()
        for row in runs:
            path = Path(row['model'])
            assert sha256(path) == row['model_sha256']
            cols = FEATURES if row['variant'] == 'full' else AD_FEATURES
            model = lgb.Booster(model_file=str(path))
            predictions[f"{row['variant']}-s{row['seed']}"] = model.predict(x[cols], num_threads=4)
        groups, _ = seed_mean_outcomes(frame, predictions, config['seeds'])
        groups.to_parquet(root/f'discovery-refit-fold-{fold}.parquet', index=False)
        outcomes.append(groups)
        fold_reports[str(fold)] = summarize(groups, config)
        value = fold_reports[str(fold)]['cohorts']['all_large']['comparisons']['full-uniform-top1']
        print(f"Same fold-{fold} model, discovery large group: {value['estimate']:+.3f} /1000, CI={value['ci95']}", flush=True)
    combined = outcomes[0].copy()
    for group in outcomes[1:]:
        np.testing.assert_array_equal(combined.impression_id, group.impression_id)
    for column in combined.columns:
        if column.startswith(('full', 'ad_only')):
            combined[column] = np.mean([g[column].to_numpy() for g in outcomes], axis=0)
    write_json(target, {'protocol_sha256': sha256(root/'matched_models_protocol.json'),
                       'fold_models_on_discovery': fold_reports, 'nine_model_outcome_mean_on_discovery': summarize(combined, config),
                       'primary_replication_result_changed': False,
                       'limitation': 'Adaptive diagnostic on the already observed discovery sample; not new evidence from an independent population. Does not identify the causal reason for cohort differences.'})


if __name__ == '__main__':
    main()
