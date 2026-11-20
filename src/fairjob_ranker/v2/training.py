"""Preregistered 2x2 development experiment with local MLflow tracking/registry."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import random
import shutil
from time import perf_counter

import lightgbm as lgb
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..evaluation import click_metrics, evaluate, paired_user_bootstrap
from ..features import FeatureTransformer
from ..io import read_json, sha256, write_json
from ..schema import CATEGORICAL, FEATURES, VERSION
from ..torch_model import EmbeddingMLP, predict_tensors, to_tensors
from .models import DeepCross, PairSampler, load_bundle


class RankerPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from fairjob_ranker.v2.models import load_bundle
        self.bundle = load_bundle(Path(context.artifacts['bundle']))

    def predict(self, context, model_input, params=None):
        return self.bundle.predict(model_input)


def development_split(frame, salt, percent):
    users = sorted(frame.user_id.unique())
    assignments = {u: int.from_bytes(hashlib.sha256(f'{salt}:{int(u)}'.encode()).digest()[:8], 'big') % 100 < percent for u in users}
    mask = frame.user_id.map(assignments)
    train, valid = frame.loc[mask].reset_index(drop=True), frame.loc[~mask].reset_index(drop=True)
    for key in ['user_id', 'impression_id']:
        if set(train[key]) & set(valid[key]):
            raise ValueError('Development split leakage')
    return train, valid


def fit_network(kind, weight, seed, config, architecture, train_tensors, valid_tensors, labels, valid_labels, pairs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = (DeepCross if kind == 'dcn' else EmbeddingMLP)(**architecture)
    nn.init.normal_(model.network[-1].weight, std=.01)
    nn.init.constant_(model.network[-1].bias, float(torch.logit(labels.mean())))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    bce = nn.BCEWithLogitsLoss()
    # Pair sampling never consumes the row permutation RNG.
    rng = np.random.default_rng(seed)
    ct, nt = train_tensors
    best, best_loss, stale, history = None, float('inf'), 0, []
    started = perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(labels))
        totals, pair_total, steps = 0., 0., 0
        for start in range(0, len(order), config['batch_size']):
            ids = order[start:start + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            point_loss = bce(model(ct[ids], nt[ids]), labels[ids])
            pair_loss = torch.tensor(0.)
            if weight:
                p, n = pairs.sample(config['pair_groups_per_batch'], rng)
                pair_loss = nn.functional.softplus(-(model(ct[p], nt[p]) - model(ct[n], nt[n]))).mean()
            loss = point_loss + weight * pair_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            totals += point_loss.item() * len(ids)
            pair_total += pair_loss.item()
            steps += 1
        scores = predict_tensors(model, *valid_tensors)
        valid_loss = click_metrics(valid_labels, scores)['logloss']
        history.append({'epoch': epoch, 'train_bce': totals / len(labels), 'mean_pair_loss': pair_total / steps, 'inner_logloss': valid_loss})
        print(f'{kind} pair={weight} seed={seed} epoch={epoch}: inner LogLoss={valid_loss:.6f}', flush=True)
        if valid_loss < best_loss:
            best_loss, stale, best_epoch = valid_loss, 0, epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config['patience']:
                break
    model.load_state_dict(best)
    return model, history, best_epoch, perf_counter() - started


def save_bundle(directory, model, kind, architecture, transformer, fallback, data, common, name):
    directory.mkdir(parents=True)
    weights = 'model.txt' if kind == 'lightgbm' else 'weights.pt'
    if kind == 'lightgbm':
        model.save_model(str(directory / weights))
    else:
        torch.save(model.state_dict(), directory / weights)
    transformer.save(directory / 'features.json')
    shutil.copyfile(data / 'ad_catalog.parquet', directory / 'ad_catalog.parquet')
    write_json(directory / 'fallback.json', fallback)
    artifacts = {n: sha256(directory / n) for n in [weights, 'features.json', 'ad_catalog.parquet', 'fallback.json']}
    meta = {**common, 'bundle_version': 2, 'schema_version': VERSION, 'feature_order': FEATURES,
            'kind': kind, 'architecture': architecture, 'model_version': name + ':' + artifacts[weights][:12],
            'artifact_sha256': artifacts, 'feature_version': artifacts['features.json'],
            'catalog_version': artifacts['ad_catalog.parquet']}
    write_json(directory / 'metadata.json', meta)
    return load_bundle(directory)


def register_bundle(bundle, sample, run_id, registry_name):
    predicted = bundle.predict(sample)
    info = mlflow.pyfunc.log_model(
        name='ranker', python_model=RankerPythonModel(), artifacts={'bundle': str(bundle.directory.resolve())},
        code_paths=['src/fairjob_ranker'], signature=infer_signature(sample, predicted),
        pip_requirements=[f'{n}=={version(n)}' for n in ['torch','numpy','pandas','lightgbm','scikit-learn','pyarrow']],
    )
    registered = mlflow.register_model(info.model_uri, registry_name)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(registry_name, registered.version, 'bundle_metadata_sha256', sha256(bundle.directory / 'metadata.json'))
    client.set_model_version_tag(registry_name, registered.version, 'local_bundle', str(bundle.directory.resolve()))
    return {'run_id': run_id, 'registry_version': registered.version, 'model_uri': info.model_uri}


def run_experiment(data, config_path, output, model_root, tracking_uri):
    for path in [output, model_root]:
        if path.exists():
            raise FileExistsError(path)
    config = read_json(config_path)
    output.mkdir(parents=True)
    model_root.mkdir(parents=True)
    torch.set_num_threads(config['num_threads'])
    manifest = read_json(data / 'manifest.json')
    for filename in ['train.parquet', 'validation.parquet', 'ad_catalog.parquet']:
        if sha256(data / filename) != manifest['files'][filename]['sha256']:
            raise ValueError(f'Dataset modified: {filename}')
    # Do not open test.parquet, even for shape or checksum verification.
    original_train = pd.read_parquet(data / 'train.parquet')
    train, inner = development_split(original_train, config['development_split_salt'], config['inner_train_percent'])
    transformer = FeatureTransformer.fit(train)
    xt, xi = transformer.transform(train), transformer.transform(inner)
    tensors, valid_tensors = to_tensors(xt), to_tensors(xi)
    labels = torch.from_numpy(train.click.to_numpy(dtype=np.float32))
    pairs = PairSampler(train)
    counts = train.groupby('product_id').click.agg(['sum','size'])
    ctr, alpha = float(train.click.mean()), config['fallback_alpha']
    fallback = {'fitted_on': 'inner_train', 'alpha': alpha, 'train_ctr': ctr,
                'scores': {str(int(i)): float((r['sum']+alpha*ctr)/(r['size']+alpha)) for i,r in counts.iterrows()}}
    preregistered = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                    'config_sha256': sha256(config_path), 'dataset_manifest_sha256': sha256(data/'manifest.json'),
                    'test_read': False, 'inner_train_rows':len(train), 'inner_validation_rows':len(inner),
                    'inner_train_users':int(train.user_id.nunique()), 'inner_validation_users':int(inner.user_id.nunique()),
                    'pairwise_groups':len(pairs.groups), 'tracking_uri':tracking_uri,
                    'source_code_sha256':{str(p):sha256(p) for p in sorted(Path('src').rglob('*.py'))}}
    write_json(output/'preregistered.json',preregistered)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(config['experiment_name'])
    registry_name = 'fairjob-v2'
    products = set(train.product_id)
    rows, groups, predictions = [], {}, {}
    sample = train[FEATURES].head(2)
    for kind in ['mlp','dcn']:
        for pair_weight in [0.,config['pairwise_weight']]:
            family = f"{kind}-{'pair' if pair_weight else 'bce'}"
            for seed in config['seeds']:
                name = f'{family}-s{seed}'
                arch = {'category_sizes':[len(transformer.vocabularies[c])+1 for c in CATEGORICAL],
                        'embedding_dim':config['embedding_dim'],'hidden_dims':config['hidden_dims'],'dropout':config['dropout']}
                if kind == 'dcn':
                    arch['cross_layers'] = config['cross_layers']
                with mlflow.start_run(run_name=name) as run:
                    mlflow.log_params({'kind':kind,'pair_weight':pair_weight,'seed':seed,'config_sha256':sha256(config_path),
                                       'dataset_sha256':preregistered['dataset_manifest_sha256']})
                    model, history, best_epoch, seconds = fit_network(kind,pair_weight,seed,config,arch,tensors,valid_tensors,labels,inner.click.to_numpy(),pairs)
                    scores = predict_tensors(model,*valid_tensors)
                    metrics, ranked = evaluate(inner,scores,products)
                    bundle = save_bundle(model_root/name,model,kind,arch,transformer,fallback,data,
                                         {'experiment_config_sha256':sha256(config_path),'inner_training':True,'seed':seed,'pair_weight':pair_weight},name)
                    np.testing.assert_allclose(scores,bundle.predict(inner),rtol=1e-5,atol=1e-6)
                    registration = register_bundle(bundle,sample,run.info.run_id,registry_name)
                    row = {'name':name,'family':family,'seed':seed,'kind':kind,'pair_weight':pair_weight,
                           'best_epoch':best_epoch,'history':history,'training_seconds':seconds,
                           'parameters':sum(p.numel() for p in model.parameters()), 'inner':metrics,
                           'bundle':str(bundle.directory),'model_version':bundle.metadata['model_version'],**registration}
                    mlflow.log_metrics({'inner_logloss':metrics['all']['click']['logloss'],'inner_ndcg':metrics['all']['ranking']['ndcg_at_5'],
                                        'training_seconds':seconds,'parameters':row['parameters']})
                    mlflow.log_artifact(str(output/'preregistered.json'))
                    mlflow.log_dict(row,'result.json')
                    rows.append(row); groups[name] = ranked; predictions[name] = scores
                    write_json(output/'progress.json',{'completed':len(rows),'runs':rows})
    print('Fitting LightGBM on exactly the same inner training split',flush=True)
    lgb_config = read_json(Path('configs/lightgbm_baseline.json'))
    started = perf_counter()
    baseline = lgb.train(lgb_config['params'], lgb.Dataset(xt,label=train.click,categorical_feature=CATEGORICAL),
                         num_boost_round=lgb_config['num_boost_round'],
                         valid_sets=[lgb.Dataset(xi,label=inner.click,categorical_feature=CATEGORICAL)],
                         callbacks=[lgb.early_stopping(lgb_config['early_stopping_rounds'],verbose=False)])
    with mlflow.start_run(run_name='lightgbm-s42') as run:
        bundle = save_bundle(model_root/'lightgbm-s42',baseline,'lightgbm',{},transformer,fallback,data,
                             {'experiment_config_sha256':sha256(config_path),'inner_training':True,'seed':42},'lightgbm-s42')
        scores = bundle.predict(inner)
        metrics, ranked = evaluate(inner,scores,products)
        registration = register_bundle(bundle,sample,run.info.run_id,registry_name)
        rows.append({'name':'lightgbm-s42','family':'lightgbm','seed':42,'kind':'lightgbm','inner':metrics,
                     'training_seconds':perf_counter()-started,'bundle':str(bundle.directory),
                     'model_version':bundle.metadata['model_version'],**registration})
        mlflow.log_params(lgb_config['params'])
        mlflow.log_metrics({'inner_logloss':metrics['all']['click']['logloss'],'inner_ndcg':metrics['all']['ranking']['ndcg_at_5']})
        mlflow.log_artifact(str(output/'preregistered.json'))
        groups['lightgbm-s42'] = ranked
    families = {}
    for family in ['mlp-bce','mlp-pair','dcn-bce','dcn-pair']:
        members = [r for r in rows if r['family']==family]
        values = lambda section,key:[r['inner']['all'][section][key] for r in members]
        families[family] = {'logloss_mean':float(np.mean(values('click','logloss'))),
                            'ndcg_mean':float(np.mean(values('ranking','ndcg_at_5'))),
                            'ndcg_seed_std':float(np.std(values('ranking','ndcg_at_5'),ddof=1))}
    eligible = [f for f,v in families.items() if f!='mlp-bce' and v['logloss_mean']<=families['mlp-bce']['logloss_mean']*1.005]
    selected_family = sorted(eligible or [f for f in families if f!='mlp-bce'],key=lambda f:(-families[f]['ndcg_mean'],f))[0]
    selected_name = f"{selected_family}-s{config['representative_seed']}"
    selection = {'family':selected_family,'name':selected_name,'family_passed_logloss_guard':selected_family in eligible,
                 'families':families,'selected_before_outer_validation':True}
    write_json(output/'selection.json',selection)
    # Outer validation is only read after family and representative seed are fixed.
    outer = pd.read_parquet(data/'validation.parquet')
    outer_groups = {}
    for row in rows:
        b = load_bundle(row['bundle'])
        scores = b.predict(outer)
        row['outer'], outer_groups[row['name']] = evaluate(outer,scores,products)
        with mlflow.start_run(run_id=row['run_id']):
            mlflow.log_metrics({'outer_logloss':row['outer']['all']['click']['logloss'],
                                'outer_ndcg':row['outer']['all']['ranking']['ndcg_at_5']})
            mlflow.log_dict(row['outer'],'outer_metrics.json')
    comparison = paired_user_bootstrap(outer_groups[selected_name],outer_groups['lightgbm-s42'],repeats=config['bootstrap_repeats'])
    comparison['difference'] = selected_name+' - lightgbm-s42'
    candidate = next(r for r in rows if r['name']==selected_name)
    champion = next(r for r in rows if r['name']=='lightgbm-s42')
    delta = comparison['all']['ndcg_at_5']
    gate = config['quality_gate']
    checks = {'inner_logloss_guard':selection['family_passed_logloss_guard'],
              'outer_logloss_guard':candidate['outer']['all']['click']['logloss']<=champion['outer']['all']['click']['logloss']*gate['max_logloss_ratio'],
              'outer_ndcg_point_estimate':delta['mean_difference']>=gate['min_ndcg_difference'],
              'outer_ndcg_noninferiority':delta['ci95'][0]>=gate['min_ndcg_ci_lower']}
    quality = {'candidate_metadata_sha256':sha256(Path(candidate['bundle'])/'metadata.json'),
               'champion_metadata_sha256':sha256(Path(champion['bundle'])/'metadata.json'),
               'passed':all(checks.values()),'checks':checks,'thresholds':gate,'comparison':comparison,
               'scope':'Development comparison, no independent test or online uplift claim'}
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(registry_name,'champion',champion['registry_version'])
    client.set_registered_model_alias(registry_name,'challenger',candidate['registry_version'])
    client.set_model_version_tag(registry_name,candidate['registry_version'],'quality_gate','passed' if quality['passed'] else 'blocked')
    result = {'config':config,'tracking_uri':tracking_uri,'registry_name':registry_name,'test_read':False,
              'selection':selection,'quality_gate':quality,'champion':champion,'challenger':candidate,'runs':rows}
    write_json(output/'metrics.json',result)
    write_json(output/'quality_gate.json',quality)
    (output/'progress.json').unlink()
    return result
