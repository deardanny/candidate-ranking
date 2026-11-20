"""Log all development runs and register six independent news bundles locally."""
import json
import os
from importlib.metadata import version
from pathlib import Path
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY','true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from fairjob_ranker.ebnerd.bundle import Bundle
from fairjob_ranker.io import sha256

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'reports/ebnerd-experiment'; modelroot=ROOT/'models/ebnerd-v1'
record=modelroot/'registry.json'
if record.exists(): raise SystemExit('Registry record already exists; refusing duplicate registrations')
runroot=ROOT/'runs/ebnerd'; runroot.mkdir(parents=True,exist_ok=True)
uri='sqlite:///'+str(runroot/'mlflow.db'); mlflow.set_tracking_uri(uri)
client=mlflow.MlflowClient(); name='EB-NeRD temporal ranking v1'
experiment=client.get_experiment_by_name(name)
experiment_id=experiment.experiment_id if experiment else client.create_experiment(name,artifact_location=(runroot/'artifacts').as_uri())
selection=json.loads((out/'selection.json').read_text()); final=json.loads((out/'final_metrics.json').read_text())
for row in selection['runs']:
    with mlflow.start_run(experiment_id=experiment_id,run_name=row['fold']+'-'+row['variant']):
        mlflow.log_params({k:row[k] for k in ['fold','variant','rounds','fit_requests','validation_requests']})
        mlflow.log_metrics(row['metrics']); mlflow.set_tag('evaluation_split','development')
        mlflow.log_artifact(str(out/'protocol.json'))
models=[]; x=np.load(ROOT/'data/processed/ebnerd-v1/train_X.npy',mmap_mode='r')
for variant,metrics in final['metrics'].items():
    directory=modelroot/('final-'+variant)
    if not directory.exists(): continue
    bundle=Bundle(directory); sample=pd.DataFrame(np.array(x[:5,bundle.indices]),columns=bundle.features)
    with mlflow.start_run(experiment_id=experiment_id,run_name='final-'+variant) as run:
        mlflow.log_params({'variant':variant,'rounds':bundle.meta['rounds'],'content_snapshot_assumption':bundle.meta['content_snapshot_assumption']})
        mlflow.log_metrics(metrics); mlflow.set_tag('evaluation_split','frozen_local_test')
        mlflow.log_artifacts(str(directory),artifact_path='bundle')
        info=mlflow.lightgbm.log_model(bundle.model,name='ranker',signature=infer_signature(sample,bundle.model.predict(sample,num_threads=1)),
            pip_requirements=[f'{n}=={version(n)}' for n in ['lightgbm','numpy','pandas','scipy','scikit-learn']])
        registered=mlflow.register_model(info.model_uri,'ebnerd-v1')
        client.set_model_version_tag('ebnerd-v1',registered.version,'bundle_metadata_sha256',bundle.digest)
        client.set_model_version_tag('ebnerd-v1',registered.version,'local_bundle',str(directory))
        models.append({'variant':variant,'version':registered.version,'metadata_sha256':bundle.digest,
            'run_id':run.info.run_id,'model_uri':info.model_uri,'bundle':str(directory)})
client.set_registered_model_alias('ebnerd-v1','champion',next(x['version'] for x in models if x['variant']=='metadata'))
client.set_registered_model_alias('ebnerd-v1','challenger',next(x['version'] for x in models if x['variant']=='personalized'))
record.write_text(json.dumps({'tracking_uri':uri,'registry_name':'ebnerd-v1','experiment_id':experiment_id,'models':models},indent=2)+'\n')
print('registered',len(models),'models; metadata remains champion until serving gate')
