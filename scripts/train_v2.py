import argparse
import os
from pathlib import Path

os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY','true')
from fairjob_ranker.v2.training import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=Path('data/processed/fairjob-v1'))
parser.add_argument('--config',type=Path,default=Path('configs/v2_experiment.json'))
parser.add_argument('--output',type=Path,default=Path('reports/v2/experiment'))
parser.add_argument('--models',type=Path,default=Path('models/v2'))
parser.add_argument('--tracking-uri',default='sqlite:///'+str(Path('runs/v2/mlflow.db').resolve()))
args = parser.parse_args()
Path('runs/v2').mkdir(parents=True,exist_ok=True)
result=run_experiment(args.data,args.config,args.output,args.models,args.tracking_uri)
print('Selected:',result['selection']['name'],'quality gate:',result['quality_gate']['checks'])
