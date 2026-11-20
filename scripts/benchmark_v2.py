import argparse
import asyncio
import json
import os
from pathlib import Path
import random

from fairjob_ranker.io import read_json,sha256,write_json
from fairjob_ranker.loadtest import environment
from fairjob_ranker.replay import load_requests
from fairjob_ranker.v2.load import run_load
from local_runtime import local_stack


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/v2_load.json'))
    parser.add_argument('--output',type=Path,default=Path('reports/v2/load'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=read_json(args.config)
    experiment=read_json(Path('reports/v2/experiment/metrics.json'))
    requests,corpus=load_requests(Path('data/replay/validation.jsonl'))
    assert corpus['split']=='validation'
    output=args.output; output.mkdir(parents=True)
    manifest={'config':config,'config_sha256':sha256(args.config),'corpus':corpus,'environment':environment(),
              'experiment_sha256':sha256(Path('reports/v2/experiment/metrics.json')),
              'source_code_sha256':{str(p):sha256(p) for p in sorted(Path('src').rglob('*.py'))},'runs':[]}
    write_json(output/'preregistered.json',manifest)
    write_json(output/'running.json',{'status':'running'})
    rng=random.Random(config['shuffle_seed'])
    stacks=[(repeat,capacity) for repeat in range(config['repeats']) for capacity in config['capacities']]
    rng.shuffle(stacks)
    for repeat,capacity in stacks:
        settings={'FAIRJOB_MAX_INFLIGHT':str(capacity),'FAIRJOB_SHADOW_RATE':str(config['shadow_sample_rate']),
                  'MLFLOW_DISABLE_TELEMETRY':'true','MLFLOW_DISABLE_AGENT_HINT':'1','MPLCONFIGDIR':str(Path('.tools/matplotlib').resolve())}
        with local_stack(Path('models/pytorch-v1'),8030,6396,output/'logs'/f'r{repeat}-q{capacity}',
                         environment_overrides=settings,api_app='v2_server:app') as stack:
            conditions=[(rate,burst) for rate in config['rates'] for burst in config['burst_sizes']]
            rng.shuffle(conditions)
            for rate,burst in conditions:
                kwargs=dict(burst_size=burst,expected_version=experiment['champion']['model_version'],pids=[stack.api_process.pid,stack.redis_process.pid])
                asyncio.run(run_load(stack.api_url,requests,rate,config['warmup_seconds'],**kwargs))
                result=asyncio.run(run_load(stack.api_url,requests,rate,config['duration_seconds'],**kwargs))
                records=result.pop('records')
                result.update(repeat=repeat,capacity=capacity,shadow_sample_rate=config['shadow_sample_rate'])
                with (output/f'requests-r{repeat}-q{capacity}-{rate}-b{burst}.jsonl').open('w') as f:
                    for row in records:f.write(json.dumps(row,separators=(',',':'))+'\n')
                manifest['runs'].append(result)
                write_json(output/'metrics.json',manifest)
                print(f"repeat={repeat} capacity={capacity} rps={rate} burst={burst}: on-time={result['on_time_success_rate']:.3%} failures={result['failure_rate']:.3%}",flush=True)
    manifest['completed']=True
    write_json(output/'metrics.json',manifest)
    (output/'running.json').unlink()
    print('Completed',len(manifest['runs']),'load runs',flush=True)


if __name__=='__main__':main()
