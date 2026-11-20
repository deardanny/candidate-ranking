"""Run the validated news ranker locally; Ctrl+C stops both owned processes."""
from pathlib import Path
import argparse
from ebnerd_runtime import stack

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-port',type=int,default=8001); parser.add_argument('--redis-port',type=int,default=6381)
    args=parser.parse_args()
    try:
        with stack(Path('models/ebnerd-v1/final-personalized'),Path('models/ebnerd-v1/final-metadata'),args.api_port,args.redis_port) as runtime:
            print('News ranking API:',runtime['api_url'],'Swagger:',runtime['api_url']+'/docs',flush=True)
            print('Ctrl+C stops the owned API and Redis.',flush=True)
            runtime['api_process'].wait()
    except KeyboardInterrupt: print('Stopped news demo.')

if __name__=='__main__': main()
