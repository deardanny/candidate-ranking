"""Bounded, spawned shadow scorer. HTTP responses never wait for the child."""
import multiprocessing as mp
from queue import Empty, Full
from time import perf_counter, sleep

from ..serving import RankRequest
from .service import Ranker


def compare(active, shadow):
    if active['schema_version'] != shadow['schema_version'] or active['catalog_version'] != shadow['catalog_version']:
        raise ValueError('Shadow contract mismatch')
    left = {r['product_id']:r['score'] for r in active['rankings']}
    right = {r['product_id']:r['score'] for r in shadow['rankings']}
    shared = set(left)&set(right)
    return {'active_version':active['model_version'],'shadow_version':shadow['model_version'],
            'top1_agreement':active['rankings'][0]['product_id']==shadow['rankings'][0]['product_id'],
            'topk_overlap':len(shared)/len(set(left)|set(right)),
            'max_shared_score_delta':max((abs(left[i]-right[i]) for i in shared),default=None),
            'shadow_degraded':shadow['degraded']}


def shadow_worker(directory, incoming, outgoing, ready, fault_delay=0., fault_error=False):
    ranker = None
    try:
        ranker = Ranker(directory)
        ready.send({'ready':True,'version':ranker.bundle.metadata['model_version']})
        while True:
            item = incoming.get()
            if item is None:
                break
            request, active, enqueued = item
            try:
                if fault_delay:
                    sleep(fault_delay)
                if fault_error:
                    raise RuntimeError('Injected shadow failure')
                result = compare(active,ranker.rank(RankRequest.model_validate(request)))
                result.update(status='compared',elapsed_ms=(perf_counter()-enqueued)*1000)
            except Exception as error:
                result = {'status':'error','error':type(error).__name__}
            # Results are bounded too. The parent drains between enqueue calls.
            outgoing.put(result)
    except Exception as error:
        try:
            ready.send({'ready':False,'error':type(error).__name__})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if ranker:
            ranker.close()
        ready.close()


class Shadow:
    def __init__(self, directory, capacity=16, fault_delay=0., fault_error=False):
        ctx = mp.get_context('spawn')
        self.incoming, self.outgoing = ctx.Queue(maxsize=capacity),ctx.Queue(maxsize=capacity)
        receiver, sender = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=shadow_worker,args=(str(directory),self.incoming,self.outgoing,sender,fault_delay,fault_error),daemon=True)
        self.process.start()
        sender.close()
        try:
            if not receiver.poll(30):
                raise TimeoutError('Shadow startup timed out')
            state = receiver.recv()
            if not state.get('ready'):
                raise RuntimeError(f'Shadow startup failed: {state}')
            self.version = state['version']
        except BaseException:
            self.close()
            raise
        finally:
            receiver.close()

    def submit(self, request, active):
        if not self.process.is_alive():
            return False
        try:
            self.incoming.put_nowait((request,active,perf_counter()))
            return True
        except Full:
            return False

    def drain(self):
        results = []
        while True:
            try:
                results.append(self.outgoing.get_nowait())
            except Empty:
                return results

    def close(self):
        if self.process.is_alive():
            try:
                self.incoming.put_nowait(None)
            except Full:
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        for q in [self.incoming,self.outgoing]:
            q.cancel_join_thread()
            q.close()
