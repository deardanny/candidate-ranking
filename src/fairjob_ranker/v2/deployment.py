"""Fail-closed, artifact-bound deployment gates and atomic in-process swaps."""
from pathlib import Path
from threading import Lock
import math

from ..io import read_json, sha256
from .service import Ranker


def check_evidence(directory, active_directory, quality, performance):
    candidate_digest = sha256(Path(directory)/'metadata.json')
    active_digest = sha256(Path(active_directory)/'metadata.json')
    reasons = []
    for label, evidence in [('quality',quality),('performance',performance)]:
        if evidence.get('candidate_metadata_sha256') != candidate_digest:
            reasons.append(label+'_candidate_mismatch')
    if quality.get('champion_metadata_sha256') != active_digest:
        reasons.append('quality_champion_mismatch')
    checks = quality.get('checks',{})
    required = {'inner_logloss_guard','outer_logloss_guard','outer_ndcg_point_estimate','outer_ndcg_noninferiority'}
    if set(checks) != required or any(v is not True for v in checks.values()) or quality.get('passed') is not True:
        reasons.append('quality_gate_failed')
    if performance.get('evaluation_split') != 'validation':
        reasons.append('performance_split_mismatch')
    if performance.get('scheduled',0) < 500 or performance.get('duration_seconds',0) < 10:
        reasons.append('insufficient_measurement')
    rate = performance.get('on_time_success_rate')
    if type(rate) not in (float,int) or not math.isfinite(rate) or not .999 <= rate <= 1 or performance.get('deadline_ms') != 50:
        reasons.append('deadline_gate_failed')
    if performance.get('invalid_responses') != 0 or performance.get('degraded_successful') != 0:
        reasons.append('invalid_or_fallback_response')
    if performance.get('generator_dropped') != 0:
        reasons.append('generator_dropped_requests')
    return reasons


class Deployment:
    def __init__(self, directory, redis_url=None, alias_callback=None):
        self.active = Ranker(directory,redis_url)
        self.redis_url, self.alias_callback = redis_url,alias_callback
        self.previous, self.retired = None,[]
        self.lock = Lock()
        self.transition = Lock()
        self.history = []

    def snapshot(self):
        with self.lock:
            return self.active

    def promote(self, directory, quality_path, performance_path, warmup_request):
        # Serialize control operations, but never hold the request snapshot lock
        # while loading a model or waiting for the registry database.
        with self.transition:
            active = self.snapshot()
            reasons = check_evidence(directory,active.bundle.directory,read_json(Path(quality_path)),read_json(Path(performance_path)))
            if reasons:
                raise ValueError(';'.join(reasons))
            staged = Ranker(directory,self.redis_url)
            try:
                response = staged.rank(warmup_request)
                if response['scoring_method']=='smoothed_ad_ctr':
                    raise ValueError('Staged model warmup failed')
                if self.alias_callback:
                    self.alias_callback(staged.bundle.directory)
                with self.lock:
                    if self.previous:
                        self.retired.append(self.previous)
                    self.previous,self.active = self.active,staged
                    self.history.append({'action':'promote','from':active.bundle.metadata['model_version'],'to':staged.bundle.metadata['model_version']})
            except BaseException:
                staged.close()
                raise
            return self.history[-1]

    def rollback(self):
        with self.transition:
            if self.previous is None:
                raise ValueError('No previous verified model')
            if self.alias_callback:
                self.alias_callback(self.previous.bundle.directory)
            with self.lock:
                current = self.active
                self.active,self.previous = self.previous,self.active
                self.history.append({'action':'rollback','from':current.bundle.metadata['model_version'],'to':self.active.bundle.metadata['model_version']})
            return self.history[-1]

    def close(self):
        seen = set()
        for service in [self.active,self.previous,*self.retired]:
            if service is not None and id(service) not in seen:
                service.close(); seen.add(id(service))
