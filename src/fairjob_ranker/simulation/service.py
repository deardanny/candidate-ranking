"""Local synthetic experiment API with durable, idempotent exposure/feedback logs.

SQLite is a single-host demonstration store. It is not a production event queue.
The server never reads personas' private state or generates its own click labels.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..io import sha256
from ..log_contract import RequestSnapshot, ClosedFeedback
from .policies import Policies
from .world import FEATURES, canonical, digest, features


def now():
    return datetime.now(timezone.utc)


def assignment(salt, user_id):
    # Independent of observed/private preferences and of request order.
    return 'A' if int(hashlib.sha256(f'{salt}:{user_id}'.encode()).hexdigest()[:16], 16) < 2**63 else 'B'


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Context(Strict):
    interest: int = Field(ge=0, le=5)
    budget: float = Field(gt=0, le=1000)
    max_distance: float = Field(gt=0, le=1000)
    available_minutes: float = Field(gt=0, le=10000)


class RankRequest(Strict):
    schema_version: Literal['synthetic-rank-v1'] = 'synthetic-rank-v1'
    experiment_id: str = Field(min_length=1, max_length=60)
    request_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(pattern=r'^eval-[0-9]{5}$')
    visit_index: int = Field(ge=0, le=7)
    context: Context
    candidate_ids: list[str] = Field(min_length=8, max_length=8)

    @model_validator(mode='after')
    def unique(self):
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('Duplicate candidates')
        return self


class Feedback(Strict):
    request_id: str = Field(min_length=1, max_length=128)
    request_snapshot_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    display_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    clicked_item_ids: list[str] = Field(max_length=1)
    attribution_window_closed_at: str
    synthetic: Literal[True] = True


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path, protocol_sha):
        self.lock = Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS assignments(
            experiment_id TEXT, user_id TEXT, arm TEXT NOT NULL CHECK(arm IN ('A','B')),
            assigned_at TEXT NOT NULL, PRIMARY KEY(experiment_id,user_id));
          CREATE TABLE IF NOT EXISTS requests(
            request_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, user_id TEXT NOT NULL,
            visit_index INTEGER NOT NULL, arm TEXT NOT NULL, policy TEXT NOT NULL,
            request_digest TEXT NOT NULL, snapshot_digest TEXT NOT NULL, display_digest TEXT NOT NULL,
            snapshot TEXT NOT NULL, response TEXT NOT NULL, served_at TEXT NOT NULL,
            UNIQUE(experiment_id,user_id,visit_index),
            FOREIGN KEY(experiment_id,user_id) REFERENCES assignments(experiment_id,user_id));
          CREATE TABLE IF NOT EXISTS feedback(
            request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
            payload_digest TEXT NOT NULL, clicked_item_id TEXT,
            window_closed_at TEXT NOT NULL, received_at TEXT NOT NULL, body TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS request_experiment ON requests(experiment_id,arm,user_id);
        ''')
        existing = self.db.execute("SELECT value FROM metadata WHERE key='protocol_sha256'").fetchone()
        if existing and existing[0] != protocol_sha:
            raise ValueError('Database belongs to another frozen protocol')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('protocol_sha256',?)", (protocol_sha,))

    def cached(self, request):
        with self.lock:
            row = self.db.execute('SELECT request_digest,response FROM requests WHERE request_id=?', (request.request_id,)).fetchone()
            if not row:
                return None
            if row['request_digest'] != digest(request.model_dump()):
                raise Conflict('Request ID reused with different content')
            return json.loads(row['response'])

    def save_request(self, request, arm, policy, snapshot, result):
        with self.lock, self.db:
            existing = self.db.execute('SELECT request_digest,response FROM requests WHERE request_id=?', (request.request_id,)).fetchone()
            if existing:
                if existing['request_digest'] != digest(request.model_dump()):
                    raise Conflict('Conflicting request retry')
                return json.loads(existing['response'])
            self.db.execute('INSERT OR IGNORE INTO assignments VALUES (?,?,?,?)',
                            (request.experiment_id, request.user_id, arm, result['served_at']))
            saved = self.db.execute('SELECT arm FROM assignments WHERE experiment_id=? AND user_id=?',
                                    (request.experiment_id, request.user_id)).fetchone()[0]
            if saved != arm:
                raise Conflict('Assignment changed')
            try:
                self.db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                (request.request_id, request.experiment_id, request.user_id, request.visit_index,
                                 arm, policy, digest(request.model_dump()), snapshot.digest(), result['display_sha256'],
                                 canonical(snapshot.model_dump(mode='json')), canonical(result), result['served_at']))
            except sqlite3.IntegrityError as error:
                raise Conflict('Visit already served with a different request ID') from error
        return result

    def save_feedback(self, event):
        body_digest = digest(event.model_dump())
        with self.lock, self.db:
            row = self.db.execute('SELECT * FROM requests WHERE request_id=?', (event.request_id,)).fetchone()
            if not row:
                raise ValueError('Feedback references an unknown request')
            previous = self.db.execute('SELECT payload_digest FROM feedback WHERE request_id=?', (event.request_id,)).fetchone()
            if previous:
                if previous[0] != body_digest:
                    raise Conflict('Conflicting closed feedback retry')
                return {'accepted': True, 'duplicate': True, 'synthetic': True}
            snapshot = RequestSnapshot.model_validate_json(row['snapshot'])
            feedback = ClosedFeedback.model_validate({
                'request_id': event.request_id, 'request_snapshot_sha256': event.request_snapshot_sha256,
                'attribution_window_closed_at': event.attribution_window_closed_at,
                'received_at': now(), 'clicked_item_ids': event.clicked_item_ids})
            feedback.validate_against(snapshot)
            response = json.loads(row['response'])
            if event.display_sha256 != row['display_digest']:
                raise ValueError('Displayed ranking hash mismatch')
            if not set(event.clicked_item_ids) <= {r['item_id'] for r in response['rankings']}:
                raise ValueError('Click outside displayed top three')
            self.db.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?)',
                            (event.request_id, body_digest, next(iter(event.clicked_item_ids), None),
                             feedback.attribution_window_closed_at.isoformat(), feedback.received_at.isoformat(),
                             canonical(feedback.model_dump(mode='json'))))
        return {'accepted': True, 'duplicate': False, 'synthetic': True}

    def close(self):
        self.db.close()


def create_app(root):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    config = protocol['configuration']
    # Verify all policy-visible artifacts, including fixture values used in features.
    for name, expected in protocol['artifact_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    offers = json.loads((root/'catalog.json').read_text())
    catalog = {a['item_id']: a for a in offers}
    policies = Policies(root/'models')
    experiments = {e['id']: e for e in config['experiments']}
    registry = CollectorRegistry()
    rank_count = Counter('synthetic_rank_requests', 'Rank endpoint outcomes', ['status'], registry=registry)
    event_count = Counter('synthetic_feedback', 'Closed synthetic feedback', ['status'], registry=registry)
    timing = Histogram('synthetic_rank_seconds', 'Rank endpoint latency, including durable log write', registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(root/'events.sqlite', sha256(root/'protocol.json'))
        yield
        app.state.store.close()

    app = FastAPI(title='Synthetic ranking A/B laboratory', lifespan=lifespan)

    @app.get('/readyz')
    def ready():
        return {'ready': True, 'synthetic': True, 'protocol_sha256': sha256(root/'protocol.json'),
                'models': policies.versions}

    @app.get('/metrics')
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post('/rank')
    def rank(request: RankRequest):
        start = perf_counter()
        try:
            cached = app.state.store.cached(request)
            if cached is not None:
                rank_count.labels('retry').inc()
                return cached
            if request.experiment_id not in experiments:
                raise ValueError('Unknown experiment')
            if int(request.user_id.split('-')[1]) >= config['evaluation_users']:
                raise ValueError('User outside synthetic cohort')
            if any(i not in catalog for i in request.candidate_ids):
                raise ValueError('Unknown candidate')
            if request.visit_index < 4 and any(catalog[i]['cold'] for i in request.candidate_ids):
                raise ValueError('Cold candidate is not released yet')
            arm = assignment(config['assignment_salt'], request.user_id)
            policy = experiments[request.experiment_id]['policies'][arm]
            source_time = now()
            matrix = [features(request.context.model_dump(), catalog[i]) for i in request.candidate_ids]
            available = now()
            scores = policies.score(policy, matrix, request.candidate_ids)
            ranked = sorted(zip(request.candidate_ids, scores), key=lambda x: (-float(x[1]), x[0]))[:3]
            served = now()
            snapshot = RequestSnapshot.model_validate({
                'request_id': request.request_id, 'user_key': request.user_id,
                'served_at': served, 'policy_version': policy, 'model_version': policies.versions[policy],
                'candidates': [{'item_id': i, 'features': {'model_inputs': {
                    'value': vector, 'version': 'synthetic-' + digest(FEATURES)[:12],
                    'source_event_at': source_time, 'computed_at': available, 'available_at': available}}}
                    for i, vector in zip(request.candidate_ids, matrix)]})
            rankings = [{'item_id': i, 'score': float(s), 'position': n + 1} for n, (i, s) in enumerate(ranked)]
            display_hash = digest({'request_snapshot_sha256': snapshot.digest(), 'rankings': rankings})
            result = {'schema_version': 'synthetic-ranking-v1', 'synthetic': True,
                      'experiment_id': request.experiment_id, 'request_id': request.request_id,
                      'user_id': request.user_id, 'arm': arm, 'policy': policy,
                      'model_version': policies.versions[policy], 'served_at': served.isoformat(),
                      'request_snapshot_sha256': snapshot.digest(), 'display_sha256': display_hash,
                      'rankings': rankings, 'inference_and_snapshot_ms': 1000 * (perf_counter() - start)}
            result = app.state.store.save_request(request, arm, policy, snapshot, result)
            rank_count.labels('served').inc()
            return result
        except Conflict as error:
            rank_count.labels('conflict').inc()
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            rank_count.labels('invalid').inc()
            raise HTTPException(400, str(error)) from error
        finally:
            timing.observe(perf_counter() - start)

    @app.post('/events/feedback')
    def feedback(event: Feedback):
        try:
            result = app.state.store.save_feedback(event)
            event_count.labels('duplicate' if result['duplicate'] else 'accepted').inc()
            return result
        except Conflict as error:
            event_count.labels('conflict').inc()
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            event_count.labels('invalid').inc()
            raise HTTPException(400, str(error)) from error

    return app


def app_factory():
    return create_app(Path(os.environ['SYNTHETIC_AB_RUN']))
