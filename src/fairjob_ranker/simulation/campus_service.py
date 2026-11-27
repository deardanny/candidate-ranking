"""One-slot campus ad API; reuse the durable request/click contract and add coupons."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CollectorRegistry, Counter, generate_latest, CONTENT_TYPE_LATEST
from pydantic import Field, model_validator

from ..io import sha256
from ..log_contract import RequestSnapshot
from .service import Store, Strict, Conflict, Feedback, assignment, now
from .world import canonical, digest
from .campus_world import FEATURES, Policies, features


class Context(Strict):
    interest: int = Field(ge=0, le=2)
    budget: float = Field(gt=0, le=100)
    minutes_until_class: float = Field(gt=0, le=1440)


class Request(Strict):
    experiment_id: str = Field(min_length=1, max_length=40)
    request_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(pattern=r'^eval-[0-9]{5}$')
    visit_index: int = Field(ge=0, le=7)
    context: Context
    candidate_ids: list[str] = Field(min_length=8, max_length=8)

    @model_validator(mode='after')
    def valid(self):
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('Duplicate candidate')
        return self


class Coupon(Strict):
    request_id: str = Field(min_length=1, max_length=128)
    request_snapshot_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    display_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    redeemed_item_id: str | None
    virtual_redemption_delay_minutes: float | None = Field(default=None, ge=0, le=1440)
    virtual_observation_window_minutes: Literal[1440] = 1440
    virtual_window_closed: Literal[True] = True
    clock: Literal['accelerated-simulation'] = 'accelerated-simulation'
    synthetic: Literal[True] = True

    @model_validator(mode='after')
    def valid(self):
        if (self.redeemed_item_id is None) != (self.virtual_redemption_delay_minutes is None):
            raise ValueError('Redemption and virtual delay must both be present or both absent')
        return self


class CampusStore(Store):
    def __init__(self, path, protocol_sha):
        super().__init__(path, protocol_sha)
        with self.db:
            self.db.execute('''CREATE TABLE IF NOT EXISTS coupons(
              request_id TEXT PRIMARY KEY REFERENCES feedback(request_id),
              payload_digest TEXT NOT NULL, redeemed_item_id TEXT,
              received_at TEXT NOT NULL, body TEXT NOT NULL)''')

    def save_coupon(self, event):
        payload_digest = digest(event.model_dump())
        with self.lock, self.db:
            r = self.db.execute('SELECT r.*,f.clicked_item_id FROM requests r JOIN feedback f USING(request_id) WHERE request_id=?',
                                (event.request_id,)).fetchone()
            if not r:
                raise ValueError('Coupon outcome requires closed click feedback first')
            if event.request_snapshot_sha256 != r['snapshot_digest'] or event.display_sha256 != r['display_digest']:
                raise ValueError('Coupon attribution hash mismatch')
            if event.redeemed_item_id is not None and event.redeemed_item_id != r['clicked_item_id']:
                raise ValueError('Redemption must follow a click on the same displayed item')
            old = self.db.execute('SELECT payload_digest FROM coupons WHERE request_id=?', (event.request_id,)).fetchone()
            if old:
                if old[0] != payload_digest:
                    raise Conflict('Conflicting closed coupon outcome')
                return {'accepted': True, 'duplicate': True, 'synthetic': True}
            self.db.execute('INSERT INTO coupons VALUES (?,?,?,?,?)',
                (event.request_id, payload_digest, event.redeemed_item_id, now().isoformat(), canonical(event.model_dump())))
        return {'accepted': True, 'duplicate': False, 'synthetic': True}


def create_app(path):
    path = Path(path)
    protocol = json.loads((path/'protocol.json').read_text())
    cfg = protocol['configuration']
    for file, expected in protocol['artifact_sha256'].items():
        if sha256(path/file) != expected:
            raise ValueError('Frozen artifact mismatch: ' + file)
    ads = {a['item_id']: a for a in json.loads((path/'catalog.json').read_text())}
    policies = Policies(path/'models')
    experiments = {e['id']: e for e in cfg['experiments']}
    registry = CollectorRegistry()
    events = Counter('campus_events', 'Synthetic campus events', ['kind', 'status'], registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = CampusStore(path/'events.sqlite', sha256(path/'protocol.json'))
        yield
        app.state.store.close()

    app = FastAPI(title='Fictional campus coupon experiment', lifespan=lifespan)

    @app.get('/readyz')
    def ready():
        return {'ready': True, 'synthetic': True, 'models': policies.versions}

    @app.get('/metrics')
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post('/rank')
    def rank(request: Request):
        try:
            cached = app.state.store.cached(request)
            if cached:
                events.labels('rank', 'duplicate').inc()
                return cached
            if request.experiment_id not in experiments or int(request.user_id.split('-')[1]) >= cfg['evaluation_users']:
                raise ValueError('Unknown experiment or user')
            if any(i not in ads or not ads[i]['campaign_eligible'] for i in request.candidate_ids):
                raise ValueError('Unknown or ineligible offer')
            arm = assignment(cfg['assignment_salt'], request.user_id)
            policy = experiments[request.experiment_id]['policies'][arm]
            source = now(); ctx = request.context.model_dump()
            chosen_ads = [ads[i] for i in request.candidate_ids]
            vectors = [features(ctx, ad) for ad in chosen_ads]
            available = now()
            scores = policies.score(policy, ctx, chosen_ads)
            top, score = sorted(zip(request.candidate_ids, scores), key=lambda x: (-float(x[1]), x[0]))[0]
            snapshot = RequestSnapshot.model_validate({
                'request_id': request.request_id, 'user_key': request.user_id, 'served_at': now(),
                'policy_version': policy, 'model_version': policies.versions[policy],
                'candidates': [{'item_id': item, 'features': {'model_inputs': {'value': vector,
                    'version': 'campus-' + digest(FEATURES)[:12], 'source_event_at': source,
                    'computed_at': available, 'available_at': available}}}
                    for item, vector in zip(request.candidate_ids, vectors)]})
            rankings = [{'item_id': top, 'score': float(score), 'position': 1}]
            output = {'request_id': request.request_id, 'experiment_id': request.experiment_id,
                'user_id': request.user_id, 'arm': arm, 'policy': policy, 'synthetic': True,
                'model_version': policies.versions[policy], 'served_at': snapshot.served_at.isoformat(),
                'request_snapshot_sha256': snapshot.digest(), 'rankings': rankings,
                'display_sha256': digest({'request_snapshot_sha256': snapshot.digest(), 'rankings': rankings})}
            output = app.state.store.save_request(request, arm, policy, snapshot, output)
            events.labels('rank', 'accepted').inc()
            return output
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post('/events/feedback')
    def feedback(event: Feedback):
        try:
            result = app.state.store.save_feedback(event)
            events.labels('click', 'duplicate' if result['duplicate'] else 'accepted').inc()
            return result
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post('/events/coupon')
    def coupon(event: Coupon):
        try:
            result = app.state.store.save_coupon(event)
            events.labels('coupon', 'duplicate' if result['duplicate'] else 'accepted').inc()
            return result
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    return app


def app_factory():
    return create_app(os.environ['CAMPUS_COUPON_RUN'])
