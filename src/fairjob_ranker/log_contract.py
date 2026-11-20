"""Acceptance contract for future real logs, not reconstructed EB-NeRD fields.

The current EB-NeRD experiment cannot supply these missing timestamps. Do not
manufacture them from request time. Collect them when requests are served and
feedback is received. Snapshot hashes detect later mutation, not dishonest logs.
"""
from datetime import datetime
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class LoggedFeature(StrictRecord):
    value: float | list[float]
    version: str = Field(min_length=1)
    source_event_at: datetime
    available_at: datetime
    computed_at: datetime

    @model_validator(mode='after')
    def validate_times(self):
        for t in (self.source_event_at, self.available_at, self.computed_at):
            if t.tzinfo is None: raise ValueError('Timezone-aware timestamps required')
        if not self.source_event_at <= self.computed_at <= self.available_at:
            raise ValueError('Require event <= computation <= availability')
        return self


class LoggedCandidate(StrictRecord):
    item_id: str = Field(min_length=1)
    features: dict[str, LoggedFeature] = Field(min_length=1)


class RequestSnapshot(StrictRecord):
    schema_version: Literal['ranking-request-v1'] = 'ranking-request-v1'
    request_id: str = Field(min_length=1)
    user_key: str = Field(min_length=1)
    served_at: datetime
    policy_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    candidates: list[LoggedCandidate] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_snapshot(self):
        if self.served_at.tzinfo is None: raise ValueError('Timezone-aware served_at required')
        ids=[c.item_id for c in self.candidates]
        if len(set(ids))!=len(ids): raise ValueError('Duplicate candidate')
        for c in self.candidates:
            for f in c.features.values():
                if f.available_at > self.served_at: raise ValueError('Feature unavailable when served')
        return self

    def digest(self):
        canonical=json.dumps(self.model_dump(mode='json'),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


class ClosedFeedback(StrictRecord):
    schema_version: Literal['ranking-feedback-v1'] = 'ranking-feedback-v1'
    request_id: str = Field(min_length=1)
    request_snapshot_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    attribution_window_closed_at: datetime
    received_at: datetime
    clicked_item_ids: list[str]

    @model_validator(mode='after')
    def validate_times(self):
        if self.attribution_window_closed_at.tzinfo is None or self.received_at.tzinfo is None:
            raise ValueError('Timezone-aware feedback timestamps required')
        if self.received_at < self.attribution_window_closed_at:
            raise ValueError('Closed feedback cannot arrive before its attribution window closes')
        if len(set(self.clicked_item_ids))!=len(self.clicked_item_ids): raise ValueError('Duplicate binary click target')
        return self

    def validate_against(self, request: RequestSnapshot):
        if self.request_id!=request.request_id or self.request_snapshot_sha256!=request.digest():
            raise ValueError('Request or immutable feature snapshot mismatch')
        if self.attribution_window_closed_at <= request.served_at:
            raise ValueError('Attribution window must end after serving')
        if not set(self.clicked_item_ids)<={c.item_id for c in request.candidates}:
            raise ValueError('Click outside recorded candidates')

    def usable_before(self, cutoff: datetime):
        """Both training and prior-response features use the same arrival clock."""
        if cutoff.tzinfo is None: raise ValueError('Timezone-aware cutoff required')
        return self.received_at < cutoff
