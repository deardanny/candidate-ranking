"""Test-only ASGI app. Fault injection is not an endpoint in the normal API."""
from contextlib import asynccontextmanager
import os
import time

from fairjob_ranker.serving import create_app

app = create_app()
original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def faulty_lifespan(application):
    async with original_lifespan(application):
        mode = os.environ.get("FAIRJOB_FAILURE_MODE")
        service = application.state.service
        if service is None:
            raise RuntimeError("Fault test requires a ready service")
        original_forward = service.bundle.model.forward
        if mode == "model_error":
            def fail(*_):
                raise RuntimeError("deliberately injected model error")
            service.bundle.model.forward = fail
        elif mode == "slow_model":
            def slow(*args):
                time.sleep(0.05)
                return original_forward(*args)
            service.bundle.model.forward = slow
        else:
            raise ValueError("Unknown test-only failure mode")
        yield


app.router.lifespan_context = faulty_lifespan
