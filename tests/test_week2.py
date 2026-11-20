import asyncio
import json
import shutil
import threading

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pandas as pd
import pytest
import redis
import torch

from fairjob_ranker.io import read_json, sha256, write_json
from fairjob_ranker.schema import AD_FEATURES, CATEGORICAL, FEATURES, NUMERICAL, VERSION
from fairjob_ranker.serving import Settings, create_app
from fairjob_ranker.torch_model import load_bundle, to_tensors
from fairjob_ranker.torch_training import train_pytorch


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("week2")
    data = root / "data"
    data.mkdir()
    frames = {}
    for name, offset in [("train", 0), ("validation", 1000)]:
        n = 128
        ids = np.arange(n) % 4 + 1
        group = np.arange(n) // 2
        frame = pd.DataFrame({**{c: ids for c in AD_FEATURES},
                              "cat0": group % 3, "cat1": group % 2,
                              **{c: (group % 5).astype(float) for c in NUMERICAL}})
        frame = frame.assign(user_id=group + offset, impression_id=group + offset,
                             product_id=ids, click=np.arange(n) % 2, rank=np.arange(n) % 2 + 1,
                             displayrandom=1)
        frame.to_parquet(data / f"{name}.parquet", index=False)
        frames[name] = frame
    catalog = frames["train"][["product_id", *AD_FEATURES]].drop_duplicates()
    # Product 5 is known in the catalog, but all its categories are OOV in train.
    catalog = pd.concat([catalog, pd.DataFrame([{c: 5 for c in ["product_id", *AD_FEATURES]}])], ignore_index=True)
    catalog.to_parquet(data / "ad_catalog.parquet", index=False)
    (data / "test.parquet").write_bytes(b"TEST MUST NEVER BE READ DURING WEEK 2")
    write_json(data / "manifest.json", {
        "version": VERSION, "source_sha256": "synthetic-week2", "splits": {k: {"rows": len(v)} for k, v in frames.items()},
        "files": {p.name: {"sha256": sha256(p)} for p in data.glob("*.parquet")},
    })
    config = {"run_name": "test-mlp", "seed": 42, "embedding_dim": 2, "hidden_dims": [8],
              "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 32,
              "max_epochs": 2, "patience": 2, "num_threads": 1, "bootstrap_repeats": 20, "fallback_alpha": 100.0}
    write_json(root / "config.json", config)
    result = train_pytorch(data, root / "config.json", root / "model", root / "report")
    return root, frames, result


@pytest.fixture
def request_body():
    return {"request_id": "synthetic", "schema_version": VERSION,
            "context": {"cat0": 0, "cat1": 0, "numeric": {c: 0.0 for c in NUMERICAL}},
            "candidate_ids": [1, 2], "top_k": 2}


@pytest.fixture
def client(experiment):
    app = create_app(Settings(model_dir=experiment[0] / "model", redis_url=None))
    with TestClient(app) as client:
        yield client


class MemoryCache:
    def __init__(self, catalog):
        self.values = {catalog.key(i): json.dumps(catalog.payload(i)) for i in catalog.records}
        self.calls = 0

    def mget(self, keys):
        self.calls += 1
        return [self.values.get(k) for k in keys]

    def close(self):
        pass


def test_training_bundle_reloads_and_holds_out_test(experiment):
    root, frames, result = experiment
    assert result["test_evaluated"] is False
    assert result["reload_max_abs_error"] <= 1e-6
    bundle = load_bundle(root / "model")
    assert bundle.transformer.vocabularies["cat2"] == [1, 2, 3, 4]
    assert bundle.fallback["fitted_on"] == "train"
    assert "5" not in bundle.fallback["scores"]
    saved = pd.read_parquet(root / "model/validation_predictions.parquet")
    np.testing.assert_allclose(saved.pytorch_score, bundle.predict(frames["validation"]), atol=1e-6, rtol=1e-5)
    assert (root / "report/RESULTS.md").exists()


def test_api_uses_identical_tensors_and_scores_and_is_order_invariant(client, request_body, monkeypatch):
    import fairjob_ranker.serving as serving
    service = client.app.state.service
    raw = pd.DataFrame([{"product_id": i, **service.catalog.records[i], "cat0": 0, "cat1": 0,
                         **request_body["context"]["numeric"]} for i in request_body["candidate_ids"]])
    expected_tensors = to_tensors(service.bundle.transformer.transform(raw[FEATURES]))
    expected_scores = service.bundle.predict(raw)
    captured = []
    original = serving.to_tensors
    def capture(frame):
        tensors = original(frame)
        captured.append(tensors)
        return tensors
    monkeypatch.setattr(serving, "to_tensors", capture)
    service.catalog.client = MemoryCache(service.catalog)
    response = client.post("/rank", json=request_body)
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False and body["cache_hits"] == 2
    assert service.catalog.client.calls == 1  # One MGET per request, not one GET per ad.
    for expected, actual in zip(expected_tensors, captured[0]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    by_id = {r["product_id"]: r["score"] for r in body["rankings"]}
    np.testing.assert_allclose([by_id[i] for i in request_body["candidate_ids"]], expected_scores, atol=1e-6, rtol=1e-5)
    request_body["candidate_ids"].reverse()
    second = client.post("/rank", json=request_body).json()
    assert [r["product_id"] for r in body["rankings"]] == [r["product_id"] for r in second["rankings"]]
    np.testing.assert_allclose([r["score"] for r in body["rankings"]], [r["score"] for r in second["rankings"]], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(candidate_ids=[1, 1]),
    lambda r: r.update(candidate_ids=list(range(65))),
    lambda r: r.update(candidate_ids=[]),
    lambda r: r.update(candidate_ids=[True, 2]),
    lambda r: r.update(top_k=3),
    lambda r: r.update(top_k=0),
    lambda r: r.update(schema_version="wrong"),
    lambda r: r.update(click=1),
    lambda r: r["context"].update(cat0="0"),
    lambda r: r["context"].update(cat1=0.5),
    lambda r: r["context"].update(cat2=1),
    lambda r: r["context"]["numeric"].pop("num16"),
    lambda r: r["context"]["numeric"].update(num16=True),
    lambda r: r["context"]["numeric"].update(num16=float("nan")),
    lambda r: r["context"]["numeric"].update(num16=float("inf")),
    lambda r: r["context"]["numeric"].update(num16=10**400),
])
def test_invalid_requests_return_422(client, request_body, mutation):
    mutation(request_body)
    response = client.post("/rank", content=json.dumps(request_body), headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text


def test_unknown_ids_and_oov_are_different(client, request_body):
    request_body["candidate_ids"] = [5, 999]
    request_body["context"]["cat0"] = 999
    response = client.post("/rank", json=request_body).json()
    assert response["skipped_candidate_ids"] == [999]
    assert [r["product_id"] for r in response["rankings"]] == [5]
    assert response["oov_categories"] == 12
    request_body.update(candidate_ids=[999], top_k=1)
    assert client.post("/rank", json=request_body).status_code == 422


@pytest.mark.parametrize("failure", ["disconnected", "missing", "invalid", "wrong_length"])
def test_redis_failures_use_same_catalog_scores(client, request_body, failure):
    service = client.app.state.service
    first = client.post("/rank", json=request_body).json()["rankings"]
    cache = MemoryCache(service.catalog)
    if failure == "disconnected":
        def fail(_):
            raise redis.ConnectionError("test disconnect")
        cache.mget = fail
    elif failure == "missing":
        cache.values.clear()
    elif failure == "invalid":
        cache.values = {k: '{"catalog_version":"wrong"}' for k in cache.values}
    else:
        cache.mget = lambda _: []
    service.catalog.client = cache
    response = client.post("/rank", json=request_body).json()
    assert response["degraded"] is True
    assert response["rankings"] == first
    expected = {"missing": "cache_miss", "invalid": "cache_invalid"}.get(failure, "redis_unavailable")
    assert expected in response["fallback_reason"]


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_model_failure_falls_back_for_all_candidates(client, request_body, monkeypatch, failure):
    bundle = client.app.state.service.bundle
    def broken(c, n):
        if failure == "exception":
            raise RuntimeError("test inference failure")
        return torch.full((len(c),), float("nan"))
    monkeypatch.setattr(bundle.model, "forward", broken)
    request_body["candidate_ids"] = [1, 5]
    body = client.post("/rank", json=request_body).json()
    assert body["scoring_method"] == "smoothed_ad_ctr"
    assert body["model_version"] != body["attempted_model_version"]
    assert "model_inference_error" in body["fallback_reason"]
    for item in body["rankings"]:
        expected = bundle.fallback["scores"].get(str(item["product_id"]), bundle.fallback["train_ctr"])
        assert item["score"] == expected


def test_ties_use_product_id(client, request_body, monkeypatch):
    monkeypatch.setattr(client.app.state.service.bundle.model, "forward", lambda c, n: torch.zeros(len(c)))
    request_body["candidate_ids"] = [2, 1]
    body = client.post("/rank", json=request_body).json()
    assert [item["product_id"] for item in body["rankings"]] == [1, 2]


def test_individual_and_batched_candidate_inference_match(client, request_body):
    batch = client.post('/rank', json=request_body).json()['rankings']
    client.app.state.service.settings.batch_candidates = False
    individual = client.post('/rank', json=request_body).json()['rankings']
    assert [r['product_id'] for r in batch] == [r['product_id'] for r in individual]
    np.testing.assert_allclose([r['score'] for r in batch], [r['score'] for r in individual], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("failure", ["weights", "catalog_version", "feature_order"])
def test_invalid_bundle_is_unready(experiment, tmp_path, request_body, failure):
    directory = tmp_path / "model"
    shutil.copytree(experiment[0] / "model", directory)
    if failure == "weights":
        (directory / "weights.pt").write_bytes(b"broken")
    else:
        meta = read_json(directory / "metadata.json")
        if failure == "catalog_version":
            meta["catalog_version"] = "wrong"
        else:
            meta["feature_order"] = list(reversed(FEATURES))
        write_json(directory / "metadata.json", meta)
    with TestClient(create_app(Settings(model_dir=directory, redis_url=None))) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/rank", json=request_body).status_code == 503


def test_cancelled_http_request_does_not_release_running_cpu_capacity(experiment, request_body):
    async def scenario():
        app = create_app(Settings(model_dir=experiment[0] / "model", redis_url=None, max_inflight=1))
        async with app.router.lifespan_context(app):
            started, release = threading.Event(), threading.Event()
            original = app.state.service.rank
            def blocking(request):
                started.set()
                assert release.wait(timeout=5)
                return original(request)
            app.state.service.rank = blocking
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/rank", json=request_body))
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    assert (await client.get("/healthz")).status_code == 200
                    assert (await client.post("/rank", json=request_body)).status_code == 503
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    assert (await client.post("/rank", json=request_body)).status_code == 503
                finally:
                    release.set()
                for _ in range(100):
                    if not app.state.tokens.empty():
                        break
                    await asyncio.sleep(0.01)
                assert (await client.post("/rank", json=request_body)).status_code == 200
                assert (await client.get("/metrics")).json()["rejected_overload"] == 2
    asyncio.run(scenario())
