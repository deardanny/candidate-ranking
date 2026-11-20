"""Open-loop HTTP load: scheduled latency includes generator lag and queueing."""
import asyncio
from collections import Counter
import math
import os
from pathlib import Path
import platform
from time import perf_counter

import httpx
import numpy as np
import psutil


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": len(values), **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
            "max": float(max(values))}


class Resources:
    def __init__(self, pids: list[int], include_children: bool = True):
        self.pids, self.samples, self.previous = pids, [], {}
        self.include_children, self.sampling_errors = include_children, 0

    def sample(self):
        now = perf_counter()
        processes = {}
        for pid in self.pids:
            try:
                parent = psutil.Process(pid)
                processes[parent.pid] = parent
                for p in parent.children(recursive=True) if self.include_children else []:
                    processes[p.pid] = p
            except (psutil.Error, OSError):
                self.sampling_errors += 1
                continue
        cpu, rss = 0.0, 0
        for pid, process in processes.items():
            try:
                times = process.cpu_times()
                total = times.user + times.system
                if pid in self.previous:
                    old_time, old_cpu = self.previous[pid]
                    cpu += 100 * max(0, total - old_cpu) / (now - old_time)
                self.previous[pid] = (now, total)
                rss += process.memory_info().rss
            except (psutil.Error, OSError):
                self.sampling_errors += 1
                continue
        self.samples.append({"seconds": now, "cpu_percent_one_core_100": cpu, "rss_mib": rss / 2**20})

    def summary(self):
        return {"sample_count": len(self.samples), "sampling_errors": self.sampling_errors,
                "cpu_percent_one_core_100": percentiles([s["cpu_percent_one_core_100"] for s in self.samples[1:]]),
                "rss_mib": percentiles([s["rss_mib"] for s in self.samples])}


async def run_phase(url: str, requests: list[dict], rate: float, duration: float,
                    timeout: float = 5, max_inflight: int = 256, pids: list[int] | None = None,
                    transport=None, start_index: int = 0) -> dict:
    if rate <= 0 or duration <= 0 or max_inflight < 1 or not requests:
        raise ValueError("Positive rate/duration/capacity and nonempty requests required.")
    count = math.ceil(rate * duration)
    records, pending = [], set()
    server, generator = Resources(pids or []), Resources([os.getpid()], include_children=False)
    stop = asyncio.Event()

    async def monitor():
        next_progress = perf_counter() + 30
        while True:
            await asyncio.to_thread(server.sample)
            await asyncio.to_thread(generator.sample)
            if duration >= 60 and perf_counter() >= next_progress:
                statuses = Counter(r['status'] for r in records)
                print(f"progress: completed={len(records)}, statuses={dict(statuses)}", flush=True)
                next_progress += 30
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
                return
            except asyncio.TimeoutError:
                pass

    limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=max_inflight)
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits, trust_env=False, transport=transport) as client:
        async def send(index, due):
            sent = perf_counter()
            body = requests[(start_index + index) % len(requests)]
            item = {"index": index, "issued": True, "status": "network_error", "degraded": False,
                    "send_lag_ms": max(0, sent - due) * 1000, "candidates": len(body["candidate_ids"])}
            try:
                response = await client.post("/rank", json=body)
                item["status"] = str(response.status_code)
                if response.status_code == 200:
                    data = response.json()
                    if data["request_id"] != body["request_id"] or not data["rankings"]:
                        raise ValueError("Invalid ranking response")
                    item.update(degraded=data["degraded"], timings_ms=data["timings_ms"],
                                scoring_method=data["scoring_method"])
            except httpx.TimeoutException:
                item["status"] = "timeout"
            except httpx.HTTPError:
                item["status"] = "network_error"
            except (ValueError, KeyError, TypeError):
                item["status"] = "invalid_response"
            end = perf_counter()
            item.update(latency_ms=(end - due) * 1000, http_ms=(end - sent) * 1000)
            records.append(item)

        monitor_task = asyncio.create_task(monitor())
        began = perf_counter()
        try:
            for i in range(count):
                due = began + i / rate
                await asyncio.sleep(max(0, due - perf_counter()))
                if len(pending) >= max_inflight:
                    records.append({"index": i, "issued": False, "status": "client_capacity",
                                    "send_lag_ms": max(0, perf_counter() - due) * 1000})
                    continue
                task = asyncio.create_task(send(i, due))
                pending.add(task)
                task.add_done_callback(pending.discard)
            await asyncio.sleep(max(0, began + duration - perf_counter()))
            if pending:
                await asyncio.gather(*pending)
        finally:
            stop.set()
            await monitor_task
        elapsed = perf_counter() - began
    records.sort(key=lambda r: r["index"])
    statuses = dict(Counter(r["status"] for r in records))
    issued = [r for r in records if r["issued"]]
    success = [r for r in issued if r["status"] == "200"]
    failures = len(records) - len(success)
    timings = {key: percentiles([r["timings_ms"][key] for r in success if key in r["timings_ms"]])
               for key in ["lookup", "transform", "inference", "sort", "service", "handler"]}
    return {"target_rps": rate, "duration_seconds": duration, "elapsed_including_drain_seconds": elapsed,
            "scheduled": count, "issued": len(issued), "completed": len(issued), "successful": len(success),
            "generator_dropped": statuses.get("client_capacity", 0), "status_counts": statuses,
            "successful_qps": len(success) / elapsed, "issued_qps": len(issued) / duration,
            "failure_rate": failures / count, "degraded_rate": sum(r.get("degraded", False) for r in success) / len(success) if success else None,
            "latency_success_ms": percentiles([r["latency_ms"] for r in success]),
            "latency_all_issued_ms": percentiles([r["latency_ms"] for r in issued]),
            "http_success_ms": percentiles([r["http_ms"] for r in success]),
            "generator_lag_ms": percentiles([r["send_lag_ms"] for r in records]),
            "stages_ms": timings, "server_resources": server.summary(), "generator_resources": generator.summary(),
            "records": records}


def environment() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "load_generator_same_host": True, "runtime": "native macOS processes", "docker_resource_limits": None}
