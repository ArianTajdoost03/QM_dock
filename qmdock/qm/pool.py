import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

from qmdock.config import COST_EXPONENT
from qmdock.qm.engine import init_worker, run_job
from qmdock.ui import bar, log


def job_cost(job):
    steps = job.get("steps", 1) if job["kind"] == "relax" else 1
    return len(job["numbers"]) ** COST_EXPONENT * steps


def available_gb():
    try:
        with open("/proc/meminfo") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 1e6
    except (OSError, StopIteration):
        return None


def default_workers():
    cores = os.cpu_count() or 1
    avail = available_gb()
    if avail is None:
        return cores
    return max(1, min(cores, int(avail // 1.5)))


class Runner:
    def __init__(self, workers, fn=run_job):
        self.workers = workers
        self.fn = fn
        self.pool = None
        self.lock = threading.Lock()

    def _new_pool(self, n):
        return ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=init_worker)

    def __enter__(self):
        if self.workers > 1:
            self.pool = self._new_pool(self.workers)
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.shutdown(cancel_futures=True)

    def isolated(self, job):
        with self._new_pool(1) as ex:
            try:
                return ex.submit(self.fn, job).result()
            except BrokenProcessPool:
                return {"ok": False, "error": "worker process crashed (segfault or killed)",
                        "key": job["key"], "seconds": 0.0}

    def map(self, jobs, label):
        if not jobs:
            return {}
        jobs = sorted(jobs, key=job_cost, reverse=True)
        t0 = time.time()
        results = {}
        with bar(len(jobs), label) as progress:
            def record(r):
                results[r["key"]] = r
                progress.update(1)

            if self.pool is None:
                for job in jobs:
                    record(self.fn(job))
            else:
                current = self.pool
                try:
                    for r in current.map(self.fn, jobs, chunksize=1):
                        record(r)
                except BrokenProcessPool:
                    log("  a worker process died (segfault or memory kill); rerunning unfinished jobs one per process")
                    with self.lock:
                        if self.pool is current:
                            current.shutdown(cancel_futures=True)
                            self.pool = self._new_pool(self.workers)
                    for job in [j for j in jobs if j["key"] not in results]:
                        record(self.isolated(job))
        crashed = sum(1 for r in results.values() if not r["ok"] and "crashed" in r.get("error", ""))
        wall = time.time() - t0
        spent = [r.get("seconds", 0.0) for r in results.values()]
        detail = ""
        if self.pool is not None and len(jobs) > 1 and wall > 5 and sum(spent) > 0:
            detail = f" (longest job {max(spent):.0f}s, worker use {sum(spent) / (self.workers * wall):.0%})"
        log(f"  {label}: {len(jobs)} jobs in {wall:.0f}s{detail}" + (f", {crashed} crashed the QM engine" if crashed else ""))
        return results
