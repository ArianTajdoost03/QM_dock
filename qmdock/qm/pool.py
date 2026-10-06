import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

from qmdock.qm.engine import init_worker, run_job


def log(msg):
    print(msg, flush=True)


def default_workers():
    cores = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 1e6
        return max(1, min(cores, int(avail // 1.5)))
    except (OSError, StopIteration):
        return cores


class Runner:
    def __init__(self, workers, fn=run_job):
        self.workers = workers
        self.fn = fn
        self.pool = None

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
        t0 = time.time()
        results = {}
        step = 1 if len(jobs) <= 100 else max(1, len(jobs) // 20)

        def record(r):
            results[r["key"]] = r
            if len(results) % step == 0 or len(results) == len(jobs):
                log(f"  {label}: {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s, last job {r['seconds']:.0f}s)")

        if self.pool is None:
            for job in jobs:
                record(self.fn(job))
            return results
        try:
            for r in self.pool.map(self.fn, jobs, chunksize=1):
                record(r)
        except BrokenProcessPool:
            log("  a worker process died (segfault or memory kill); rerunning unfinished jobs one per process")
            self.pool.shutdown(cancel_futures=True)
            self.pool = self._new_pool(self.workers)
            for job in [j for j in jobs if j["key"] not in results]:
                record(self.isolated(job))
        crashed = sum(1 for r in results.values() if not r["ok"] and "crashed" in r.get("error", ""))
        if crashed:
            log(f"  {label}: {crashed} job(s) crashed the QM engine")
        return results
