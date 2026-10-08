import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

from qmdock.qm.engine import init_worker, run_job
from qmdock.ui import bar, log


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
        with bar(len(jobs), label) as progress:
            def record(r):
                results[r["key"]] = r
                progress.update(1)

            if self.pool is None:
                for job in jobs:
                    record(self.fn(job))
            else:
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
        log(f"  {label}: {len(jobs)} jobs in {time.time() - t0:.0f}s" + (f", {crashed} crashed the QM engine" if crashed else ""))
        return results
