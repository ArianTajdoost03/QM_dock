import threading
import time
from types import SimpleNamespace

from qmdock.config import Config
from qmdock.pipeline.run import run_states
from qmdock.qm.pool import Runner
from qmdock.ui import log


def sleep_job(job):
    time.sleep(job["sleep"])
    return {"key": job["key"], "ok": True, "seconds": job["sleep"]}


def jobs(tag, n, sleep):
    return [{"key": (tag, i), "kind": "energy", "numbers": [6] * 10, "sleep": sleep} for i in range(n)]


def test_two_threads_share_one_pool_and_fill_idle_workers():
    with Runner(4, fn=sleep_job) as runner:
        runner.map(jobs("warm", 4, 0.0), "warm")
        out = {}

        def go(tag, n):
            out[tag] = runner.map(jobs(tag, n, 1.0), tag)

        t0 = time.time()
        threads = [threading.Thread(target=go, args=(tag, 2)) for tag in ("a", "b")]
        [t.start() for t in threads]
        [t.join() for t in threads]
        elapsed = time.time() - t0
    assert set(out["a"]) == {("a", 0), ("a", 1)}
    assert set(out["b"]) == {("b", 0), ("b", 1)}
    assert elapsed < 1.8


def test_run_states_keeps_order_and_runs_concurrently():
    seen = []
    runner = SimpleNamespace(workers=4)

    def work(i, st):
        log(f"state body {i}")
        seen.append(threading.get_ident())
        time.sleep(0.5)
        return {"state": i}

    cfg = Config("r", "l")
    t0 = time.time()
    rows = run_states(cfg, [{}, {}, {}], runner, work)
    assert [r["state"] for r in rows] == [0, 1, 2]
    assert time.time() - t0 < 1.2
    assert len(set(seen)) == 3


def test_run_states_sequential_when_disabled():
    runner = SimpleNamespace(workers=4)
    order = []

    def work(i, st):
        order.append(i)
        return {"state": i}

    cfg = Config("r", "l", parallel_states=False)
    rows = run_states(cfg, [{}, {}], runner, work)
    assert order == [0, 1] and [r["state"] for r in rows] == [0, 1]
