import sys

import numpy as np

from qmdock.qm.pool import Runner

Z = np.array([8, 1, 1, 8, 1, 1], dtype=np.int32)
POS = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                [2.9, 0, 0], [3.3, 0.8, 0.3], [3.3, -0.8, -0.3]])


def job(i):
    return {"kind": "energy", "key": ("w", i), "numbers": Z, "pos": POS + 0.01 * i, "charge": 0,
            "uhf": 0, "accuracy": 1.0}


def test_pool_matches_serial():
    jobs = [job(i) for i in range(4)]
    with Runner(1) as serial:
        a = serial.map(jobs, "serial")
    with Runner(2) as pool:
        b = pool.map(jobs, "pool")
    assert all(b[k]["ok"] for k in b)
    assert all(abs(a[k]["energy"] - b[k]["energy"]) < 1e-8 for k in a)


def test_entrypoint_does_not_import_torch():
    import subprocess

    code = "import qmdock.cli, sys; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def crash_fn(job):
    import os

    if job["key"][1] == 1:
        os._exit(1)
    from qmdock.qm.engine import run_job

    return run_job(job)


def test_crash_is_isolated():
    jobs = [job(i) for i in range(3)]
    with Runner(2, fn=crash_fn) as pool:
        res = pool.map(jobs, "crash")
    assert res[("w", 1)]["ok"] is False and "crashed" in res[("w", 1)]["error"]
    assert res[("w", 0)]["ok"] and res[("w", 2)]["ok"]
