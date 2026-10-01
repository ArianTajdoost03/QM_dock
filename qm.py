import os

os.environ["OMP_NUM_THREADS"] = os.environ.get("QMDOCK_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_STACKSIZE", "128M")

import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize
from tblite.interface import Calculator

from config import BOHR_TO_ANG, HARTREE_TO_KCAL

LADDER = (
    {"max-iter": 250},
    {"mixer-damping": 0.3, "max-iter": 200},
    {"mixer-damping": 0.1, "max-iter": 200, "guess": 0},
)


PROBE = (
    "import sys, numpy as np; from tblite.interface import Calculator;"
    "c = Calculator('GFN2-xTB', np.array([8,1,1]), np.array([[0,0,0],[1.8,0,0],[-.5,1.7,0]]), charge=0.0, uhf=0);"
    "c.set('verbosity', 0);"
    "len(sys.argv) > 2 and c.add(sys.argv[1] + '-solvation', sys.argv[2]);"
    "print(c.singlepoint().get('energy'))"
)


def preflight(solvent=None, model="alpb"):
    if solvent:
        r = subprocess.run([sys.executable, "-c", PROBE, model, solvent], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise ValueError(f"solvent '{solvent}' is not available for the {model} model in tblite: "
                             f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(
            "tblite crashes on a water single point in this environment (exit code "
            f"{r.returncode}). This is an installation problem, usually MKL/OpenMP conflicts with other "
            "QM packages. Create a clean environment: conda env create -f environment.yml -n qmdock")


def init_worker():
    try:
        import resource

        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


class ConvergenceError(RuntimeError):
    pass


def check_parity(numbers, charge, uhf):
    electrons = int(np.sum(numbers)) - int(charge)
    if (electrons - int(uhf)) % 2:
        raise ValueError(f"{electrons} electrons is incompatible with uhf={uhf}")


def solve(numbers, pos_ang, charge, uhf, gradient=False, accuracy=1.0, solv=None):
    numbers = np.asarray(numbers, dtype=np.int32)
    check_parity(numbers, charge, uhf)
    last = None
    for attempt, opts in enumerate(LADDER):
        try:
            calc = Calculator("GFN2-xTB", numbers, np.asarray(pos_ang) / BOHR_TO_ANG,
                              charge=float(charge), uhf=int(uhf))
            calc.set("verbosity", 0)
            if accuracy != 1.0:
                calc.set("accuracy", float(accuracy))
            for key, value in opts.items():
                calc.set(key, value)
            if solv:
                calc.add(f"{solv[0]}-solvation", solv[1])
            res = calc.singlepoint()
            grad = np.array(res.get("gradient")) if gradient else None
            return float(res.get("energy")), grad, attempt
        except RuntimeError as exc:
            last = exc
    raise ConvergenceError(f"SCF failed after {len(LADDER)} attempts: {last}")


def relax(numbers, pos_ang, movable, charge, uhf, max_steps, fmax, accuracy=1.0, solv=None):
    pos = np.array(pos_ang, dtype=float)
    movable = np.asarray(movable)
    worst = [0]

    def fun(x):
        p = pos.copy()
        p[movable] = x.reshape(-1, 3)
        e, g, a = solve(numbers, p, charge, uhf, gradient=True, accuracy=accuracy, solv=solv)
        worst[0] = max(worst[0], a)
        return e, g[movable].ravel() / BOHR_TO_ANG

    res = minimize(fun, pos[movable].ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_steps, "gtol": fmax / BOHR_TO_ANG, "maxcor": 10})
    out = pos.copy()
    out[movable] = res.x.reshape(-1, 3)
    return out, float(res.fun), int(res.nit), worst[0]


def run_job(job):
    t0 = time.time()
    solv = (job.get("model", "alpb"), job["solvent"]) if job.get("solvent") else None
    try:
        if job["kind"] == "energy":
            e, _, a = solve(job["numbers"], job["pos"], job["charge"], job["uhf"],
                            accuracy=job["accuracy"], solv=solv)
            out = {"energy": e, "attempt": a}
        else:
            pos, e, steps, a = relax(job["numbers"], job["pos"], job["movable"], job["charge"],
                                     job["uhf"], job["steps"], job["fmax"], job["accuracy"], solv)
            out = {"energy": e, "pos": pos, "steps": steps, "attempt": a}
            if "lig_slice" in job:
                lo, hi = job["lig_slice"]
                e_lig, _, _ = solve(job["numbers"][lo:hi], pos[lo:hi], job["lig_charge"],
                                    job["uhf"], accuracy=job["accuracy"], solv=solv)
                out["lig_energy"] = e_lig
        out["ok"] = True
    except (RuntimeError, ValueError) as exc:
        out = {"ok": False, "error": str(exc)[:200]}
    out["key"] = job["key"]
    out["seconds"] = time.time() - t0
    return out


def to_kcal(hartree):
    return hartree * HARTREE_TO_KCAL
