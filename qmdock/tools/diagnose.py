import importlib.metadata as md
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WATER = """
import numpy as np
from qmdock.qm.engine import solve
Z = np.array([8,1,1,8,1,1]); P = np.array([[0,0,0],[0.96,0,0],[-0.24,0.93,0],[2.9,0,0],[3.3,0.8,0.3],[3.3,-0.8,-0.3]])
"""
LIGAND = """
import numpy as np, sys
from rdkit import Chem
from qmdock.chem.elements import ATOMIC_NUMBER
from qmdock.chem.ligand import load_ligand
from qmdock.qm.engine import solve, relax
m = load_ligand(sys.argv[1]); Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()]); P = m.GetConformer().GetPositions()
"""
STEPS = [
    ("water single point", WATER + "print(solve(Z, P, 0, 0)[0])"),
    ("water gradient", WATER + "print(solve(Z, P, 0, 0, gradient=True)[1].shape)"),
    ("ligand single point", LIGAND + "print(solve(Z, P, 0, 0)[0])"),
    ("ligand gradient", LIGAND + "print(solve(Z, P, 0, 0, gradient=True)[1].shape)"),
    ("ligand relax 3 steps", LIGAND + "print(relax(Z, P, np.arange(len(Z)), 0, 0, 3, 0.002)[1])"),
]


def run(code, env, ligand):
    r = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code, ligand], cwd=HERE, env=env,
                       capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode, tail, r.stderr.strip().splitlines()[-6:]


def main():
    ligand = sys.argv[1] if len(sys.argv) > 1 else "ligand.sdf"
    print("python", sys.version.split()[0])
    for pkg in ("tblite", "numpy", "scipy", "rdkit", "torch", "pyscf", "xtb"):
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    for k in ("OMP_NUM_THREADS", "OMP_STACKSIZE", "MKL_NUM_THREADS", "LD_LIBRARY_PATH", "CONDA_DEFAULT_ENV"):
        print(f"{k}={os.environ.get(k)}")
    probe = subprocess.run([sys.executable, "-c", "import tblite.interface, os;"
                            "print([l.split()[-1] for l in open('/proc/self/maps') if any(t in l for t in ('omp','blas','mkl','lapack'))][:0] or "
                            "sorted({l.split()[-1] for l in open('/proc/self/maps') if any(t in l for t in ('omp','blas','mkl','lapack'))}))"],
                           capture_output=True, text=True)
    print("OpenMP/BLAS libraries loaded with tblite:\n ", "\n  ".join(eval(probe.stdout or "[]")))
    for label, threads in (("1 thread", "1"), ("default threads", None)):
        env = dict(os.environ)
        env["QMDOCK_THREADS"] = threads or str(os.cpu_count())
        print(f"\n== {label} ==")
        for name, code in STEPS:
            rc, out, err = run(code, env, ligand)
            status = "ok" if rc == 0 else ("SEGFAULT" if rc in (-11, 139) else f"exit {rc}")
            print(f"{name:24s} {status:9s} {out}")
            if rc != 0:
                print("   ", "\n    ".join(err))
                break


if __name__ == "__main__":
    main()
