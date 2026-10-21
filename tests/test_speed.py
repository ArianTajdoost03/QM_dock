import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from tblite.interface import Calculator

from qmdock.config import BOHR_TO_ANG
from qmdock.qm.engine import Evaluator, relax, solve
from qmdock.qm.pool import job_cost

WATERS = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
                   [3.0, 0.1, 0.0], [3.96, 0.1, 0.0], [2.76, 1.03, 0.0]])
NUMBERS = np.array([8, 1, 1, 8, 1, 1])


def displaced(seed, amount=0.15):
    return WATERS + np.random.default_rng(seed).normal(scale=amount, size=WATERS.shape)


def test_reusing_the_previous_wavefunction_gives_the_same_energy_and_gradient_as_a_fresh_calculation():
    evaluate = Evaluator(NUMBERS, 0, 0, 1.0, ("alpb", "water"))
    evaluate(WATERS)
    for seed in (1, 2, 3):
        pos = displaced(seed)
        e, g, _ = evaluate(pos)
        e_ref, g_ref, _ = solve(NUMBERS, pos, 0, 0, gradient=True, solv=("alpb", "water"))
        assert abs(e - e_ref) < 1e-8 and np.abs(g - g_ref).max() < 2e-5


def test_a_failed_restart_falls_back_to_a_fresh_calculation(monkeypatch):
    evaluate = Evaluator(NUMBERS, 0, 0, 1.0, None)
    evaluate(WATERS)
    original = Calculator.singlepoint
    calls = []

    def flaky(self, res=None, copy=False):
        calls.append(res is not None)
        if len(calls) == 1:
            raise RuntimeError("SCF did not converge")
        return original(self, res, copy)

    monkeypatch.setattr(Calculator, "singlepoint", flaky)
    pos = displaced(4)
    e, g, _ = evaluate(pos)
    assert abs(e - solve(NUMBERS, pos, 0, 0, gradient=True)[0]) < 1e-7
    assert evaluate.calc is None or evaluate.res is not None


def test_relaxation_with_reuse_reaches_the_same_minimum_as_independent_calculations():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    pos = mol.GetConformer().GetPositions()
    out, e, nit, _ = relax(numbers, pos, np.arange(len(numbers)), 0, 0, 100, 0.0005)

    def fresh(x):
        energy, gradient, _ = solve(numbers, x.reshape(-1, 3), 0, 0, gradient=True)
        return energy, gradient.ravel() / BOHR_TO_ANG

    reference = minimize(fresh, pos.ravel(), jac=True, method="L-BFGS-B",
                         options={"maxiter": 100, "gtol": 0.0005 / BOHR_TO_ANG, "maxcor": 10})
    assert nit < 100 and abs(e - reference.fun) < 1e-6
    assert np.abs(out - reference.x.reshape(-1, 3)).max() < 0.02


def test_the_biggest_jobs_are_scheduled_first():
    jobs = [{"kind": "energy", "numbers": np.zeros(n)} for n in (50, 400, 120, 400)]
    jobs.append({"kind": "relax", "numbers": np.zeros(150), "steps": 30})
    ordered = sorted(jobs, key=job_cost, reverse=True)
    assert [j["kind"] for j in ordered[:1]] == ["relax"]
    assert [len(j["numbers"]) for j in ordered[1:]] == [400, 400, 120, 50]
