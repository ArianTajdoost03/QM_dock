import numpy as np
import pytest

pytest.importorskip("tblite")
from qmdock.qm.engine import check_parity, relax, solve

WATER2 = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                   [2.9, 0, 0], [3.3, 0.8, 0.3], [3.3, -0.8, -0.3]])
Z = np.array([8, 1, 1, 8, 1, 1])


def test_binding_is_attractive():
    e_dimer = solve(Z, WATER2, 0, 0)[0]
    e_a = solve(Z[:3], WATER2[:3], 0, 0)[0]
    e_b = solve(Z[3:], WATER2[3:], 0, 0)[0]
    assert -15 < (e_dimer - e_a - e_b) * 627.5 < 0


def test_parity_is_checked():
    with pytest.raises(ValueError):
        check_parity(Z[:3], 1, 0)
    with pytest.raises(ValueError):
        solve(Z[:3], WATER2[:3], 1, 0)


def test_relax_lowers_energy():
    e0 = solve(Z, WATER2, 0, 0)[0]
    _, e1, steps, _ = relax(Z, WATER2, np.arange(3, 6), 0, 0, 20, 0.001)
    assert e1 <= e0 + 1e-9 and steps > 0
