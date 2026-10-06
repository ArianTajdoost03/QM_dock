import json
import os

import numpy as np
import pytest
from rdkit import Chem

from qmdock.tools import dft_tools
from qmdock.chem.protonation import embed, ligand_states

pytest.importorskip("tblite")
from qmdock.qm.engine import preflight, solve

Z = np.array([8, 1, 1, 8, 1, 1])
P = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [2.9, 0, 0], [3.3, 0.8, 0.3], [3.3, -0.8, -0.3]])


def test_solvation_lowers_polar_energy():
    gas = solve(Z, P, 0, 0)[0]
    wet = solve(Z, P, 0, 0, solv=("alpb", "water"))[0]
    assert wet < gas


def test_unknown_solvent_is_reported():
    with pytest.raises(ValueError):
        preflight("not-a-solvent", "alpb")


def test_protonation_states_for_tropolone():
    mol = embed(Chem.MolFromSmiles("O=c1cccccc1O"), 1)
    states = ligand_states(mol, "all", 6.4, 8.4, 4, 1)
    assert sorted(s["charge"] for s in states) == [-1, 0]


def test_dft_tools_roundtrip(tmp_path):
    d = tmp_path / "pose_3"
    d.mkdir()
    for part in ("complex", "pocket", "ligand"):
        (d / f"{part}.xyz").write_text("1\nx\nH 0.0 0.0 0.0\n")
    (d / "manifest.json").write_text(json.dumps({"pose": 3, "charge_complex": -1, "charge_pocket": 0,
                                                 "charge_ligand": -1, "xtb_e_final_kcal": -5.0}))
    made = dft_tools.make_inputs(str(tmp_path), "r2SCAN-3c", 2, 1, 1, 1)
    assert len(made) == 3 and "* xyz -1 1" in open(d / "complex.inp").read()
    for part, e in (("complex", -10.0), ("pocket", -6.0), ("ligand", -3.99)):
        (d / f"{part}.out").write_text(f"FINAL SINGLE POINT ENERGY      {e:.8f}\n")
    rows = dft_tools.collect(str(tmp_path))
    assert abs(rows[0]["dft_e_int_kcal"] - (-0.01 * 627.509474)) < 1e-3
