import numpy as np
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")
from qmdock.chem.ligand import rotatable_bonds
from qmdock.chem.protonation import embed
from qmdock.config import Config
from qmdock.pipeline.joint import heavy_mappings, transfer_exact
from qmdock.pipeline.ligandprep import LigandData
from qmdock.search.poses import PoseSet
from qmdock.search.sampling import search
from qmdock.search.scoring import Scorer
from qmdock.search.torsions import TorsionModel


def molecule(smiles, seed=1):
    mol = embed(Chem.MolFromSmiles(smiles), seed)
    return mol, mol.GetConformer().GetPositions()


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return np.arctan2(np.cross(b1, v) @ w, v @ w)


def test_rotor_detection():
    assert len(rotatable_bonds(molecule("CCCCO")[0], 12)) == 2
    assert rotatable_bonds(molecule("CC(=O)NC")[0], 12) == []
    assert rotatable_bonds(molecule("c1ccccc1")[0], 12) == []
    assert len(rotatable_bonds(molecule("CCCCCC")[0], 2)) == 2


def test_rotation_keeps_bonds_and_sets_the_torsion():
    mol, xyz = molecule("CCCCO")
    rotors = rotatable_bonds(mol, 12)
    model = TorsionModel(rotors, mol, torch.device("cpu"))
    X = torch.tensor(xyz[None], dtype=torch.float32)
    delta = torch.zeros(1, model.K)
    delta[0, 1] = 1.0
    out = model.apply(X, delta)[0].numpy()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        assert abs(np.linalg.norm(out[i] - out[j]) - np.linalg.norm(xyz[i] - xyz[j])) < 1e-4
    a, b, _ = rotors[1]
    x = next(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b and n.GetAtomicNum() > 1)
    y = next(n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a and n.GetAtomicNum() > 1)
    change = dihedral(out[x], out[a], out[b], out[y]) - dihedral(xyz[x], xyz[a], xyz[b], xyz[y])
    assert abs(np.cos(change) - np.cos(1.0)) < 1e-3
    assert np.allclose(model.apply(X, torch.zeros(1, model.K))[0].numpy(), xyz, atol=1e-5)


def test_gradient_reaches_the_torsions_and_extended_chain_is_clash_free():
    mol, xyz = molecule("CCCCCC")
    model = TorsionModel(rotatable_bonds(mol, 12), mol, torch.device("cpu"))
    delta = torch.zeros(1, model.K, requires_grad=True)
    X = model.apply(torch.tensor(xyz[None], dtype=torch.float32), delta)
    energy, hard = model.intra(X)
    (energy.sum() + X[:, 0, 0].sum()).backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
    assert not bool(hard.any())


def test_search_with_torsions_is_clash_free_and_keeps_bond_lengths():
    mol, xyz = molecule("CCCCO")
    grid = np.array([[x, y, 0.0] for x in np.arange(-6, 6.1, 1.5) for y in np.arange(-6, 6.1, 1.5)])
    cfg = Config("r", "l", n_random=1500, n_optimize=60, opt_steps=20, n_qm=5, search_radius=4.0, torsions=True)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    scorer = Scorer(np.array(["C"] * len(grid)), grid, elements, torch.device("cpu"), cfg)
    model = TorsionModel(rotatable_bonds(mol, 12), mol, torch.device("cpu"))
    heavy = np.array([e != "H" for e in elements])
    ps = search(scorer, xyz[None], heavy, np.array([0.0, 0.0, 4.0]), cfg, 1, 0, model)
    assert len(ps) > 0 and ps.diag["torsions"] == 2
    assert not scorer.evaluate(torch.tensor(ps.coords, dtype=torch.float32), full=True)[1].any()
    bond = mol.GetBonds()[1]
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    ref = np.linalg.norm(xyz[i] - xyz[j])
    assert np.allclose(np.linalg.norm(ps.coords[:, i] - ps.coords[:, j], axis=1), ref, atol=1e-3)


def test_exact_transfer_rebuilds_hydrogens_for_another_state():
    template, tx = molecule("O=c1cccccc1O", 1)
    state, _ = molecule("[O-]c1cccccc1=O", 5)

    def data(m):
        return LigandData(m, [a.GetSymbol() for a in m.GetAtoms()], None,
                          np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()]), 0, None, 1.0)

    pose = PoseSet(tx[None] + np.array([5.0, 0, 0]), np.zeros(1, int), np.zeros(1), np.zeros(1, int))
    out = transfer_exact(data(template), data(state), pose, heavy_mappings(template, state)[0])[0]
    assert out.shape == (state.GetNumAtoms(), 3) and (np.abs(out).sum(1) > 0).all()
