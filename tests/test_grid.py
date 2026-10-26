import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rdkit import Chem
from rdkit.Chem import AllChem

from qmdock.chem.atomtypes import ReceptorTypes, ligand_types
from qmdock.config import Config
from qmdock.search.sampling import place, random_rotations, search
from qmdock.search.scoring import Scorer

CENTER = np.array([0.0, 0.0, 0.0])


def environment(seed=0, n=400):
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=9.0, size=(n * 3, 3))
    keep = [p for p in pts if np.linalg.norm(p) > 5.5][:n]
    xyz = np.array(keep)
    elements = rng.choice(["C", "N", "O", "S"], size=len(xyz), p=[0.6, 0.15, 0.2, 0.05])
    elements[0] = "Zn"
    xyz[0] = [3.0, 0.0, 0.0]
    types = ReceptorTypes(donor=rng.random(len(xyz)) < 0.2, acceptor=rng.random(len(xyz)) < 0.3,
                          hydrophobic=rng.random(len(xyz)) < 0.3,
                          charge=np.where(rng.random(len(xyz)) < 0.1, rng.choice([-1.0, 1.0], len(xyz)), 0.0),
                          metal_free=np.where(elements == "Zn", 2.0, 0.0))
    return elements, xyz, types


def ligand():
    mol = Chem.AddHs(Chem.MolFromSmiles("NS(=O)(=O)c1ccc(CC(=O)[O-])cc1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    return mol


def make(elec=0.0):
    elements, xyz, types = environment()
    mol = ligand()
    cfg = Config("r", "l", elec_weight=elec, search_radius=5.0, n_random=1500, n_optimize=60, opt_steps=40, n_qm=5)
    scorer = Scorer(elements, xyz, [a.GetSymbol() for a in mol.GetAtoms()], torch.device("cpu"), cfg,
                    ligand_types(mol), types)
    pos = mol.GetConformer().GetPositions()
    return scorer, cfg, mol, pos - pos.mean(0)


@pytest.mark.parametrize("elec", [0.0, 0.05])
def test_grid_energy_matches_exact_energy(elec):
    scorer, cfg, mol, pos = make(elec)
    scorer.attach_grid(CENTER, 14.0, cfg)
    gen = torch.Generator().manual_seed(1)
    b = 600
    R = random_rotations(b, gen, torch.device("cpu"))
    t = torch.randn(b, 3, generator=gen) * 1.5
    X = place(torch.tensor(pos[None], dtype=torch.float32), torch.zeros(b, dtype=torch.long), R, t)
    with torch.no_grad():
        exact = scorer.evaluate(X, full=False)[0]
        approx = scorer.energy(X)
    sel = exact < 3.0
    assert sel.sum() > 20
    assert (exact[sel] - approx[sel]).abs().max() < 0.4
    assert torch.corrcoef(torch.stack([exact[sel], approx[sel]]))[0, 1] > 0.99


def test_grid_gradient_points_the_same_way_as_the_exact_gradient():
    scorer, cfg, mol, pos = make()
    scorer.attach_grid(CENTER, 14.0, cfg)
    C = torch.tensor(pos[None], dtype=torch.float32)
    gen = torch.Generator().manual_seed(2)
    R = random_rotations(200, gen, torch.device("cpu"))
    w = (torch.randn(200, 3, generator=gen) * 1.5).requires_grad_(True)
    zero = torch.zeros(200, dtype=torch.long)
    exact = torch.autograd.grad(scorer.evaluate(place(C, zero, R, w), full=False)[0].clamp(max=30).sum(), w)[0]
    w2 = w.detach().clone().requires_grad_(True)
    approx = torch.autograd.grad(scorer.energy(place(C, zero, R, w2)).clamp(max=30).sum(), w2)[0]
    keep = exact.norm(dim=1) > 0.05
    cosine = (exact[keep] * approx[keep]).sum(1) / (exact[keep].norm(dim=1) * approx[keep].norm(dim=1))
    assert cosine.mean() > 0.9


def test_search_with_the_grid_returns_clash_free_poses_scored_exactly():
    scorer, cfg, mol, pos = make()
    conf = pos[None]
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    ps = search(scorer, conf, heavy, CENTER, cfg, 1, 0)
    assert len(ps) > 0
    X = torch.tensor(ps.coords, dtype=torch.float32)
    exact, clash = scorer.evaluate(X, full=True)
    assert not clash.any()
    assert np.allclose(scorer.evaluate(X, full=False)[0].numpy(), ps.score, atol=1e-3)


def test_disabling_the_grid_restores_the_exact_path():
    scorer, cfg, mol, pos = make()
    cfg.grid_spacing = 0.0
    assert scorer.attach_grid(CENTER, 14.0, cfg) is None
    X = torch.tensor(pos[None], dtype=torch.float32)
    assert torch.equal(scorer.energy(X), scorer.evaluate(X, full=False)[0])
