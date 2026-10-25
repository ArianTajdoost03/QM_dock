from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from qmdock.chem.atomtypes import LigandTypes, ReceptorTypes, ligand_types, receptor_types, score_types
from qmdock.chem.elements import DONOR_ELEMENTS, METALS, VDW_RADII
from qmdock.chem.receptor import load_receptor
from qmdock.cli import build_parser, config_from_args
from qmdock.config import Config
from qmdock.search.terms import pair_energy, ramp, relu

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2"


def typed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return mol, ligand_types(mol)


def of(mol, types, symbol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == symbol]


def test_amide_nitrogen_nitro_oxygen_and_ammonium_are_not_metal_donors_but_acids_amines_and_azoles_are():
    mol, t = typed("CC(=O)N")
    assert t.metal_weight[of(mol, t, "N")[0]] == 0 and t.metal_weight[of(mol, t, "O")[0]] == 1
    mol, t = typed("O=[N+]([O-])c1ccccc1")
    assert all(t.metal_weight[i] == 0 for i in of(mol, t, "O") + of(mol, t, "N"))
    mol, t = typed("C[NH3+]")
    assert t.metal_weight[of(mol, t, "N")[0]] == 0
    mol, t = typed("c1c[nH]cn1")
    pyrrole, pyridine = sorted(of(mol, t, "N"), key=lambda i: not mol.GetAtomWithIdx(i).GetTotalNumHs())
    assert t.metal_weight[pyrrole] == 0 and t.metal_weight[pyridine] == 1
    for smiles in ("CC(=O)[O-]", "NS(=O)(=O)c1ccccc1", "CCS", "c1ccncc1"):
        mol, t = typed(smiles)
        assert any(t.metal_weight[i] for i in range(mol.GetNumAtoms()))


def test_hydrogen_bond_and_hydrophobic_atoms_are_typed():
    mol, t = typed("CC(=O)O")
    donors = [i for i in range(mol.GetNumAtoms()) if t.donor[i]]
    assert len(donors) == 1 and mol.GetAtomWithIdx(donors[0]).GetSymbol() == "O"
    assert t.acceptor[of(mol, t, "O")].all()
    mol, t = typed("Cc1ccc(Cl)cc1")
    chlorinated = [i for i in of(mol, t, "C") if any(n.GetSymbol() == "Cl" for n in mol.GetAtomWithIdx(i).GetNeighbors())]
    assert t.hydrophobic[of(mol, t, "Cl")[0]] and len(chlorinated) == 1
    assert all(t.hydrophobic[i] for i in of(mol, t, "C") if i not in chlorinated) and not t.hydrophobic[chlorinated[0]]
    mol, t = typed("CCO")
    assert not t.hydrophobic[of(mol, t, "C")[1]] and t.hydrophobic[of(mol, t, "C")[0]]
    mol, t = typed("c1ccncc1")
    assert t.acceptor[of(mol, t, "N")[0]] and not t.donor[of(mol, t, "N")[0]]


def test_resonance_equivalent_oxygens_share_their_charge():
    mol, t = typed("CC(=O)[O-]")
    assert np.allclose(t.charge[of(mol, t, "O")], -0.5)
    mol, t = typed("CP(=O)([O-])[O-]")
    assert np.allclose(t.charge[of(mol, t, "O")], -2 / 3)
    mol, t = typed("O=[N+]([O-])c1ccccc1")
    assert abs(t.charge.sum()) < 1e-9 and np.allclose(t.charge[of(mol, t, "O")], -0.5)


def environment(rec):
    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False))), addCoords=True)
    env = rec.environment(ligand.GetConformer().GetPositions().mean(axis=0), 30.0)
    return ligand, env


def label_index(rec, env, residue, number, atom):
    for k, a in enumerate(env):
        res = rec.residues[int(rec.residue_of_atom[a])]
        if res[3] == residue and res[1] == str(number) and rec.names[a] == atom:
            return k


def test_receptor_atoms_are_typed_and_metal_sites_count_their_free_positions():
    rec = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    ligand, env = environment(rec)
    t = receptor_types(rec, env)
    zinc = [k for k, a in enumerate(env) if rec.elements[a] == "Zn"][0]
    mercury = [k for k, a in enumerate(env) if rec.elements[a] == "Hg"][0]
    assert t.metal_free[zinc] == 1 and t.metal_free[mercury] == 1
    bound = label_index(rec, env, "HIS", 94, "NE2")
    assert not t.acceptor[bound] and not t.donor[bound]
    threonine = label_index(rec, env, "THR", 199, "OG1")
    assert t.donor[threonine] and t.acceptor[threonine]
    glutamate = label_index(rec, env, "GLU", 106, "OE1")
    assert t.charge[glutamate] == -0.5 and t.acceptor[glutamate]
    lysine = label_index(rec, env, "LYS", 18, "NZ")
    assert t.charge[lysine] == 1.0 and t.donor[lysine]
    carbon = [k for k, a in enumerate(env) if rec.elements[a] == "C"]
    assert 0.3 < t.hydrophobic[carbon].mean() < 0.7


def pair_for(n_lig, n_env, **kw):
    pair = dict(rsum=np.full((n_lig, n_env), 3.3), coord=np.zeros((n_lig, n_env), bool), metal_weight=np.zeros(n_lig),
                metal_cols=None, metal_free=None, hbond=None, hydrophobic=None, charges=None,
                w_hbond=0.587, w_hydrophobic=0.0351, w_elec=3.0)
    pair.update(kw)
    return SimpleNamespace(**pair)


def test_ramp_and_relu_helpers_are_piecewise_linear():
    x = np.array([-1.0, 0.0, 0.25, 1.0, 2.0])
    assert np.allclose(relu(x), [0, 0, 0.25, 1, 2]) and np.allclose(ramp(x), [0, 0, 0.25, 1, 1])


def test_a_second_donor_adds_little_when_the_metal_has_one_free_site_and_nothing_when_it_has_none():
    d = np.full((1, 2, 1), 2.1)
    base = dict(metal_cols=np.array([0]), metal_weight=np.array([1.0, 1.0]), coord=np.ones((2, 1), bool), rsum=np.full((2, 1), 3.3))
    one = pair_energy(np, d[:, :1], pair_for(1, 1, **{**base, "metal_weight": np.array([1.0]), "coord": np.ones((1, 1), bool),
                                                      "rsum": np.full((1, 1), 3.3), "metal_free": np.array([1.0])}))[0]
    two = pair_energy(np, d, pair_for(2, 1, **{**base, "metal_free": np.array([1.0])}))[0]
    none = pair_energy(np, d, pair_for(2, 1, **{**base, "metal_free": np.array([0.0])}))[0]
    legacy = pair_energy(np, d, pair_for(2, 1, **base))[0]
    assert abs(one - -np.tanh(1.0)) < 1e-5
    assert one - two < 0.5 * abs(one) and abs(none) < 1e-5 and abs(legacy - -2.0) < 1e-9


def test_hydrogen_bonds_hydrophobic_contacts_and_charges_have_the_expected_shape_and_sign():
    def energy(distance, **kw):
        return pair_energy(np, np.full((1, 1, 1), distance), pair_for(1, 1, **kw))[0]

    hb = np.ones((1, 1))
    near, far, beyond = (energy(x, hbond=hb) - energy(x) for x in (2.8, 3.15, 3.6))
    assert abs(near - -0.587) < 1e-9 and abs(far - -0.587 / 2) < 1e-9 and abs(beyond) < 1e-9
    hp = np.ones((1, 1))
    close, mid, apart = (energy(3.3 + s, hydrophobic=hp) - energy(3.3 + s) for s in (0.5, 1.4, 2.5))
    assert abs(close - -0.0351) < 1e-9 and abs(mid - -0.0351 / 2) < 1e-9 and abs(apart) < 1e-9
    attract = energy(3.0, charges=-np.ones((1, 1))) - energy(3.0)
    repel = energy(3.0, charges=np.ones((1, 1))) - energy(3.0)
    assert attract < 0 < repel and abs(attract + repel) < 1e-9
    assert abs(energy(9.0, charges=-np.ones((1, 1))) - energy(9.0)) < 1e-12


def test_the_vectorised_energy_matches_explicit_loops():
    rng = np.random.default_rng(3)
    B, L, E = 3, 5, 7
    d = rng.uniform(1.4, 9.0, size=(B, L, E))
    rsum = rng.uniform(3.0, 3.6, size=(L, E))
    cols, weight = np.array([2, 5]), np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    coord = np.zeros((L, E), bool)
    coord[np.ix_(weight > 0, cols)] = True
    hb, hp = (rng.random((L, E)) < 0.3).astype(float), (rng.random((L, E)) < 0.3).astype(float)
    qq = rng.normal(size=(L, E)) * (rng.random((L, E)) < 0.4)
    free = np.array([1.0, 2.0])
    pair = pair_for(L, E, rsum=rsum, coord=coord, metal_weight=weight, metal_cols=cols, metal_free=free, hbond=hb,
                    hydrophobic=hp, charges=qq)
    clamp = lambda v: min(max(v, 0.0), 1.0)
    for b in range(B):
        e = 0.0
        for i in range(L):
            for j in range(E):
                s = d[b, i, j] - rsum[i, j]
                e += (5.0 * max(1.8 - d[b, i, j], 0.0) ** 2 if coord[i, j] else
                      -0.0356 * np.exp(-(s / 0.5) ** 2) - 0.00516 * np.exp(-((s - 3.0) / 2.0) ** 2) + 0.84 * min(s, 0.0) ** 2)
                e -= 0.587 * hb[i, j] * clamp((3.5 - d[b, i, j]) / 0.7) + 0.0351 * hp[i, j] * clamp(1.9 - s)
                e += 3.0 * qq[i, j] * (1.0 / max(d[b, i, j], 2.0) - 0.125) * clamp(8.0 - d[b, i, j])
        for c, m in enumerate(cols):
            total = sum(weight[i] * np.exp(-((d[b, i, m] - 2.1) / 0.35) ** 2) for i in range(L))
            e -= free[c] * np.tanh(total / (free[c] + 1e-6))
        assert abs(pair_energy(np, d, pair)[b] - e) < 1e-9


def test_chemical_scoring_can_be_switched_off_and_electrostatics_are_off_by_default():
    rec = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    ligand, env = environment(rec)
    assert score_types(rec, env, ligand, Config(receptor="r", ligand="l", chemical_score=False)) == (None, None)
    cfg = Config(receptor="r", ligand="l")
    assert cfg.chemical_score and cfg.elec_weight == 0.0 and cfg.hbond_weight == 0.587
    lt, et = score_types(rec, env, ligand, cfg)
    assert len(lt.metal_weight) == ligand.GetNumAtoms() and len(et.donor) == len(env)
    args = build_parser("dock.py").parse_args(["--receptor", "r.pdb", "--ligand", "l.sdf", "--center", "0", "0", "0",
                                                "--no-chemical-score", "--elec-weight", "2.0"])
    changed = config_from_args(args, "l.sdf")
    assert not changed.chemical_score and changed.elec_weight == 2.0


def test_the_torch_scorer_agrees_with_the_numpy_reference():
    torch = pytest.importorskip("torch")
    from qmdock.search.scoring import Scorer

    rng = np.random.default_rng(1)
    lig_elements = np.array(["O", "C", "H", "N", "O"])
    env_elements = np.array(["C", "O", "N", "Zn", "H", "C"])
    env_coords = rng.uniform(-6, 6, size=(6, 3))
    lt = LigandTypes(metal_weight=np.array([1.0, 0.0, 0.0, 0.0, 1.0]), donor=np.array([0, 0, 0, 1, 0], bool),
                     acceptor=np.array([1, 0, 0, 0, 1], bool), hydrophobic=np.array([0, 1, 0, 0, 0], bool),
                     charge=np.array([0.0, 0.0, 0.0, 0.0, -0.5]))
    et = ReceptorTypes(donor=np.array([0, 1, 0, 0, 0, 0], bool), acceptor=np.array([0, 1, 1, 0, 0, 0], bool),
                       hydrophobic=np.array([1, 0, 0, 0, 0, 1], bool), charge=np.array([0, 0, 0, 0, 0, 0.5]),
                       metal_free=np.array([0, 0, 0, 1.0, 0, 0]))
    cfg = Config(receptor="r", ligand="l", elec_weight=3.0)
    scorer = Scorer(env_elements, env_coords, lig_elements, torch.device("cpu"), cfg, lt, et)
    X = rng.uniform(-4, 4, size=(4, 5, 3)).astype(np.float32)
    for full in (False, True):
        lig_i = np.arange(5) if full else np.array([0, 1, 3, 4])
        env_i = np.arange(6) if full else np.array([0, 1, 2, 3, 5])
        rl = np.array([VDW_RADII[e] for e in lig_elements[lig_i]])
        re_ = np.array([VDW_RADII[e] for e in env_elements[env_i]])
        metal = np.array([e in METALS for e in env_elements[env_i]])
        weight = lt.metal_weight[lig_i]
        mat = lambda m: m[np.ix_(lig_i, env_i)]
        hb = (lt.donor[lig_i][:, None] & et.acceptor[env_i][None]) | (lt.acceptor[lig_i][:, None] & et.donor[env_i][None])
        columns = np.where(metal)[0]
        pair = pair_for(len(lig_i), len(env_i), rsum=rl[:, None] + re_[None], coord=(weight > 0)[:, None] & metal[None],
                        metal_weight=weight, metal_cols=columns, metal_free=et.metal_free[env_i][columns],
                        hbond=hb.astype(float), hydrophobic=(lt.hydrophobic[lig_i][:, None] & et.hydrophobic[env_i][None]).astype(float),
                        charges=lt.charge[lig_i][:, None] * et.charge[env_i][None])
        d = np.linalg.norm(X[:, lig_i][:, :, None, :] - env_coords[env_i][None, None], axis=3)
        expected = pair_energy(np, d, pair)
        got, _ = scorer.evaluate(torch.as_tensor(X), full=full)
        assert np.allclose(got.numpy(), expected, atol=1e-3), (full, got.numpy(), expected)
