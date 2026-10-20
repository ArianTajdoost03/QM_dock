from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from qmdock.chem.receptor import load_receptor
from qmdock.config import HARTREE_TO_KCAL, Config
from qmdock.pipeline.ligandprep import describe_ligand, prepare_conformers
from qmdock.pipeline.qmstages import Session
from qmdock.qm.engine import relax, run_job, solve
from qmdock.qm.jobs import strain_job

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2"
WATER = ("alpb", "water")


def embed(smiles, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def numbers(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()])


def heavy_index(mol):
    return np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])


def full_relax(mol, pos, charge, steps=300, restraint=None, info=None):
    return relax(numbers(mol), pos, np.arange(mol.GetNumAtoms()), charge, 0, steps, 0.0005, 1.0, WATER, restraint, info)


def heavy_rmsd(mol, a, b):
    h = heavy_index(mol)
    return float(np.sqrt(((a[h] - b[h]) ** 2).sum(axis=1).mean()))


def test_restrained_relaxation_removes_protonation_state_geometry_noise_without_moving_the_ligand():
    neutral = embed("OP(=O)(O)CCCc1ccccc1")
    template, _, _, _ = full_relax(neutral, neutral.GetConformer().GetPositions(), 0)
    heavy = heavy_index(neutral)
    flat = Chem.RemoveHs(embed("[O-]P(=O)(O)CCCc1ccccc1"))
    match = flat.GetSubstructMatch(Chem.RemoveHs(neutral))
    conformer = Chem.Conformer(flat.GetNumAtoms())
    for i, j in enumerate(match):
        conformer.SetAtomPosition(j, [float(v) for v in template[heavy[i]]])
    flat.RemoveAllConformers()
    flat.AddConformer(conformer)
    anion = Chem.AddHs(flat, addCoords=True)
    start = anion.GetConformer().GetPositions()
    raw = solve(numbers(anion), start, -1, 0, solv=WATER)[0]
    free_pos, free, _, _ = full_relax(anion, start, -1)
    info = {}
    held_pos, held, _, _ = full_relax(anion, start, -1, 200, (heavy_index(anion), 0.2), info)
    assert (raw - free) * HARTREE_TO_KCAL > 5.0
    assert abs(info["e_start"] - raw) < 1e-6
    assert abs(held - free) * HARTREE_TO_KCAL < 0.5
    assert heavy_rmsd(anion, start, held_pos) < 0.15


def test_restrained_relaxation_keeps_a_twisted_torsion_that_free_relaxation_undoes():
    mol = embed("CC(=O)Nc1ccccc1")
    minimum, e_min, _, _ = full_relax(mol, mol.GetConformer().GetPositions(), 0, 400)
    twist = (2, 1, 3, 4)
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(minimum):
        conformer.SetAtomPosition(i, [float(v) for v in p])
    work.AddConformer(conformer, assignId=True)
    conformer = work.GetConformer()
    start_angle = rdMolTransforms.GetDihedralDeg(conformer, *twist)
    rdMolTransforms.SetDihedralDeg(conformer, *twist, start_angle + 60.0)
    start = conformer.GetPositions()
    target = rdMolTransforms.GetDihedralDeg(conformer, *twist)

    def angle(pos):
        c = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(pos):
            c.SetAtomPosition(i, [float(v) for v in p])
        return rdMolTransforms.GetDihedralDeg(c, *twist)

    def drift(pos):
        return abs((angle(pos) - target + 180) % 360 - 180)

    free_pos, free, _, _ = full_relax(mol, start, 0, 200)
    held_pos, held, _, _ = full_relax(mol, start, 0, 200, (heavy_index(mol), 0.2))
    assert drift(free_pos) > 40.0 and drift(held_pos) < 25.0
    assert (held - free) * HARTREE_TO_KCAL > 2.0
    assert (solve(numbers(mol), start, 0, 0, solv=WATER)[0] - e_min) * HARTREE_TO_KCAL > (held - e_min) * HARTREE_TO_KCAL


def test_strain_job_reports_the_start_energy_and_never_ends_above_it():
    mol = embed("CCCO")
    cfg = Config(receptor="r", ligand="l", solvent="water")
    job = strain_job(("lig", 0), numbers(mol), mol.GetConformer().GetPositions(), heavy_index(mol), 0, cfg)
    result = run_job(job)
    assert result["ok"] and result["energy"] <= result["e_start"] + 1e-9 and result["steps"] >= 1


class Runner:
    def __init__(self, shifts=None):
        self.kinds = []
        self.shifts = shifts or {}

    def map(self, jobs, label):
        out = {}
        for job in jobs:
            key = job["key"]
            self.kinds.append(key[0])
            if key[0] == "pocket":
                out[key] = {"ok": True, "energy": -500.0, "attempt": 0}
            elif key[0] == "complex":
                out[key] = {"ok": True, "energy": -550.0 - 0.01 * key[1], "attempt": 0}
            elif key[0] == "lig":
                out[key] = {"ok": True, "e_start": -50.0, "energy": -50.02, "steps": 5}
            elif key[0] == "relax":
                pos = job["pos"].copy()
                pos[job["movable"]] += [self.shifts.get(key[1], 0.1), 0.0, 0.0]
                out[key] = {"ok": True, "pos": pos, "energy": -550.04 - 0.01 * key[1], "e_start": -550.0 - 0.01 * key[1],
                            "lig_energy": -50.04, "steps": 12}
            elif key[0] in ("conf", "ref"):
                out[key] = {"ok": True, "energy": -10.0 - 0.001 * key[1] - (0.01 if key[0] == "ref" else 0.0),
                            "pos": job["pos"]}
        return out


class Poses:
    def __init__(self, coords):
        self.coords = coords
        self.conf = np.zeros(len(coords), dtype=int)

    def __len__(self):
        return len(self.coords)


def setup_session(cfg, shifts=None):
    receptor = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False))), addCoords=True)
    coords = ligand.GetConformer().GetPositions()
    lig = SimpleNamespace(numbers=numbers(ligand), heavy=numbers(ligand) > 1, charge=0, n_atoms=ligand.GetNumAtoms(),
                          e_ref=-50.05, conf_e={}, conf_rel={})
    runner = Runner(shifts)
    sel = SimpleNamespace(poses=Poses(np.stack([coords, coords + 0.05, coords + 0.1])), first=[0, 1, 2], flex=True,
                          rejected={}, groups=None)
    return Session(cfg, receptor, runner), runner, lig, sel


def test_strain_comes_from_the_relaxed_ligand_while_the_interaction_energy_keeps_the_raw_one():
    session, runner, lig, sel = setup_session(Config(receptor="r", ligand="l"))
    result = session.run(lig, sel)
    row = result.stage1[0]
    assert abs(row["strain"] - 0.03 * HARTREE_TO_KCAL) < 1e-6 and abs(row["strain_raw"] - 0.05 * HARTREE_TO_KCAL) < 1e-6
    assert abs(row["e_int"] - (-550.0 + 500.0 + 50.0) * HARTREE_TO_KCAL) < 1e-6
    assert abs(row["e_bind"] - (row["e_int"] + row["strain"])) < 1e-9
    assert abs(row["e_bind_raw"] - (row["e_int"] + row["strain_raw"])) < 1e-9


def test_relaxation_energy_is_added_to_the_raw_binding_energy_and_the_final_stage_uses_complete_pockets():
    cfg = Config(receptor="r", ligand="l", relax_top=3, final_top=2)
    session, runner, lig, sel = setup_session(cfg, shifts={2: 3.0})
    result = session.run(lig, sel)
    assert set(result.relaxed) == {0, 1}
    assert any("pose 2 moved" in w for w in result.warnings)
    for p, r in result.relaxed.items():
        assert abs(r["e_bind_relaxed"] - (result.stage1[p]["e_bind_raw"] - 0.04 * HARTREE_TO_KCAL)) < 1e-6
        assert abs(r["strain_relaxed"] - 0.01 * HARTREE_TO_KCAL) < 1e-6
        assert r["relax_atoms"] <= 200 + 4 and r["relax_steps_used"] == 12
    assert set(result.stage3) == {0, 1}
    assert all(row["status"] == "ok" and abs(row["strain"] - 0.01 * HARTREE_TO_KCAL) < 1e-6 for row in result.stage3.values())
    assert runner.kinds.count("relax") == 3


def test_ligand_strain_of_rigid_poses_is_computed_once_per_conformer():
    cfg = Config(receptor="r", ligand="l")
    session, runner, lig, sel = setup_session(cfg)
    sel.flex = False
    lig.conf_coords = np.stack([sel.poses.coords[0]])
    lig.conf_e = {0: -50.0}
    lig_e, lig_rel = {}, {}
    session.ligand_energies(lig, sel, [0, 1, 2], {p: sel.poses.coords[p] for p in range(3)}, lig_e, lig_rel)
    session.ligand_energies(lig, sel, [0, 1, 2], {p: sel.poses.coords[p] for p in range(3)}, lig_e, lig_rel)
    assert runner.kinds.count("lig") == 1 and set(lig_rel) == {0, 1, 2}
    assert all(abs(lig_e[p] - -50.0) < 1e-12 and abs(lig_rel[p] - -50.02) < 1e-12 for p in range(3))


def test_the_reference_energy_comes_from_conformers_relaxed_to_convergence():
    cfg = Config(receptor="r", ligand="l", n_conformers=6, ref_conformers=2, ref_steps=400)
    mol = embed("CCCCO")
    lig = describe_ligand(cfg, mol)
    runner = Runner()
    refined, _ = prepare_conformers(cfg, lig, runner)
    assert runner.kinds.count("ref") == 2
    best = max(refined.conf_e)
    assert abs(refined.e_ref - (-10.0 - 0.001 * best - 0.01)) < 1e-12
    assert refined.e_ref == min(refined.conf_e.values())
