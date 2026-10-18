import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

BENCH = Path(__file__).parent.parent / "data" / "bench_test_1"
CA2 = BENCH.parent / "carbonic anhydrase 2"
sys.path.insert(0, str(BENCH))

import bench_common
import check_inputs
import run_benchmark
import split_complexes
from qmdock.config import POOL_FILE, POSES_FILE, RESULTS_FILE, STATE_PREFIX


def crystal_ligand(tmp_path):
    ligand = next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False)))
    path = tmp_path / "crystal.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(Chem.RemoveHs(ligand))
    writer.close()
    return ligand, path


def test_options_are_validated_by_the_qmdock_parser():
    assert bench_common.read_options(text="--solvent water --torsions --search-radius 12")[0] == "--solvent"
    for bad in ("--solvnt water", "--search-radius far", "--protonation maybe", "--receptor x.pdb"):
        with pytest.raises(ValueError):
            bench_common.read_options(text=bad)


def test_default_options_file_is_accepted_by_qmdock():
    assert bench_common.read_options(path=str(BENCH / "dock_options.txt"))


def test_fresh_ligand_is_prepared_by_qmdock_and_centred_on_the_crystal_ligand(tmp_path):
    ligand, path = crystal_ligand(tmp_path)
    fresh, center, info = run_benchmark.fresh_ligand(str(path), "1a42", 7)
    heavy = [a.GetIdx() for a in ligand.GetAtoms() if a.GetAtomicNum() > 1]
    assert np.allclose(center, ligand.GetConformer().GetPositions()[heavy].mean(axis=0), atol=1e-3)
    assert fresh.GetNumConformers() == 1 and fresh.GetProp("_Name") == "1a42"
    assert info["n_heavy"] == len(heavy) and info["charge"] == 0


def test_unreadable_ligand_is_reported(tmp_path):
    path = tmp_path / "bad.sdf"
    path.write_text("not an sdf\n")
    row = check_inputs.check_complex("bad", str(CA2 / "1A42.pdb"), str(path), False, 7)
    assert row["status"] == "fail" and "ligand_failed" in row["problems"]


def build_raw_complex(tmp_path):
    ligand = next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False)))
    cache = tmp_path / "split" / "_ccd_templates"
    cache.mkdir(parents=True)
    ideal = Chem.Mol(ligand)
    Chem.Kekulize(ideal, clearAromaticFlags=True)
    writer = Chem.SDWriter(str(cache / "LIG.sdf"))
    writer.SetKekulize(True)
    writer.write(ideal)
    writer.close()
    heavy = Chem.RemoveHs(ligand)
    xyz = heavy.GetConformer().GetPositions()
    lines = [l for l in (CA2 / "1A42.pdb").read_text().splitlines(True) if l.startswith(("ATOM", "HETATM"))]
    counts = {}
    for i, atom in enumerate(heavy.GetAtoms()):
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        lines.append("HETATM%5d %-4s LIG A 999    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                     % (9000 + i, f"{symbol}{counts[symbol]}", *xyz[i], symbol.upper()))
    raw = tmp_path / "raw" / "1a42.pdb"
    raw.parent.mkdir()
    raw.write_text("".join(lines) + "END\n")
    return raw, tmp_path / "split", cache


def test_split_complexes_uses_qmdock_protonation_and_keeps_metals(tmp_path):
    pytest.importorskip("pdbfixer")
    from qmdock.chem.receptor import load_receptor

    raw, out, cache = build_raw_complex(tmp_path)
    row = split_complexes.process(str(raw), str(out), str(cache), True, 7.4, 6.0, 4)
    assert row["status"] == "ok" and row["ligand_bonds"] == "bond_orders_from_template"
    assert row["ligand"] == "LIG" and row["heavy_atoms"] == 23 and "ZN:1" in row["metals_kept"]
    receptor = load_receptor(str(out / "1a42" / "1a42_protein.pdb"))
    assert receptor.hydrogen_fraction() > 0.9
    assert Chem.MolFromMolFile(str(out / "1a42" / "1a42_ligand.sdf"), removeHs=False) is not None
    found = bench_common.find_complexes(str(out))
    assert check_inputs.check_complex("1a42", *found["1a42"], False, 7)["status"] == "ok"


def test_split_without_hydrogens_leaves_the_receptor_bare(tmp_path):
    raw, out, cache = build_raw_complex(tmp_path)
    row = split_complexes.process(str(raw), str(out), str(cache), False, 7.4, 6.0, 4)
    assert row["status"] == "ok"
    from qmdock.chem.receptor import load_receptor

    assert load_receptor(str(out / "1a42" / "1a42_protein.pdb")).hydrogen_fraction() == 0.0


def test_failures_are_attributed_to_search_selection_or_qm_ranking(tmp_path):
    import attribute_failures

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCc1ccccc1"))
    AllChem.EmbedMolecule(molecule, randomSeed=3)
    xyz0 = molecule.GetConformer().GetPositions()
    rng = np.random.default_rng(0)

    def write_sdf(path, coords):
        mol = Chem.Mol(molecule)
        mol.RemoveAllConformers()
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(coords):
            conformer.SetAtomPosition(i, [float(v) for v in p])
        mol.AddConformer(conformer)
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()

    def results(folder, pool_index, energy):
        folder.mkdir(parents=True, exist_ok=True)
        write_sdf(folder / POSES_FILE, xyz0)
        pd.DataFrame({"pose": range(len(pool_index)), "status": "ok", "pool_index": pool_index,
                      "e_final": energy}).to_csv(folder / RESULTS_FILE, index=False)

    def case(cid, native_present=True, shift=0.0, qm=None, states=None):
        (tmp_path / "data" / cid).mkdir(parents=True)
        write_sdf(tmp_path / "data" / cid / f"{cid}_ligand.sdf", xyz0)
        (tmp_path / "data" / cid / f"{cid}_protein.pdb").write_text("")
        run = tmp_path / "runs" / cid
        run.mkdir(parents=True)
        coords = np.stack([xyz0 + rng.normal(size=3) * 6 + rng.normal(scale=0.2, size=xyz0.shape) for _ in range(40)])
        if native_present:
            coords[5] = xyz0 + rng.normal(scale=0.2, size=xyz0.shape)
        coords += shift
        score = rng.uniform(-8, -2, 40)
        score[5] = -10
        np.savez(run / POOL_FILE, conf_coords=xyz0[None], conf_keys=np.array([0]), conf_vals=np.array([0.0]),
                 pool_coords=coords, pool_conf=np.zeros(40, int), pool_score=score, pool_site=np.zeros(40, int))
        if states is None:
            results(run, *qm)
        else:
            for k, (index, energy) in enumerate(states):
                results(run / f"{STATE_PREFIX}{k}", index, energy)

    case("a_ok", qm=([5, 1, 2], [-30, -20, -10]))
    case("b_search", native_present=False, shift=8.0, qm=([1, 2, 3], [-30, -20, -10]))
    case("c_selection", qm=([1, 2, 3], [-30, -20, -10]))
    case("d_qmrank", qm=([5, 1, 2], [-10, -30, -20]))
    case("e_state", states=[([5, 1], [-20, -10]), ([2, 3], [-40, -30])])
    expected = {"a_ok": "ok", "b_search": "search", "c_selection": "selection", "d_qmrank": "qm_ranking",
                "e_state": "state_selection"}
    crystal = {cid: str(tmp_path / "data" / cid / f"{cid}_ligand.sdf") for cid in expected}
    for cid, label in expected.items():
        assert attribute_failures.attribute(cid, str(tmp_path / "runs" / cid), crystal[cid], 2.0)["diagnosis"] == label
