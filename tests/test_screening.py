import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qmdock.chem.protonation import embed
from qmdock.screening.ligands import read_ligands
from qmdock.screening.report import choose_best_state, ligand_result, ranked_table, write_best_poses
from qmdock.screening.runner import ScreenOptions, check_inputs, saved_result, save_result
from qmdock.config import Config


def write_sdf(path, entries):
    writer = Chem.SDWriter(path)
    for name, mol in entries:
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()


def flat(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return mol


def test_reader_handles_good_bad_and_odd_entries(tmp_path):
    path = str(tmp_path / "lib.sdf")
    write_sdf(path, [("tropolone", embed(Chem.MolFromSmiles("O=c1cccccc1O"), 1)),
                     ("twod", flat("NS(=O)(=O)c1ccccc1")),
                     ("twod", flat("CCO")),
                     ("salt", flat("[Na+].[O-]c1cccccc1=O")),
                     ("tin", flat("C[Sn](C)C")),
                     ("radical", flat("C[CH2]")),
                     ("huge", flat("C" * 80))])
    with open(path, "a") as handle:
        handle.write("garbage record\n$$$$\n")
    records = read_ligands(path, max_heavy=60)
    by = {r.name: r for r in records}
    assert by["tropolone"].status == "ready" and by["tropolone"].mol.GetNumConformers() == 1
    assert by["twod"].status == "ready" and any("embedded" in n for n in by["twod"].notes)
    assert by["twod_3"].status == "ready"
    assert by["salt"].status == "ready" and any("fragments" in n for n in by["salt"].notes)
    assert by["tin"].status == "invalid" and "unsupported" in by["tin"].error
    assert by["radical"].status == "invalid" and "odd electron" in by["radical"].error
    assert by["huge"].status == "skipped" and "heavy atoms" in by["huge"].error
    assert any(r.status == "invalid" and r.name.startswith("lig") for r in records)
    assert len({r.name for r in records}) == len(records)


def test_reader_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_ligands(str(tmp_path / "missing.sdf"))
    empty = tmp_path / "empty.sdf"
    empty.write_text("")
    with pytest.raises(ValueError):
        read_ligands(str(empty))
    junk = tmp_path / "junk.sdf"
    junk.write_text("this is not an sdf\n")
    with pytest.raises(ValueError):
        read_ligands(str(junk))


def row(state, energy, level=1, error=None, directory="d"):
    if error:
        return {"state": state, "smiles": "x", "charge": 0, "dir": directory, "error": error}
    return {"state": state, "smiles": f"s{state}", "charge": state, "dir": directory, "best_e_final_kcal": energy,
            "best_level": level, "gap_to_next_family_kcal": None}


def test_best_state_is_the_lowest_energy_at_the_highest_level():
    assert choose_best_state([row(0, -10), row(1, -30), row(2, 0, error="boom")])["state"] == 1
    assert choose_best_state([row(0, -50, level=1), row(1, -20, level=2)])["state"] == 1
    assert choose_best_state([row(0, 0, error="a"), row(1, 0, error="b")]) is None


class Rec:
    def __init__(self, name, index, heavy=10):
        self.name, self.index, self.heavy_atoms, self.input_smiles, self.notes = name, index, heavy, "C", []


def test_results_ranking_and_failures():
    good_a = ligand_result(Rec("a", 0), [row(0, -10), row(1, -25)], 5.0, "d")
    good_b = ligand_result(Rec("b", 1, heavy=40), [row(0, -30)], 5.0, "d")
    bad = ligand_result(Rec("c", 2), [row(0, 0, error="SCF failed")], 5.0, "d")
    assert good_a["best_state"] == 1 and good_a["e_final_kcal"] == -25 and bad["status"] == "failed"
    table = ranked_table([good_a, good_b, bad], "energy")
    assert list(table.name) == ["b", "a", "c"] and table["rank"].iloc[0] == 1 and table["rank"].isna().iloc[2]
    assert list(ranked_table([good_a, good_b, bad], "per-atom").name)[0] == "a"


def test_best_poses_file_and_resume_store(tmp_path):
    pose = embed(Chem.MolFromSmiles("CCO"), 1)
    pose_path = str(tmp_path / "poses.sdf")
    write_sdf(pose_path, [("pose_0", pose)])
    result = ligand_result(Rec("a", 0), [dict(row(0, -12), dir=str(tmp_path))], 1.0, str(tmp_path))
    assert write_best_poses(str(tmp_path / "best.sdf"), [result]) == 1
    save_result(str(tmp_path), result)
    assert saved_result(str(tmp_path))["name"] == "a"
    (tmp_path / "result.json").write_text("{broken")
    assert saved_result(str(tmp_path)) is None


def test_input_checks(tmp_path):
    receptor, ligands = tmp_path / "r.pdb", tmp_path / "l.sdf"
    receptor.write_text("x")
    ligands.write_text("x")
    cfg = Config(str(receptor), str(ligands), out=str(tmp_path / "out"), centers=[(0.0, 0.0, 0.0)])
    check_inputs(cfg, ScreenOptions(str(ligands)))
    assert os.path.isdir(cfg.out)
    with pytest.raises(ValueError):
        check_inputs(Config(str(receptor), str(ligands), out=cfg.out), ScreenOptions(str(ligands)))
    with pytest.raises(FileNotFoundError):
        check_inputs(cfg, ScreenOptions(str(tmp_path / "nope.sdf")))
