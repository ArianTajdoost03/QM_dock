from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from qmdock.chem.metals import (SCAN_MARGIN_KCAL, cluster_plan, describe, parse_state_overrides, planar_four,
                                spin_candidates)
from qmdock.chem.receptor import load_receptor
from qmdock.chem.spin import cluster_uhf
from qmdock.config import Config
from qmdock.pipeline.qmstages import Session

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2"


def make_pdb(tmp_path, metal="Zn", resname="ZN", charge="", second=None, donor=False, name="r.pdb"):
    out = []
    for line in (CA2 / "1A42.pdb").read_text().splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == "ZN":
            line = line.ljust(80)
            line = line[:12] + f" {metal.upper():<3s}" + line[16:17] + f"{resname:>3s}" + line[20:76] + f"{metal.upper():>2s}" + f"{charge:>2s}"
            out.append(line)
            if second:
                x = float(line[30:38]) + second[1]
                partner = line[:30] + f"{x:8.3f}" + line[38:]
                partner = partner[:6] + f"{9990:5d}" + partner[11:22] + f"{9990 % 10000:>4d}" + partner[26:]
                partner = partner[:12] + f" {second[0].upper():<3s}" + partner[16:17] + f"{second[2]:>3s}" + partner[20:76] + f"{second[0].upper():>2s}" + (second[3] if len(second) > 3 else "  ")
                out.append(partner)
            if donor:
                x = float(line[30:38]) + 1.9
                oxygen = line[:30] + f"{x:8.3f}" + line[38:]
                oxygen = oxygen[:6] + f"{9991:5d}" + oxygen[11:22] + f"{9991 % 10000:>4d}" + oxygen[26:]
                oxygen = oxygen[:12] + " O   " + oxygen[17:17] + " OH" + oxygen[20:76] + " O" + "  "
                out.append(oxygen)
            continue
        out.append(line)
    path = tmp_path / name
    path.write_text("\n".join(out) + "\n")
    return str(path)


def load(tmp_path, metal="Zn", resname="ZN", keep_waters=False, overrides=(), **kw):
    return load_receptor(make_pdb(tmp_path, metal, resname, **kw), 10, False, remove_waters=not keep_waters,
                         metal_overrides=overrides)


def only_site(rec):
    assert len(rec.metals) >= 1
    return next(iter(rec.metals.values()))


def test_oxidation_state_comes_from_the_pdb_ligand_code(tmp_path):
    site = only_site(load(tmp_path, "Cu", "CU1"))
    assert (site.element, site.oxidation, site.confidence) == ("Cu", 1, "medium") and "CU1" in site.source
    assert site.candidates == (0,)
    site = only_site(load(tmp_path, "Fe", "FE2"))
    assert site.oxidation == 2 and site.candidates[0] == 4
    site = only_site(load(tmp_path, "Fe", "FE"))
    assert site.oxidation == 3 and site.candidates[0] == 5
    site = only_site(load(tmp_path, "Co", "3CO"))
    assert site.oxidation == 3 and site.candidates[0] == 0
    site = only_site(load(tmp_path, "Mn", "MN3"))
    assert site.oxidation == 3 and site.candidates[0] == 4


def test_the_charge_column_outranks_the_ligand_code(tmp_path):
    site = only_site(load(tmp_path, "Fe", "FE2", charge="3+"))
    assert site.oxidation == 3 and site.source == "PDB charge column" and site.confidence == "high"


def test_user_override_wins_and_can_target_one_residue(tmp_path):
    site = only_site(load(tmp_path, "Cu", "CU1", overrides=parse_state_overrides(["Cu=2"])))
    assert site.oxidation == 2 and site.source == "user option"
    chain, number = next((r[0], r[1]) for r in load(tmp_path, "Cu", "CU1").residues if r[3] == "CU1")
    specific = parse_state_overrides([f"Cu@{chain}{number}=2", "Cu=3"])
    assert only_site(load(tmp_path, "Cu", "CU1", overrides=specific)).oxidation == 2
    other = parse_state_overrides(["Cu@Z999=2"])
    assert only_site(load(tmp_path, "Cu", "CU1", overrides=other)).oxidation == 1


def test_override_syntax_is_validated():
    for bad in ("Zn=2", "Cu", "Cu=x", "Cu=9"):
        with pytest.raises(ValueError):
            parse_state_overrides([bad])
    assert parse_state_overrides(["cu@a401=1"]) == [("Cu", "A401", 1)]


def test_copper_with_only_nitrogen_donors_is_inferred_to_be_cu_i_and_with_oxygen_cu_ii(tmp_path):
    site = only_site(load(tmp_path, "Cu", "CU"))
    assert site.oxidation == 1 and "coordination" in site.source and site.confidence == "low"
    site = only_site(load(tmp_path, "Cu", "CU", donor=True))
    assert site.oxidation == 2 and site.candidates == (1,) and site.donors.get("O") == 1
    site = only_site(load(tmp_path, "Cu", "CU", charge="2+"))
    assert site.oxidation == 2 and site.confidence == "high"


def test_metals_without_state_information_are_flagged_as_assumed(tmp_path):
    site = only_site(load(tmp_path, "Ni", "XYZ"))
    assert site.oxidation == 2 and site.confidence == "low" and "assumed" in site.source


def test_cluster_charge_follows_the_oxidation_state(tmp_path):
    one = load(tmp_path, "Cu", "CU1", name="a.pdb")
    two = load(tmp_path, "Cu", "CU", charge="2+", name="b.pdb")
    atom = next(iter(one.metals))
    probe = one.coords[atom][None] + np.array([[3.0, 0, 0]])
    c1 = one.build_cluster(probe, 4.0, None)
    c2 = two.build_cluster(probe, 4.0, None)
    assert c2.charge - c1.charge == 1
    assert c1.electrons() - c2.electrons() == -1 or c1.electrons() - c2.electrons() == 1


def test_spin_rules_for_common_electron_counts():
    assert spin_candidates("Cu", 1, 3, 3, False) == (0,)
    assert spin_candidates("Cu", 2, 4, 3, False) == (1,)
    assert spin_candidates("Ni", 2, 4, 4, True)[0] == 0
    assert spin_candidates("Ni", 2, 4, 4, False)[0] == 2
    assert spin_candidates("Ni", 2, 6, 6, False)[0] == 2
    assert spin_candidates("Fe", 2, 6, 6, False)[0] == 0
    assert spin_candidates("Fe", 2, 6, 2, False)[0] == 4
    assert spin_candidates("Fe", 3, 6, 6, False)[0] == 1
    assert spin_candidates("Fe", 3, 4, 1, False)[0] == 5
    assert spin_candidates("Co", 2, 6, 2, False)[0] == 3
    assert spin_candidates("Co", 3, 6, 2, False)[0] == 0
    assert spin_candidates("Mn", 2, 6, 1, False)[0] == 5
    assert spin_candidates("Mn", 4, 6, 1, False) == (3,)


def test_planarity_test_separates_square_planar_from_tetrahedral_four_coordination():
    metal = np.zeros(3)
    square = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0]], dtype=float)
    tetra = 2.0 / np.sqrt(3) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    assert planar_four(metal, square) and not planar_four(metal, tetra)


def test_two_nearby_paramagnetic_metals_are_coupled_into_a_lowest_spin_state(tmp_path):
    rec = load(tmp_path, "Cu", "CU", charge="2+", second=("Cu", 3.0, "CU"))
    sites = sorted(rec.metals.values(), key=lambda s: s.atom)
    assert len(sites) == 2 and sites[0].group == sites[1].group >= 0
    both = SimpleNamespace(atoms=np.array([s.atom for s in sites]))
    plan = cluster_plan(rec, both)
    assert plan.uhf == 0 and plan.confidence == "low" and any("coupled" in n for n in plan.notes)
    one = cluster_plan(rec, SimpleNamespace(atoms=np.array([sites[0].atom])))
    assert one.uhf == 1


def test_a_diamagnetic_neighbour_does_not_couple(tmp_path):
    rec = load(tmp_path, "Cu", "CU", charge="2+", second=("Zn", 3.0, "ZN"))
    cu = next(s for s in rec.metals.values())
    assert cu.group == -1
    assert cluster_plan(rec, SimpleNamespace(atoms=np.array([cu.atom]))).uhf == 1


def test_distant_paramagnetic_metals_stay_independent(tmp_path):
    rec = load(tmp_path, "Cu", "CU", charge="2+", second=("Cu", 12.0, "CU", "2+"))
    plan = cluster_plan(rec, SimpleNamespace(atoms=np.array(sorted(rec.metals))))
    assert plan.uhf == 2


def test_describe_lists_each_site_with_its_evidence(tmp_path):
    lines = describe(load(tmp_path, "Fe", "FE2"))
    assert len(lines) == 1 and "oxidation state 2" in lines[0] and "PDB ligand code FE2" in lines[0]


def test_protonated_receptor_keeps_the_states_found_in_the_original_file(tmp_path):
    original = load(tmp_path, "Cu", "CU1", name="orig.pdb")
    renamed = make_pdb(tmp_path, "Cu", "XXX", name="prot.pdb")
    again = load_receptor(renamed, 10, False, remove_waters=True, prior=original.metals)
    assert only_site(again).oxidation == 1 and "CU1" in only_site(again).source


def cluster_for(rec):
    atom = next(iter(rec.metals))
    return rec.build_cluster(rec.coords[atom][None] + np.array([[3.0, 0, 0]]), 4.0, None)


def test_cluster_spin_uses_the_site_plan_and_checks_parity(tmp_path):
    rec = load(tmp_path, "Cu", "CU", charge="2+")
    cl = cluster_for(rec)
    assert cluster_uhf(cl, "auto", rec) == 1
    assert cluster_uhf(cl, 0, rec) is None
    zn = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    assert cluster_uhf(zn.build_cluster(zn.coords[None, 0] * 0 + zn.coords[np.where(zn.elements == "Zn")[0][0]], 4.0, None),
                       "auto", zn) == 0


class Recorder:
    def __init__(self, delta):
        self.delta, self.calls = delta, []

    def map(self, jobs, label):
        self.calls.append((label, [(j["key"], j["uhf"], j["spinpol"]) for j in jobs]))
        out = {}
        for j in jobs:
            shift = self.delta.get(j["uhf"], 0.0) / 627.5095 if j["key"][0] == "spin" else 0.0
            out[j["key"]] = {"ok": True, "energy": -100.0 + shift, "attempt": 0}
        return out


def session_for(tmp_path, delta, **cfg):
    rec = load(tmp_path, "Fe", "FE2")
    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False))), addCoords=True)
    coords = ligand.GetConformer().GetPositions()
    lig = SimpleNamespace(numbers=np.array([a.GetAtomicNum() for a in ligand.GetAtoms()]), charge=0,
                          n_atoms=ligand.GetNumAtoms(), e_ref=-50.0)
    runner = Recorder(delta)
    return Session(Config("r", "l", **cfg), rec, runner), lig, coords, runner


def test_spin_check_keeps_the_rule_when_the_alternative_is_within_the_margin(tmp_path):
    s, lig, coords, runner = session_for(tmp_path, {0: -(SCAN_MARGIN_KCAL - 2.0)})
    rows = s.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    assert rows[0]["uhf"] == 4 and rows[0]["status"] == "ok"
    assert "spin check" in rows[0]["spin_note"]
    kinds = [k[0] for _, call in runner.calls for k, _, _ in call]
    assert kinds.count("spin") == 2 and kinds.count("pocket") == 0 and kinds.count("complex") == 1


def test_spin_check_switches_when_the_alternative_is_clearly_lower(tmp_path):
    s, lig, coords, runner = session_for(tmp_path, {0: -(SCAN_MARGIN_KCAL + 5.0)})
    rows = s.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    assert rows[0]["uhf"] == 0 and rows[0]["spin_confidence"] == "low"
    complex_calls = [c for _, call in runner.calls for c in call if c[0][0] == "complex"]
    assert complex_calls and all(u == 0 and sp == 0.0 for _, u, sp in complex_calls)


def test_open_shell_jobs_enable_spin_polarization_and_closed_shell_jobs_do_not(tmp_path):
    s, lig, coords, runner = session_for(tmp_path, {})
    s.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    spin_calls = [c for _, call in runner.calls for c in call if c[0][0] == "spin"]
    assert {u: sp for _, u, sp in spin_calls} == {4: 1.0, 0: 0.0}
    s2, lig2, coords2, runner2 = session_for(tmp_path, {}, spin_polarization=0.0)
    s2.evaluate(lig2, [0], {0: coords2}, {0: -50.0}, None)
    assert {sp for _, call in runner2.calls for _, _, sp in call} == {0.0}


def test_spin_check_runs_once_per_pocket_and_can_be_switched_off(tmp_path):
    s, lig, coords, runner = session_for(tmp_path, {})
    s.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    s.evaluate(lig, [0], {0: coords + 0.02}, {0: -50.0}, None)
    assert sum(1 for label, _ in runner.calls if label.startswith("spin check")) == 1
    s2, lig2, coords2, runner2 = session_for(tmp_path, {}, spin_scan=False)
    rows = s2.evaluate(lig2, [0], {0: coords2}, {0: -50.0}, None)
    assert rows[0]["uhf"] == 4 and not any(label.startswith("spin check") for label, _ in runner2.calls)


def test_an_explicit_uhf_bypasses_the_automatic_plan(tmp_path):
    s, lig, coords, runner = session_for(tmp_path, {}, uhf=2)
    rows = s.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    assert rows[0]["uhf"] == 2 and not any(label.startswith("spin check") for label, _ in runner.calls)
