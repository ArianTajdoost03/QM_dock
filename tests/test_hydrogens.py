from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qmdock.chem.hydrogens import (carboxylate_ok, fix_metal_sites, line_xyz, load_protonated, regularize_hydrogens,
                                   split_receptor_lines, terminal_oxygens)
from qmdock.chem.receptor import load_receptor

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2" / "1A42.pdb"


def atom_line(serial, name, resname, resseq, element, xyz, record="ATOM"):
    return (f"{record:<6s}{serial:5d} {name:<4s} {resname:>3s} A{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n")


def histidine(zinc_on):
    ring = {"CG": 90, "ND1": 162, "CE1": 234, "NE2": 306, "CD2": 18}
    xyz = {n: 1.15 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]) for n, a in ring.items()}
    lines = [atom_line(i + 1, n, "HIS", 5, n[0], xyz[n]) for i, n in enumerate(ring)]
    wrong = "ND1" if zinc_on == "NE2" else "NE2"
    bound = xyz[zinc_on]
    hydrogen = "HD1" if zinc_on == "ND1" else "HE2"
    lines.append(atom_line(6, hydrogen, "HIS", 5, "H", bound * (1 + 1.01 / 1.15)))
    zinc = np.array([bound[0], bound[1], 2.0])
    return lines, zinc, wrong


def test_split_receptor_lines(tmp_path):
    lines = [atom_line(1, "N", "ALA", 1, "N", (0, 0, 0)), atom_line(2, "H", "ALA", 1, "H", (1, 0, 0)),
             atom_line(3, "O", "HOH", 2, "O", (5, 0, 0), "HETATM"), atom_line(4, "ZN", "ZN", 3, "Zn", (9, 0, 0), "HETATM"),
             atom_line(5, "C1", "LIG", 4, "C", (12, 0, 0), "HETATM")]
    path = tmp_path / "r.pdb"
    path.write_text("".join(lines))
    protein, other = split_receptor_lines(str(path))
    assert [l[12:16].strip() for l in protein] == ["N"]
    assert [l[12:16].strip() for l in other] == ["ZN", "C1"]


@pytest.mark.parametrize("zinc_on", ["NE2", "ND1"])
def test_histidine_proton_moves_off_the_metal_bound_nitrogen(zinc_on):
    lines, zinc, free = histidine(zinc_on)
    fixed, changed = fix_metal_sites(lines, zinc[None])
    names = {l[12:16].strip(): l for l in fixed}
    held_h = "HD1" if zinc_on == "ND1" else "HE2"
    free_h = "HE2" if zinc_on == "ND1" else "HD1"
    assert held_h not in names and free_h in names and changed == 2
    gap = np.linalg.norm(line_xyz(names[free_h]) - line_xyz(names[free]))
    assert abs(gap - 1.01) < 0.01


def test_cysteine_thiol_hydrogen_removed_on_metal():
    lines = [atom_line(1, "SG", "CYS", 7, "S", (0, 0, 0)), atom_line(2, "HG", "CYS", 7, "H", (1.3, 0, 0))]
    fixed, changed = fix_metal_sites(lines, np.array([[0.0, 2.3, 0.0]]))
    assert [l[12:16].strip() for l in fixed] == ["SG"] and changed == 1
    assert fix_metal_sites(lines, np.array([[0.0, 9.0, 0.0]]))[1] == 0


def test_regularize_hydrogens_sets_bond_length_and_keeps_direction():
    lines = [atom_line(1, "CA", "ALA", 1, "C", (0, 0, 0)), atom_line(2, "HA", "ALA", 1, "H", (0.0, 1.19, 0.0))]
    fixed = regularize_hydrogens(lines)
    assert np.allclose(line_xyz(fixed[1]), [0.0, 1.09, 0.0], atol=1e-3)


def test_protonation_of_carbonic_anhydrase(tmp_path):
    pytest.importorskip("pdbfixer")
    from qmdock.chem.hydrogens import protonate_receptor

    stripped = tmp_path / "bare.pdb"
    stripped.write_text("".join(l for l in CA2.read_text().splitlines(True)
                                if not (l.startswith(("ATOM", "HETATM")) and l[76:78].strip() == "H")))
    out = tmp_path / "protonated.pdb"
    added, adjusted, relaxed = protonate_receptor(str(stripped), str(out), 7.4)
    rec = load_receptor(str(out), 10, False)
    assert added > 1800 and rec.hydrogen_fraction() > 0.9 and adjusted >= 1
    zn = np.where(rec.elements == "Zn")[0][0]
    close = np.linalg.norm(rec.coords - rec.coords[zn], axis=1) < 2.6
    assert not (close & (rec.elements == "H")).any()
    assert (rec.elements == "Hg").sum() == 1
    carbon_h = [np.linalg.norm(rec.coords[h] - rec.coords[j]) for h in np.where(rec.elements == "H")[0]
                for j in rec.neighbors[h] if rec.elements[j] == "C"]
    assert abs(np.mean(carbon_h) - 1.09) < 0.02


def test_load_protonated_skips_receptors_that_already_have_hydrogens(tmp_path):
    lines = [atom_line(1, "O", "HOH", 1, "O", (0, 0, 0), "HETATM"), atom_line(2, "H1", "HOH", 1, "H", (0.96, 0, 0), "HETATM"),
             atom_line(3, "H2", "HOH", 1, "H", (-0.24, 0.93, 0), "HETATM")]
    path = tmp_path / "w.pdb"
    path.write_text("".join(lines))
    cfg = SimpleNamespace(receptor=str(path), min_unit_atoms=3, require_metal=False, keep_waters=True,
                          add_hydrogens=True, receptor_ph=7.4)
    messages = []
    load_protonated(cfg, str(tmp_path / "out"), messages.append)
    assert messages == [] and not (tmp_path / "out" / "receptor_h.pdb").exists()


def test_add_hydrogens_can_be_disabled(tmp_path):
    path = tmp_path / "bare.pdb"
    path.write_text("".join(atom_line(i, "C1", "LIG", i, "C", (3.0 * i, 0, 0), "HETATM") for i in range(1, 6)))
    cfg = SimpleNamespace(receptor=str(path), min_unit_atoms=1, require_metal=False, keep_waters=False,
                          add_hydrogens=False, receptor_ph=7.4)
    messages = []
    load_protonated(cfg, str(tmp_path / "out"), messages.append)
    assert len(messages) == 1 and messages[0].startswith("WARNING")


def ideal_backbone_end():
    c = np.zeros(3)
    ca = np.array([1.52, 0.0, 0.0])
    n = np.array([2.1, 1.3, 0.0])
    return n, ca, c


def test_terminal_oxygens_are_rebuilt_with_ideal_geometry_when_missing():
    n, ca, c = ideal_backbone_end()
    o, oxt = terminal_oxygens(n, ca, c, np.array([0.0, 1.54, 0.0]))
    assert carboxylate_ok(c, ca, o, oxt)


def test_good_carbonyl_oxygen_is_kept_and_only_oxt_is_placed():
    n, ca, c = ideal_backbone_end()
    good = c + 1.23 * np.array([-0.5, np.sqrt(3) / 2, 0.0])
    o, oxt = terminal_oxygens(n, ca, c, good)
    assert np.allclose(o, good) and carboxylate_ok(c, ca, o, oxt)


def test_badly_placed_oxt_is_detected():
    n, ca, c = ideal_backbone_end()
    o, oxt = terminal_oxygens(n, ca, c, np.array([0.0, 1.54, 0.0]))
    assert not carboxylate_ok(c, ca, o, c + 1.76 * (oxt - c) / np.linalg.norm(oxt - c))
