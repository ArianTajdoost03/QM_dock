import numpy as np

from receptor import load_receptor


def pdb_line(i, name, el, x, y, z):
    return f"HETATM{i:5d} {name:<4s}              {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n"


def water(i, x):
    return [pdb_line(i, "O", "O", x, 0, 0), pdb_line(i + 1, "H1", "H", x + 0.96, 0, 0),
            pdb_line(i + 2, "H2", "H", x - 0.24, 0.93, 0)]


def test_units_and_eligibility(tmp_path):
    lines = water(1, 0.0) + water(4, 10.0) + [pdb_line(7, "H9", "H", 20, 0, 0)]
    lines += [pdb_line(8, "N1", "N", 30, 0, 0), pdb_line(9, "H1", "H", 31.0, 0, 0), pdb_line(10, "H2", "H", 29.7, 0.95, 0)]
    path = tmp_path / "r.pdb"
    path.write_text("".join(lines))
    rec = load_receptor(str(path), min_unit_atoms=3, remove_waters=False)
    assert len(rec.unit_atoms) == 4
    assert rec.eligible.sum() == 2
    reasons = sorted(r for r in rec.reasons if r)
    assert reasons == ["fragment", "odd-electron"]


def test_cluster_selection(tmp_path):
    path = tmp_path / "r.pdb"
    path.write_text("".join(water(1, 0.0) + water(4, 10.0)))
    rec = load_receptor(str(path), min_unit_atoms=3, remove_waters=False)
    cl = rec.build_cluster(np.array([[1.0, 2.0, 0.0]]), 3.0)
    assert len(cl.key[0]) == 1 and cl.skipped == 0 and cl.truncated == 0 and cl.charge == 0
    cl = rec.build_cluster(np.array([[5.0, 1.0, 0.0]]), 6.0)
    assert len(cl.key[0]) == 2
    cl = rec.build_cluster(np.array([[5.0, 1.0, 0.0]]), 6.0, max_atoms=3)
    assert len(cl.key[0]) == 1 and cl.truncated == 1


def test_waters_removed_by_default(tmp_path):
    lines = [l.replace("HETATM", "HETATM") for l in water(1, 0.0)]
    lines = [l[:17] + "HOH" + l[20:] for l in lines]
    lines.append(pdb_line(4, "ZN", "Zn", 5.0, 0.0, 0.0))
    path = tmp_path / "w.pdb"
    path.write_text("".join(lines))
    rec = load_receptor(str(path), min_unit_atoms=1)
    assert list(rec.elements) == ["Zn"]
    assert len(load_receptor(str(path), min_unit_atoms=1, remove_waters=False).elements) == 4
