from pathlib import Path

import numpy as np

from qmdock.chem.receptor import hydrogen_warning, load_receptor


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


def test_hydrogen_warning(tmp_path):
    protonated = tmp_path / "h.pdb"
    protonated.write_text("".join(water(1, 0.0)))
    bare = tmp_path / "b.pdb"
    bare.write_text("".join(pdb_line(i, "C1", "C", 3.0 * i, 0, 0) for i in range(1, 6)))
    rec_h = load_receptor(str(protonated), min_unit_atoms=3, remove_waters=False)
    rec_b = load_receptor(str(bare), min_unit_atoms=1, remove_waters=False)
    assert hydrogen_warning(rec_h) is None
    assert hydrogen_warning(rec_b) is not None


CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2" / "1A42.pdb"


def residue_index(rec, name, number):
    return next(i for i, r in enumerate(rec.residues) if r[3] == name and r[1] == str(number))


def atom_index(rec, name, number, atom):
    return next(a for a in rec.residue_atoms[residue_index(rec, name, number)] if rec.names[a] == atom)


def test_metal_ligands_are_mapped_to_their_metal():
    rec = load_receptor(str(CA2), 10, False)
    zinc = next(i for i, r in enumerate(rec.residues) if r[3] == "ZN")
    for number in (94, 96, 119):
        assert zinc in rec.metal_partners[residue_index(rec, "HIS", number)]


def test_touching_a_metal_ligand_brings_in_the_whole_metal_site():
    rec = load_receptor(str(CA2), 10, False)
    zinc_atom = next(i for i, e in enumerate(rec.elements) if e == "Zn")
    ring = rec.coords[atom_index(rec, "HIS", 96, "CE1")]
    away = (ring - rec.coords[zinc_atom]) / np.linalg.norm(ring - rec.coords[zinc_atom])
    probe = (ring + 3.4 * away)[None]
    assert np.linalg.norm(probe - rec.coords[zinc_atom]) > 4.5
    cl = rec.build_cluster(probe, 4.0, 150)
    members = {rec.residues[int(rec.residue_of_atom[a])][3] + rec.residues[int(rec.residue_of_atom[a])][1] for a in cl.atoms}
    assert {"ZN262", "HIS94", "HIS96", "HIS119"} <= members
    assert cl.electrons() % 2 == 0


def test_no_cap_hydrogen_is_placed_on_a_metal_and_parity_is_even_everywhere():
    rec = load_receptor(str(CA2), 10, False)
    metals = rec.coords[np.isin(rec.elements, ["Zn", "Hg"])]
    for r in range(len(rec.residues)):
        cl = rec.build_cluster(rec.coords[rec.residue_atoms[r][0]][None], 4.0, 150)
        assert cl.electrons() % 2 == 0
        if len(cl.cap_elements):
            assert (np.linalg.norm(cl.cap_coords[:, None] - metals[None], axis=2) > 1.6).all()


def ligand_coordinates():
    from rdkit import Chem

    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2.parent / "ligand.sdf"), removeHs=False))), addCoords=True)
    return ligand.GetConformer().GetPositions()


def contact_residues(rec, coords, radius):
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    found = {}
    for a in range(len(rec.coords)):
        d = tree.query(rec.coords[a])[0]
        if d <= radius and rec.polymer[rec.unit_of_atom[a]]:
            r = int(rec.residue_of_atom[a])
            found[r] = min(found.get(r, 99.0), d)
    return found


def test_every_contacting_residue_is_kept_without_a_cap():
    rec = load_receptor(str(CA2), 10, False)
    coords = ligand_coordinates()
    cl = rec.build_cluster(coords, 4.0)
    assert set(contact_residues(rec, coords, 4.0)) <= set(cl.key[1])
    assert cl.truncated == 0 and cl.electrons() % 2 == 0


def test_a_cap_cuts_nearest_first_and_never_skips_to_a_smaller_residue():
    rec = load_receptor(str(CA2), 10, False)
    coords = ligand_coordinates()
    contacts = contact_residues(rec, coords, 4.0)
    cl = rec.build_cluster(coords, 4.0, max_atoms=150)
    kept = set(cl.key[1])
    forced = rec.metal_site(contacts)
    loose = [r for r in contacts if r not in forced]
    kept_far = max(contacts[r] for r in loose if r in kept)
    dropped_near = min(contacts[r] for r in loose if r not in kept)
    assert cl.truncated > 0 and kept_far <= dropped_near


def test_charged_residues_and_their_metal_sites_are_added_out_to_the_charged_shell():
    rec = load_receptor(str(CA2), 10, False)
    coords = ligand_coordinates()
    base = rec.build_cluster(coords, 4.0, None, 0.0)
    wide = rec.build_cluster(coords, 4.0, None, 10.0)
    extra = set(wide.key[1]) - set(base.key[1])
    allowed = rec.metal_site([r for r in extra if rec.residue_charge(r) != 0]) | {r for r in extra if rec.residue_charge(r) != 0}
    assert extra and extra <= allowed
    assert set(base.key[1]) <= set(wide.key[1]) and wide.electrons() % 2 == 0
    close = contact_residues(rec, coords, 10.0)
    assert not [r for r, d in close.items() if d <= 10.0 and rec.residue_charge(r) != 0 and r not in set(wide.key[1])]
