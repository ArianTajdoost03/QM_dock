import os
import random

import numpy as np
from scipy.spatial import cKDTree

from qmdock.chem.elements import METALS, WATER_NAMES
from qmdock.chem.pdb_io import normalize_element
from qmdock.chem.receptor import COORDINATION_CUTOFF, hydrogen_warning, load_receptor
from qmdock.config import PROTONATED_RECEPTOR_FILE

HISTIDINE_HYDROGENS = {"ND1": "HD1", "NE2": "HE2"}
HISTIDINE_RING = ("CG", "ND1", "CD2", "CE1", "NE2")
NH_LENGTH = 1.01
XH_LENGTHS = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}
XH_SEARCH = 1.4
HYDROGEN_FORCEFIELD = "amber14-all.xml"
HYDROGEN_SEED = 7
CARBOXYL_BOND = 1.25
CARBOXYL_TOLERANCE = 0.2


def split_receptor_lines(path):
    protein, other = [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            element = normalize_element(line[76:78], name)
            if line[17:20].strip() in WATER_NAMES or element in ("H", "D"):
                continue
            if line.startswith("ATOM"):
                protein.append(line)
            else:
                other.append(line)
    return protein, other


def line_xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def hydrogen_line(reference, name, xyz):
    return (reference[:12] + f" {name:<3s}" + reference[16:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            + reference[54:76] + " H" + reference[78:])


def fix_metal_sites(lines, metals):
    if len(metals) == 0:
        return lines, 0
    residues = {}
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            key = (line[21], line[22:26], line[26], line[17:20].strip())
            residues.setdefault(key, {})[line[12:16].strip()] = i
    bound = lambda i: np.linalg.norm(metals - line_xyz(lines[i]), axis=1).min() < COORDINATION_CUTOFF
    drop, add = set(), {}
    for (_, _, _, resname), by in residues.items():
        if resname == "CYS" and "SG" in by and "HG" in by and bound(by["SG"]):
            drop.add(by["HG"])
        elif resname in ("HIS", "HID", "HIE", "HIP") and all(n in by for n in HISTIDINE_HYDROGENS):
            flags = {n: bound(by[n]) for n in HISTIDINE_HYDROGENS}
            if sum(flags.values()) != 1:
                continue
            held = next(n for n in flags if flags[n])
            free = next(n for n in flags if not flags[n])
            if HISTIDINE_HYDROGENS[held] in by:
                drop.add(by[HISTIDINE_HYDROGENS[held]])
            if HISTIDINE_HYDROGENS[free] not in by:
                ring = np.mean([line_xyz(lines[by[n]]) for n in HISTIDINE_RING if n in by], axis=0)
                nitrogen = line_xyz(lines[by[free]])
                direction = (nitrogen - ring) / np.linalg.norm(nitrogen - ring)
                add[by[free]] = (HISTIDINE_HYDROGENS[free], nitrogen + NH_LENGTH * direction)
    fixed = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        fixed.append(line)
        if i in add:
            fixed.append(hydrogen_line(line, *add[i]))
    return fixed, len(drop) + len(add)


def regularize_hydrogens(lines):
    atoms = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
    elements = {i: normalize_element(lines[i][76:78], lines[i][12:16].strip()) for i in atoms}
    xyz = {i: line_xyz(lines[i]) for i in atoms}
    heavy = [i for i in atoms if elements[i] != "H"]
    tree = cKDTree(np.array([xyz[i] for i in heavy]))
    fixed = list(lines)
    for i in atoms:
        if elements[i] != "H":
            continue
        distance, k = tree.query(xyz[i])
        parent = heavy[k]
        if distance > XH_SEARCH or elements[parent] not in XH_LENGTHS:
            continue
        p = xyz[parent] + (xyz[i] - xyz[parent]) / distance * XH_LENGTHS[elements[parent]]
        fixed[i] = lines[i][:30] + f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}" + lines[i][54:]
    return fixed


def carboxylate_ok(c, ca, o, oxt):
    a = (ca - c) / np.linalg.norm(ca - c)
    b = (o - c) / np.linalg.norm(o - c)
    d = (oxt - c) / np.linalg.norm(oxt - c)
    return (abs(np.linalg.norm(o - c) - CARBOXYL_BOND) < CARBOXYL_TOLERANCE
            and abs(np.linalg.norm(oxt - c) - CARBOXYL_BOND) < CARBOXYL_TOLERANCE
            and abs(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) - 118) < 12
            and abs(np.degrees(np.arccos(np.clip(a @ d, -1, 1))) - 118) < 12
            and abs(np.degrees(np.arccos(np.clip(b @ d, -1, 1))) - 124) < 12)


def terminal_oxygens(n, ca, c, o):
    a = (ca - c) / np.linalg.norm(ca - c)
    b = (o - c) / np.linalg.norm(o - c)
    if abs(np.linalg.norm(o - c) - CARBOXYL_BOND) < CARBOXYL_TOLERANCE and abs(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) - 118) < 12:
        return o, c - CARBOXYL_BOND * (a + b) / np.linalg.norm(a + b)
    normal = np.cross(a, n - ca)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    p = np.cross(normal / np.linalg.norm(normal), a)
    return (c + CARBOXYL_BOND * (-0.5 * a + np.sqrt(3) / 2 * p), c + CARBOXYL_BOND * (-0.5 * a - np.sqrt(3) / 2 * p))


def repair_terminal_oxygens(fixer):
    from openmm import Vec3
    from openmm.unit import Quantity, angstrom, nanometer

    positions = [np.array(p.value_in_unit(angstrom)) for p in fixer.positions]
    changed = False
    for residue in fixer.topology.residues():
        by = {atom.name: atom.index for atom in residue.atoms()}
        if not all(name in by for name in ("N", "CA", "C", "O", "OXT")):
            continue
        n, ca, c, o, oxt = (positions[by[k]] for k in ("N", "CA", "C", "O", "OXT"))
        if carboxylate_ok(c, ca, o, oxt):
            continue
        positions[by["O"]], positions[by["OXT"]] = terminal_oxygens(n, ca, c, o)
        changed = True
    if changed:
        fixer.positions = Quantity([Vec3(*(p / 10.0)) for p in positions], nanometer)


def build_fixer(path):
    from openmm import Platform
    from pdbfixer import PDBFixer

    random.seed(HYDROGEN_SEED)
    fixer = PDBFixer(filename=path, platform=Platform.getPlatformByName("Reference"))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    repair_terminal_oxygens(fixer)
    return fixer


def protonate_receptor(path, out_path, ph):
    try:
        import pdbfixer
        from openmm.app import ForceField, PDBFile
    except ImportError as exc:
        raise RuntimeError("adding receptor hydrogens needs pdbfixer and openmm "
                           "(conda install -c conda-forge pdbfixer openmm)") from exc
    protein, other = split_receptor_lines(path)
    if not protein:
        raise ValueError(f"no protein atoms found in {path}")
    tmp = out_path + ".tmp.pdb"
    with open(tmp, "w") as f:
        f.writelines(protein)
        f.write("END\n")
    try:
        try:
            fixer = build_fixer(tmp)
            fixer.addMissingHydrogens(ph, forcefield=ForceField(HYDROGEN_FORCEFIELD))
            relaxed = True
        except Exception:
            fixer = build_fixer(tmp)
            fixer.addMissingHydrogens(ph)
            relaxed = False
        with open(out_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with open(out_path) as f:
        kept = [line for line in f if line.startswith(("ATOM", "TER"))]
    metals = np.array([line_xyz(line) for line in other
                       if normalize_element(line[76:78], line[12:16].strip()) in METALS]).reshape(-1, 3)
    kept, adjusted = fix_metal_sites(kept, metals)
    if not relaxed:
        kept = regularize_hydrogens(kept)
    with open(out_path, "w") as f:
        f.writelines(kept)
        f.writelines(other)
        f.write("END\n")
    added = sum(1 for line in kept if line.startswith("ATOM") and line[76:78].strip() == "H")
    return added, adjusted, relaxed


def load_protonated(cfg, out_dir, log):
    receptor = load_receptor(cfg.receptor, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    if hydrogen_warning(receptor) and cfg.add_hydrogens:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, PROTONATED_RECEPTOR_FILE)
        log(f"receptor has {receptor.hydrogen_fraction():.2f} hydrogens per heavy atom: "
            f"adding hydrogens with pdbfixer at pH {cfg.receptor_ph}")
        try:
            added, adjusted, relaxed = protonate_receptor(cfg.receptor, path, cfg.receptor_ph)
        except Exception as exc:
            raise RuntimeError(f"adding hydrogens failed ({exc}); protonate the receptor yourself "
                               "or pass --no-add-hydrogens") from exc
        log(f"added {added} hydrogens ({adjusted} changed at metal sites, "
            f"{'relaxed with ' + HYDROGEN_FORCEFIELD if relaxed else 'force field relaxation failed, bond lengths regularized'}), "
            f"protonated receptor saved to {path}")
        receptor = load_receptor(path, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    warning = hydrogen_warning(receptor)
    if warning:
        log(f"WARNING: {warning}")
    return receptor
