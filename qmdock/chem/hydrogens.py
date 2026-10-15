import os

import numpy as np
from scipy.spatial import cKDTree

from qmdock.chem.elements import METALS, WATER_NAMES
from qmdock.chem.pdb_io import normalize_element
from qmdock.chem.receptor import COORDINATION_CUTOFF, hydrogen_warning, load_receptor

HISTIDINE_HYDROGENS = {"ND1": "HD1", "NE2": "HE2"}
HISTIDINE_RING = ("CG", "ND1", "CD2", "CE1", "NE2")
NH_LENGTH = 1.01
XH_LENGTHS = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}
XH_SEARCH = 1.4
HYDROGEN_FORCEFIELD = "amber14-all.xml"


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


def build_fixer(path):
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=path)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
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
        path = os.path.join(out_dir, "receptor_h.pdb")
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
