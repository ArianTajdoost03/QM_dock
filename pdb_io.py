from dataclasses import dataclass

import numpy as np

from metals import ATOMIC_NUMBER, WATER_NAMES


@dataclass
class Atoms:
    elements: np.ndarray
    coords: np.ndarray
    names: list
    residues: list


def normalize_element(symbol, name=""):
    s = "".join(c for c in symbol.strip() if c.isalpha())
    if not s:
        letters = "".join(c for c in name if c.isalpha())
        s = letters[:2] if letters[:2].capitalize() in ATOMIC_NUMBER else letters[:1]
    return s.capitalize()


def read_pdb(path, remove_waters=True):
    elements, coords, names, residues = [], [], [], []
    first_alt = {}
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip()
            if remove_waters and resname in WATER_NAMES:
                continue
            name = line[12:16].strip()
            chain, resseq, icode = line[21], line[22:26].strip(), line[26].strip()
            alt = line[16].strip()
            if alt:
                slot = (chain, resseq, icode, name)
                if first_alt.setdefault(slot, alt) != alt:
                    continue
            elements.append(normalize_element(line[76:78], name))
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            names.append(name)
            residues.append((chain, resseq, icode, resname))
    return Atoms(np.array(elements), np.array(coords, dtype=float), names, residues)


def write_xyz(path, elements, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords):
            f.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")
