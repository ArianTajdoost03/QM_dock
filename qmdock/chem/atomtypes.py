from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from qmdock.chem.elements import METALS
from qmdock.chem.receptor import COORDINATION_CUTOFF

HALOGENS = ("F", "Cl", "Br", "I")
NON_DONORS = ("[O;X1]~[N+;X3]~[O;X1]", "[N,n;+1]", "[N;X3;$(N-C=[O,S,N])]", "[n;X3]")
ACCEPTOR_NITROGEN = ("[n;X2;H0;+0]", "[N;X1]", "[N;X2;H0;+0]",
                     "[N;X3;+0;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$(N-a);!$(N-[#7,#8])]")
SIDE_CHARGE_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"),
                     "HIS": ("ND1", "NE2"), "HID": ("ND1", "NE2"), "HIE": ("ND1", "NE2"), "HIP": ("ND1", "NE2"),
                     "CYS": ("SG",), "CYM": ("SG",), "CYX": ("SG",), "TYR": ("OH",)}
METAL_CAPACITY = {"Zn": 4, "Cu": 4, "Cd": 4, "Hg": 2, "Ag": 2, "Ni": 6, "Co": 6, "Fe": 6, "Mn": 6, "Mg": 6, "Ca": 7}


@dataclass
class LigandTypes:
    metal_weight: np.ndarray
    donor: np.ndarray
    acceptor: np.ndarray
    hydrophobic: np.ndarray
    charge: np.ndarray


@dataclass
class ReceptorTypes:
    donor: np.ndarray
    acceptor: np.ndarray
    hydrophobic: np.ndarray
    charge: np.ndarray
    metal_free: np.ndarray


def heavy_neighbors(atom):
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def resonance_groups(mol):
    groups = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 7, 15, 16):
            continue
        by_element = {}
        for n in heavy_neighbors(atom):
            if n.GetAtomicNum() in (7, 8, 16) and len(heavy_neighbors(n)) == 1:
                by_element.setdefault(n.GetAtomicNum(), []).append(n.GetIdx())
        for members in by_element.values():
            if len(members) >= 2 and any(mol.GetAtomWithIdx(i).GetFormalCharge() != 0 for i in members):
                groups.append(members)
    return groups


def ligand_types(mol):
    n = mol.GetNumAtoms()
    weight = np.array([1.0 if a.GetAtomicNum() in (7, 8, 16) else 0.0 for a in mol.GetAtoms()])
    for smarts in NON_DONORS:
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False):
            weight[match[0]] = 0.0
    charge = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=float)
    for members in resonance_groups(mol):
        charge[members] = charge[members].sum() / len(members)
        weight[members] = weight[members].max()
    donor = np.zeros(n, bool)
    acceptor = np.zeros(n, bool)
    hydrophobic = np.zeros(n, bool)
    for atom in mol.GetAtoms():
        i, z = atom.GetIdx(), atom.GetAtomicNum()
        if z in (7, 8) and any(x.GetAtomicNum() == 1 for x in atom.GetNeighbors()):
            donor[i] = True
        if z == 8:
            acceptor[i] = True
        if z == 6 and all(x.GetAtomicNum() == 6 for x in heavy_neighbors(atom)):
            hydrophobic[i] = True
        if atom.GetSymbol() in HALOGENS:
            hydrophobic[i] = True
    for smarts in ACCEPTOR_NITROGEN:
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            acceptor[match[0]] = True
    return LigandTypes(weight, donor, acceptor, hydrophobic, charge)


def metal_free_sites(receptor, metal_index):
    element = receptor.elements[metal_index]
    near = receptor.tree.query_ball_point(receptor.coords[metal_index], COORDINATION_CUTOFF)
    bound = sum(1 for j in near if receptor.elements[j] in ("N", "O", "S") and j != metal_index)
    return float(max(METAL_CAPACITY.get(element, 4) - bound, 0))


def residue_charges(receptor, r, charge):
    atoms = {receptor.names[a]: int(a) for a in receptor.residue_atoms[r]}
    n_term = 1 if "N" in atoms and receptor.h_count(atoms["N"]) == 3 else 0
    c_term = -1 if "OXT" in atoms and receptor.h_count(atoms["OXT"]) == 0 else 0
    side = receptor.residue_charge(r) - n_term - c_term
    name = receptor.residues[r][3]
    present = [atoms[a] for a in SIDE_CHARGE_ATOMS.get(name, ()) if a in atoms]
    if side != 0 and present:
        for a in present:
            charge[a] += side / len(present)
    if n_term:
        charge[atoms["N"]] += 1.0
    if c_term:
        ends = [atoms[a] for a in ("O", "OXT") if a in atoms]
        for a in ends:
            charge[a] += c_term / len(ends)


def receptor_types(receptor, env):
    env = np.asarray(env)
    position = {int(a): k for k, a in enumerate(env)}
    n = len(env)
    metals = np.where(np.isin(receptor.elements, METALS))[0]
    bound = np.zeros(n, bool)
    if len(metals):
        tree = cKDTree(receptor.coords[metals])
        donors = np.array([receptor.elements[a] in ("N", "O", "S") for a in env])
        close = tree.query(receptor.coords[env], distance_upper_bound=COORDINATION_CUTOFF)[0] < np.inf
        bound = donors & close
    donor = np.zeros(n, bool)
    acceptor = np.zeros(n, bool)
    hydrophobic = np.zeros(n, bool)
    free = np.zeros(n)
    for k, a in enumerate(env):
        element = receptor.elements[a]
        heavy = [j for j in receptor.neighbors[a] if receptor.elements[j] != "H" and receptor.elements[j] not in METALS]
        if element in ("N", "O") and not bound[k]:
            h = receptor.h_count(a)
            donor[k] = h >= 1
            acceptor[k] = element == "O" or (h == 0 and len(heavy) == 2)
        elif element == "C":
            hydrophobic[k] = all(receptor.elements[j] == "C" for j in heavy)
        elif element in METALS:
            free[k] = metal_free_sites(receptor, int(a))
    charge_all = {}
    for r in {int(receptor.residue_of_atom[a]) for a in env}:
        if receptor.polymer[receptor.unit_of_atom[receptor.residue_atoms[r][0]]] and len(receptor.residue_atoms[r]) > 1:
            if receptor.residue_charge(r) != 0:
                local = np.zeros(len(receptor.coords))
                residue_charges(receptor, r, local)
                for a in receptor.residue_atoms[r]:
                    if local[a] != 0:
                        charge_all[int(a)] = local[a]
    charge = np.zeros(n)
    for a, value in charge_all.items():
        if a in position and not bound[position[a]]:
            charge[position[a]] = value
    return ReceptorTypes(donor, acceptor, hydrophobic, charge, free)


def score_types(receptor, env, mol, cfg):
    if not cfg.chemical_score:
        return None, None
    return ligand_types(mol), receptor_types(receptor, env)
