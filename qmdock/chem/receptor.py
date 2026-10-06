from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from qmdock.chem.elements import ATOMIC_NUMBER, BOND_TOLERANCE, COVALENT_RADII, LINK_BOND, METAL_CHARGE, METALS
from qmdock.chem.pdb_io import read_pdb

COORDINATION_CUTOFF = 2.8


@dataclass
class Cluster:
    key: tuple
    atoms: np.ndarray
    cap_elements: list
    cap_coords: np.ndarray
    charge: int
    n_residues: int
    skipped: int
    truncated: int
    elements_all: np.ndarray
    coords_all: np.ndarray

    @property
    def n_atoms(self):
        return len(self.atoms) + len(self.cap_elements)

    def elements(self):
        return list(self.elements_all[self.atoms]) + list(self.cap_elements)

    def coords(self):
        if len(self.cap_elements) == 0:
            return self.coords_all[self.atoms]
        return np.vstack([self.coords_all[self.atoms], self.cap_coords])

    def electrons(self):
        return sum(ATOMIC_NUMBER[e] for e in self.elements()) - self.charge


@dataclass
class Receptor:
    elements: np.ndarray
    coords: np.ndarray
    names: list
    tree: cKDTree
    unit_of_atom: np.ndarray
    unit_atoms: list
    eligible: np.ndarray
    reasons: list
    polymer: np.ndarray
    neighbors: list
    residue_of_atom: np.ndarray
    residues: list
    residue_atoms: list

    def environment(self, center, radius):
        return np.array(sorted(self.tree.query_ball_point(center, radius)), dtype=int)

    def summary(self):
        counts = {}
        for u, atoms in enumerate(self.unit_atoms):
            formula = "".join(
                f"{el}{n}" for el, n in sorted(zip(*np.unique(self.elements[atoms], return_counts=True)))
            )
            status = "polymer (residue clusters)" if self.polymer[u] else (
                "eligible" if self.eligible[u] else self.reasons[u])
            key = (formula, status)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def label(self, i):
        res = self.residues[self.residue_of_atom[i]]
        return f"{res[3]}{res[1]}:{self.names[i]}"

    def h_count(self, atom):
        return sum(1 for j in self.neighbors[atom] if self.elements[j] == "H")

    def residue_charge(self, r):
        idx = self.residue_atoms[r]
        name = self.residues[r][3]
        by_name = {self.names[i]: i for i in idx}
        if len(idx) == 1 and self.elements[idx[0]] in METAL_CHARGE:
            return METAL_CHARGE[self.elements[idx[0]]]
        q = 0
        hs = lambda *ns: [self.h_count(by_name[n]) for n in ns if n in by_name]
        if name in ("ASP", "ASH"):
            h = hs("OD1", "OD2")
            q -= 1 if h and sum(h) == 0 else 0
        elif name in ("GLU", "GLH"):
            h = hs("OE1", "OE2")
            q -= 1 if h and sum(h) == 0 else 0
        elif name == "LYS":
            q += 1 if hs("NZ") == [3] else 0
        elif name == "ARG":
            q += 1 if sum(hs("NE", "NH1", "NH2")) == 5 else 0
        elif name in ("HIS", "HID", "HIE", "HIP"):
            h = hs("ND1", "NE2")
            q += 1 if len(h) == 2 and min(h) >= 1 else 0
        elif name in ("CYS", "CYM", "CYX"):
            if "SG" in by_name and self.h_count(by_name["SG"]) == 0 and not any(
                    self.elements[j] == "S" for j in self.neighbors[by_name["SG"]]):
                q -= 1
        elif name == "TYR":
            q -= 1 if hs("OH") == [0] else 0
        if "N" in by_name and self.h_count(by_name["N"]) == 3:
            q += 1
        if "OXT" in by_name and self.h_count(by_name["OXT"]) == 0:
            q -= 1
        return q

    def coordinating_residues(self, metal_atom):
        near = self.tree.query_ball_point(self.coords[metal_atom], COORDINATION_CUTOFF)
        r0 = self.residue_of_atom[metal_atom]
        return {int(self.residue_of_atom[i]) for i in near
                if self.elements[i] in ("N", "O", "S") and self.residue_of_atom[i] != r0}

    def build_cluster(self, lig_coords, shell, max_atoms=None):
        hits = self.tree.query_ball_point(lig_coords, shell)
        ids = np.unique(np.concatenate([np.asarray(h, dtype=int) for h in hits]))
        if ids.size == 0:
            return None
        dist = cKDTree(lig_coords).query(self.coords[ids])[0]
        units, res = {}, {}
        for a, d in zip(ids, dist):
            u = int(self.unit_of_atom[a])
            if self.polymer[u]:
                r = int(self.residue_of_atom[a])
                res[r] = min(res.get(r, 1e9), float(d))
            else:
                units[u] = min(units.get(u, 1e9), float(d))
        skipped = sum(1 for u in units if not self.eligible[u] and self.reasons[u] != "fragment")
        units = {u: d for u, d in units.items() if self.eligible[u]}

        forced = set()
        for r in list(res):
            atoms = self.residue_atoms[r]
            if len(atoms) == 1 and self.elements[atoms[0]] in METALS:
                forced.add(r)
                forced |= self.coordinating_residues(atoms[0])
        chosen_res = set(forced)
        chosen_units = []
        total = sum(len(self.residue_atoms[r]) for r in chosen_res)
        cands = sorted([(d, "u", u) for u, d in units.items()] + [(d, "r", r) for r, d in res.items()
                                                                  if r not in chosen_res])
        truncated = 0
        for d, kind, i in cands:
            size = len(self.unit_atoms[i]) if kind == "u" else len(self.residue_atoms[i]) + 2
            if max_atoms is not None and (chosen_res or chosen_units) and total + size > max_atoms:
                truncated += 1
                continue
            total += size
            if kind == "u":
                chosen_units.append(i)
            else:
                chosen_res.add(i)
        if not chosen_res and not chosen_units:
            return None

        parts = [self.unit_atoms[u] for u in chosen_units] + [self.residue_atoms[r] for r in sorted(chosen_res)]
        atoms = np.sort(np.concatenate(parts))
        selected = np.zeros(len(self.elements), dtype=bool)
        selected[atoms] = True
        cap_el, cap_xyz = [], []
        for i in atoms:
            if not self.polymer[self.unit_of_atom[i]]:
                continue
            for j in self.neighbors[i]:
                if not selected[j]:
                    v = self.coords[j] - self.coords[i]
                    cap_xyz.append(self.coords[i] + v / np.linalg.norm(v) * LINK_BOND.get(self.elements[i], 1.09))
                    cap_el.append("H")
        charge = sum(self.residue_charge(r) for r in chosen_res)
        key = (tuple(sorted(chosen_units)), tuple(sorted(chosen_res)))
        return Cluster(key, atoms, cap_el, np.array(cap_xyz).reshape(-1, 3), int(charge), len(chosen_res),
                       skipped, truncated, self.elements, self.coords)


def load_receptor(path, min_unit_atoms=10, require_metal=False, tolerance=BOND_TOLERANCE,
                  remove_waters=True, polymer_atoms=400):
    atoms = read_pdb(path, remove_waters)
    unknown = sorted(set(atoms.elements) - set(ATOMIC_NUMBER))
    if unknown:
        raise ValueError(f"unsupported elements in receptor: {unknown}")
    coords = atoms.coords
    n = len(coords)
    radii = np.array([COVALENT_RADII[e] for e in atoms.elements])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(3.2, output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        pairs = pairs[d <= tolerance * (radii[pairs[:, 0]] + radii[pairs[:, 1]])]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_units, labels = connected_components(graph, directed=False)
    neighbors = [[] for _ in range(n)]
    for i, j in pairs:
        neighbors[i].append(int(j))
        neighbors[j].append(int(i))
    unit_atoms = [np.where(labels == u)[0] for u in range(n_units)]
    keys = {}
    residue_of_atom = np.empty(n, dtype=int)
    for i, key in enumerate(atoms.residues):
        residue_of_atom[i] = keys.setdefault(key, len(keys))
    residues = list(keys)
    residue_atoms = [[] for _ in residues]
    for i, r in enumerate(residue_of_atom):
        residue_atoms[r].append(i)
    residue_atoms = [np.array(a) for a in residue_atoms]
    polymer = np.array([len(a) > polymer_atoms for a in unit_atoms])
    eligible = np.ones(n_units, dtype=bool)
    reasons = [""] * n_units
    for u, idx in enumerate(unit_atoms):
        electrons = sum(ATOMIC_NUMBER[e] for e in atoms.elements[idx])
        if polymer[u]:
            eligible[u], reasons[u] = False, "polymer"
        elif len(idx) < min_unit_atoms:
            eligible[u], reasons[u] = False, "fragment"
        elif electrons % 2:
            eligible[u], reasons[u] = False, "odd-electron"
        elif require_metal and not any(e in METALS for e in atoms.elements[idx]):
            eligible[u], reasons[u] = False, "no-metal"
    return Receptor(atoms.elements, coords, atoms.names, tree, labels, unit_atoms, eligible, reasons,
                    polymer, neighbors, residue_of_atom, residues, residue_atoms)


def find_surface_sites(receptor, n, spacing, seed=7, probe=4.0):
    rng = np.random.default_rng(seed)
    heavy = receptor.coords[receptor.elements != "H"]
    picks = heavy[rng.integers(len(heavy), size=20000)]
    v = rng.normal(size=picks.shape)
    cand = picks + probe * v / np.linalg.norm(v, axis=1, keepdims=True)
    free = receptor.tree.query(cand)[0] >= probe - 0.5
    cand = cand[free]
    counts = np.array(receptor.tree.query_ball_point(cand, 8.0, return_length=True))
    cand = cand[counts <= np.percentile(counts, 40)]
    sites = [cand[rng.integers(len(cand))]]
    while len(sites) < n:
        d = np.min(np.linalg.norm(cand[:, None] - np.array(sites)[None], axis=2), axis=1)
        d[d < spacing] = -1.0
        if d.max() < 0:
            break
        sites.append(cand[int(np.argmax(d))])
    return [tuple(float(x) for x in s) for s in sites]
