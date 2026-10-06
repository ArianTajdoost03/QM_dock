import time
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem

from qmdock.chem.ligand import build_conformers
from qmdock.qm.jobs import energy_job, numbers_of, relax_job
from qmdock.qm.pool import log
from qmdock.search.clustering import auto_radius, symmetry_permutations


@dataclass
class LigandData:
    mol: Chem.Mol
    elements: list
    numbers: np.ndarray
    heavy: np.ndarray
    charge: int
    perms: np.ndarray
    family_radius: float
    conf_coords: np.ndarray = None
    conf_e: dict = field(default_factory=dict)
    e_ref: float = None

    @property
    def n_atoms(self):
        return len(self.elements)


def describe_ligand(cfg, mol):
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    numbers = numbers_of(elements)
    heavy = np.array([e != "H" for e in elements])
    charge = Chem.GetFormalCharge(mol) if cfg.ligand_charge is None else cfg.ligand_charge
    if (int(numbers.sum()) - charge) % 2:
        raise ValueError("ligand has an odd electron count; set --ligand-charge to its real protonation state")
    n_heavy = int(heavy.sum())
    perms = symmetry_permutations(mol)
    if perms.shape[1] != n_heavy:
        perms = np.arange(n_heavy)[None, :]
    radius = cfg.cluster_radius or auto_radius(n_heavy)
    medium = f", solvent {cfg.solvent} ({cfg.solvation_model})" if cfg.solvent else ", gas phase"
    log(f"ligand: {len(elements)} atoms, charge {charge}{medium}")
    return LigandData(mol, elements, numbers, heavy, charge, perms, radius)


def prepare_conformers(cfg, lig, runner):
    t = time.time()
    confs = build_conformers(lig.mol, cfg.n_conformers, cfg.seed, cfg.conformer_window, cfg.conformer_prune)
    coords = np.array(confs.coords, copy=True)
    n_conf = len(coords)
    log(f"conformers: {n_conf}")
    if cfg.prerelax_steps > 0:
        jobs = []
        for c in range(n_conf):
            job = relax_job(("conf", c), lig.numbers, coords[c], np.arange(lig.n_atoms), lig.charge, cfg)
            job["steps"] = cfg.prerelax_steps
            jobs.append(job)
        res = runner.map(jobs, "conformer GFN2 pre-relaxation")
    else:
        res = runner.map([energy_job(("conf", c), lig.numbers, coords[c], lig.charge, cfg)
                          for c in range(n_conf)], "conformer energies")
    conf_e = {}
    for c in range(n_conf):
        r = res[("conf", c)]
        if r["ok"]:
            conf_e[c] = r["energy"]
            if "pos" in r:
                coords[c] = r["pos"]
    if not conf_e:
        errors = sorted({r.get("error", "") for r in res.values() if not r["ok"]})
        raise RuntimeError(f"no conformer converged in GFN2: {errors[:3]}; run diagnose.py")
    e_ref = min(conf_e.values())
    log(f"ligand reference energy {e_ref:.6f} Ha ({len(conf_e)}/{n_conf} conformers converged)")
    return replace(lig, conf_coords=coords, conf_e=conf_e, e_ref=e_ref), time.time() - t
