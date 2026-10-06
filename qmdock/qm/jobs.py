import numpy as np

from qmdock.chem.elements import ATOMIC_NUMBER


def numbers_of(elements):
    return np.array([ATOMIC_NUMBER[e] for e in elements], dtype=np.int32)


def energy_job(key, numbers, pos, charge, cfg):
    return {"kind": "energy", "key": key, "numbers": numbers, "pos": pos, "charge": charge,
            "uhf": cfg.uhf, "accuracy": cfg.accuracy, "solvent": cfg.solvent, "model": cfg.solvation_model}


def relax_job(key, numbers, pos, movable, charge, cfg, lig_slice=None, lig_charge=0):
    job = {"kind": "relax", "key": key, "numbers": numbers, "pos": pos, "movable": movable,
           "charge": charge, "uhf": cfg.uhf, "steps": cfg.relax_steps, "fmax": cfg.relax_fmax,
           "accuracy": cfg.accuracy, "solvent": cfg.solvent, "model": cfg.solvation_model}
    if lig_slice:
        job["lig_slice"] = lig_slice
        job["lig_charge"] = lig_charge
    return job
