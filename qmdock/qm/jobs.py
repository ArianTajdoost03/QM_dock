import numpy as np

from qmdock.chem.elements import ATOMIC_NUMBER


def numbers_of(elements):
    return np.array([ATOMIC_NUMBER[e] for e in elements], dtype=np.int32)


def energy_job(key, numbers, pos, charge, cfg, uhf=0):
    return {"kind": "energy", "key": key, "numbers": numbers, "pos": pos, "charge": charge,
            "uhf": uhf, "accuracy": cfg.accuracy, "solvent": cfg.solvent, "model": cfg.solvation_model}


def relax_job(key, numbers, pos, movable, charge, cfg, lig_slice=None, lig_charge=0, steps=None, restraint=None,
              uhf=0):
    job = {"kind": "relax", "key": key, "numbers": numbers, "pos": pos, "movable": movable,
           "charge": charge, "uhf": uhf, "steps": cfg.relax_steps if steps is None else steps,
           "fmax": cfg.relax_fmax, "accuracy": cfg.relax_accuracy, "solvent": cfg.solvent, "model": cfg.solvation_model}
    if lig_slice:
        job["lig_slice"] = lig_slice
        job["lig_charge"] = lig_charge
    if restraint is not None:
        job["restraint"] = restraint
    return job


def strain_job(key, numbers, pos, heavy, charge, cfg):
    return relax_job(key, numbers, pos, np.arange(len(numbers)), charge, cfg, steps=cfg.strain_steps,
                     restraint=(heavy, cfg.strain_restraint))
