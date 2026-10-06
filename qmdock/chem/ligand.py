from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class Conformers:
    mol: Chem.Mol
    coords: np.ndarray
    energies: np.ndarray


def load_ligand(path):
    mol = next(iter(Chem.SDMolSupplier(path, removeHs=False)), None)
    if mol is None:
        raise ValueError(f"could not read ligand from {path}")
    return Chem.AddHs(mol, addCoords=True)


def build_conformers(mol, n, seed, window, prune, threads=0):
    work = Chem.Mol(mol)
    original = Chem.Conformer(work.GetConformer())
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune
    params.numThreads = threads
    params.useRandomCoords = True
    AllChem.EmbedMultipleConfs(work, n, params)
    work.AddConformer(original, assignId=True)
    if AllChem.MMFFHasAllMoleculeParams(work):
        result = AllChem.MMFFOptimizeMoleculeConfs(work, numThreads=threads, maxIters=2000)
    else:
        result = AllChem.UFFOptimizeMoleculeConfs(work, numThreads=threads, maxIters=2000)
    energies = np.array([e for _, e in result])
    order = [int(i) for i in np.argsort(energies) if energies[i] - energies.min() <= window]
    out = Chem.Mol(work)
    out.RemoveAllConformers()
    for i in order:
        out.AddConformer(Chem.Conformer(work.GetConformer(i)), assignId=True)
    coords = np.array([c.GetPositions() for c in out.GetConformers()])
    return Conformers(out, coords, energies[order])
