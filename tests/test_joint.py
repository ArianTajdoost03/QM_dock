import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qmdock.chem.protonation import embed
from qmdock.pipeline.joint import heavy_mappings, kabsch_batch, transfer_pose


def build(smiles, seed):
    return embed(Chem.MolFromSmiles(smiles), seed)


def heavy_xyz(mol):
    pos = mol.GetConformer().GetPositions()
    return pos[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]]


def test_skeleton_mapping_across_charge_and_atom_order():
    neutral = build("O=c1cccccc1O", 1)
    anion = build("[O-]c1cccccc1=O", 2)
    maps = heavy_mappings(neutral, anion)
    assert maps.shape[1] == 9 and len(maps) >= 1


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(1, 9, 3))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Q = P[0] @ R.T + np.array([3.0, -1.0, 2.0])
    rmsd, *_ = kabsch_batch(P, Q)
    assert rmsd[0] < 1e-9


def test_transfer_places_new_state_on_the_template_pose():
    template = build("O=c1cccccc1O", 1)
    state = build("[O-]c1cccccc1=O", 5)
    maps = heavy_mappings(template, state)
    target = heavy_xyz(template) + np.array([10.0, 0.0, 0.0])
    s_heavy = np.where(np.array([a.GetAtomicNum() > 1 for a in state.GetAtoms()]))[0]
    conf_all = state.GetConformer().GetPositions()[None]
    rmsd, c, xyz = transfer_pose(target, conf_all[:, s_heavy, :], conf_all, maps)
    assert rmsd < 0.5 and xyz.shape == (state.GetNumAtoms(), 3)
