import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from qmdock.search.clustering import auto_radius, cluster_poses, pairwise_rmsd, representatives, spread_members, sym_rmsd, symmetry_permutations


def benzoic():
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    return m, Chem.RemoveHs(m).GetConformer().GetPositions()


def test_symmetry_makes_flipped_ring_identical():
    m, X = benzoic()
    perms = symmetry_permutations(m)
    flip = X.copy()
    flip[[0, 4]] = X[[4, 0]]
    flip[[1, 3]] = X[[3, 1]]
    assert np.sqrt(((X - flip) ** 2).sum(-1).mean()) > 0.5
    assert sym_rmsd(X, flip, perms) < 1e-6


def test_clusters_and_representatives():
    m, X = benzoic()
    perms = symmetry_permutations(m)
    poses = np.stack([X, X + 0.1, X + 5.0, X + 5.1, X + 10.0])
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    groups = cluster_poses(pairwise_rmsd(poses, perms), scores, 1.0)
    assert sorted(len(g["members"]) for g in groups) == [1, 2, 2]
    assert len(representatives(groups, 10)) >= 3


def test_spread_members_prefers_far_poses():
    d = np.array([[0, 1, 5], [1, 0, 4], [5, 4, 0.0]])
    assert spread_members(d, [0], [0, 1, 2], 1) == [2]


def test_auto_radius_bounds():
    assert auto_radius(5) == 1.05 or auto_radius(5) >= 1.0
    assert auto_radius(100) == 2.0
