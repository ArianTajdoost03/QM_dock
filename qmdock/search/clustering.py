import numpy as np
from rdkit import Chem
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import cdist


def auto_radius(n_heavy):
    return float(np.clip(0.8 + 0.05 * n_heavy, 1.0, 2.0))


def symmetry_permutations(mol, limit=256):
    flat = Chem.RemoveHs(mol)
    matches = flat.GetSubstructMatches(flat, uniquify=False, useChirality=True, maxMatches=limit)
    return np.array(matches, dtype=int) if matches else np.arange(flat.GetNumAtoms())[None, :]


def sym_rmsd(a, b, perms):
    return float(min(np.sqrt(((a - b[p]) ** 2).sum(-1).mean()) for p in perms))


def pairwise_rmsd(coords, perms):
    n, k = coords.shape[0], coords.shape[1]
    flat = coords.reshape(n, -1)
    best = np.full((n, n), np.inf)
    for p in perms:
        d2 = cdist(flat, coords[:, p, :].reshape(n, -1), "sqeuclidean")
        best = np.minimum(best, d2)
    d = np.sqrt(best / k)
    return np.minimum(d, d.T)


def cluster_poses(dist, scores, radius):
    n = len(scores)
    if n == 1:
        return [{"members": [0], "medoid": 0, "best": 0}]
    condensed = [dist[i, j] for i in range(1, n) for j in range(i)]
    groups = Butina.ClusterData(condensed, n, radius, isDistData=True, reordering=True)
    out = []
    for g in groups:
        members = sorted(g, key=lambda i: scores[i])
        out.append({"members": [int(m) for m in members], "medoid": int(g[0]), "best": int(members[0])})
    return sorted(out, key=lambda c: scores[c["best"]])


def representatives(clusters, limit):
    picked = []
    for c in clusters:
        for p in (c["best"], c["medoid"]):
            if p not in picked:
                picked.append(p)
        if len(picked) >= limit:
            break
    return picked[:limit] if limit else picked


def spread_members(dist, scored, candidates, k):
    chosen, have = [], list(scored)
    pool = [c for c in candidates if c not in set(scored)]
    while pool and len(chosen) < k:
        gaps = [dist[c, have].min() if have else 0.0 for c in pool]
        pick = pool[int(np.argmax(gaps))]
        chosen.append(pick)
        have.append(pick)
        pool.remove(pick)
    return chosen
