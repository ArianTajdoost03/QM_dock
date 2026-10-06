from dataclasses import replace

import numpy as np
from rdkit import Chem

from qmdock.pipeline.fast import Selection, pick_device
from qmdock.qm.pool import log
from qmdock.search.poses import PoseSet
from qmdock.search.scoring import Scorer

MAX_MAPPINGS = 64


def skeleton(mol):
    rw = Chem.RWMol()
    index = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            index[atom.GetIdx()] = rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in index and j in index:
            rw.AddBond(index[i], index[j], Chem.BondType.SINGLE)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def heavy_mappings(template, state):
    matches = skeleton(template).GetSubstructMatches(skeleton(state), uniquify=False, maxMatches=MAX_MAPPINGS)
    if not matches:
        raise ValueError("protonation states do not share a heavy-atom skeleton; run them separately")
    return np.array(matches, dtype=int)


def kabsch_batch(P, Q):
    pm = P.mean(axis=1, keepdims=True)
    qm = Q.mean(axis=0)
    Pc, Qc = P - pm, Q - qm
    H = np.einsum("cni,nj->cij", Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    V, Ut = Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)
    D = np.ones((len(P), 3))
    D[:, 2] = np.sign(np.linalg.det(V @ Ut))
    R = V @ (D[:, :, None] * Ut)
    moved = np.einsum("cij,cnj->cni", R, Pc) + qm
    rmsd = np.sqrt(((moved - Q) ** 2).sum(-1).mean(-1))
    return rmsd, R, pm[:, 0, :], qm


def transfer_pose(target_heavy, conf_heavy, conf_all, mappings):
    best = (np.inf, None, None, None)
    for m in mappings:
        rmsd, R, pm, qm = kabsch_batch(conf_heavy, target_heavy[m])
        c = int(np.argmin(rmsd))
        if rmsd[c] < best[0]:
            best = (float(rmsd[c]), c, R[c], (pm[c], qm))
    rmsd, c, R, (pm, qm) = best
    return rmsd, c, (conf_all[c] - pm) @ R.T + qm


def transfer_selection(cfg, receptor, lig_t, lig_s, sel_t, pool):
    import torch

    keys = sorted(lig_s.conf_e)
    mappings = heavy_mappings(lig_t.mol, lig_s.mol)
    s_heavy = np.where(lig_s.heavy)[0]
    t_heavy = np.where(lig_t.heavy)[0]
    conf_all = lig_s.conf_coords[keys]
    conf_heavy = conf_all[:, s_heavy, :]
    poses = sel_t.poses
    P = len(poses)
    coords = np.zeros((P, lig_s.n_atoms, 3))
    conf = np.zeros(P, dtype=int)
    rmsds = np.zeros(P)
    for p in range(P):
        rmsd, c, xyz = transfer_pose(poses.coords[p][t_heavy], conf_heavy, conf_all, mappings)
        coords[p], conf[p], rmsds[p] = xyz, keys[c], rmsd

    soft = replace(cfg, clash_heavy=cfg.clash_heavy * pool.clash_scale * 0.95,
                   clash_hydrogen=cfg.clash_hydrogen * pool.clash_scale * 0.95)
    device = pick_device(cfg.device)
    clashes = np.zeros(P, dtype=bool)
    for site in np.unique(poses.site):
        idx = np.where(poses.site == site)[0]
        env = receptor.environment(pool.centers[int(site)], cfg.search_radius + pool.extent + cfg.env_margin)
        scorer = Scorer(receptor.elements[env], receptor.coords[env], lig_s.elements, device, soft)
        step = max(1, scorer.chunk(False, True))
        with torch.no_grad():
            for s in range(0, len(idx), step):
                X = torch.as_tensor(coords[idx[s:s + step]], dtype=torch.float32, device=device)
                clashes[idx[s:s + step]] = scorer.evaluate(X, full=True)[1].cpu().numpy()

    rejected = {}
    for p in range(P):
        if rmsds[p] > cfg.transfer_rmsd:
            rejected[p] = f"heavy-atom RMSD {rmsds[p]:.2f}"
        elif clashes[p]:
            rejected[p] = "clash"
    new = PoseSet(coords, conf, poses.score.copy(), poses.site.copy())
    first = [p for p in sel_t.first if p not in rejected]
    log(f"  transferred {P - len(rejected)}/{P} poses (max heavy-atom RMSD {rmsds.max():.2f} A, "
        f"{sum(1 for r in rejected.values() if r == 'clash')} clashes)")
    info = {"transfer_rejected": len(rejected), "max_transfer_rmsd": float(rmsds.max())}
    return Selection(new, first, sel_t.groups, sel_t.dist, rejected), info
