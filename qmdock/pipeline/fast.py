import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from qmdock.chem.receptor import find_surface_sites
from qmdock.pipeline.ligandprep import prepare_conformers
from qmdock.qm.pool import log
from qmdock.search.clustering import cluster_poses, pairwise_rmsd, representatives
from qmdock.search.poses import PoseSet, diverse
from qmdock.search.sampling import search
from qmdock.search.scoring import Scorer
from qmdock.search.torsions import TorsionModel


@dataclass
class Pool:
    poses: PoseSet
    centers: list
    extent: float
    clash_scale: float = 1.0


@dataclass
class Selection:
    poses: PoseSet
    first: list
    groups: list = None
    dist: np.ndarray = None
    rejected: dict = field(default_factory=dict)
    flex: bool = False


def pick_device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ligand_extent(lig):
    pos = lig.mol.GetConformer().GetPositions()
    return float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).max())


def centers_for(cfg, receptor, lig):
    centers = list(cfg.centers)
    if not centers and cfg.auto_sites:
        centers = find_surface_sites(receptor, cfg.auto_sites, cfg.search_radius * 1.5, cfg.seed)
    if not centers:
        centers = [tuple(lig.mol.GetConformer().GetPositions().mean(axis=0))]
        log(f"no site given, using ligand.sdf centroid {np.round(centers[0], 2)}")
    return centers


def fast_stage(cfg, receptor, lig, centers, warnings):
    device = pick_device(cfg.device)
    extent = ligand_extent(lig)
    log(f"fast stage on {device}, {len(centers)} site(s)")
    sets, scale_used = [], 1.0
    tors = TorsionModel(lig.rotors, lig.mol, device) if lig.rotors else None
    for i, c in enumerate(centers):
        env = receptor.environment(c, cfg.search_radius + extent + cfg.env_margin)
        scorer = Scorer(receptor.elements[env], receptor.coords[env], lig.elements, device, cfg)
        ps = search(scorer, lig.conf_coords, lig.heavy, np.array(c), cfg, cfg.seed + i, i, tors)
        for factor in (0.9, 0.8, 0.7):
            if len(ps):
                break
            log(f"  no clash-free pose; retrying with clash thresholds scaled by {factor}")
            soft = replace(cfg, clash_heavy=cfg.clash_heavy * factor, clash_hydrogen=cfg.clash_hydrogen * factor)
            scorer = Scorer(receptor.elements[env], receptor.coords[env], lig.elements, device, soft)
            ps = search(scorer, lig.conf_coords, lig.heavy, np.array(c), soft, cfg.seed + i, i, tors)
            if len(ps):
                scale_used = min(scale_used, factor)
                warnings.append(f"site {i}: poses found only with clash thresholds scaled by {factor}")
        d = ps.diag
        log(f"  site {i} {np.round(c, 1)}: {len(env)} env atoms, {len(ps)} clash-free diverse poses")
        log(f"    random poses clash-free: {100 * d['random_clash_free']:.1f}%, after optimisation: "
            f"{d['optimized_clash_free']}/{d['optimized']}")
        if d.get("mutation"):
            m = d["mutation"]
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            log(f"    mutation: {m['rounds']} round(s), {m['children']} children, "
                f"best fast score {fmt(m['best_before'])} -> {fmt(m['best_after'])}")
        if len(ps) == 0:
            worst = np.argsort(-d["partners"])[:8]
            log("    receptor atoms blocking most poses: " + ", ".join(
                f"{receptor.label(env[j])} ({int(d['partners'][j])})" for j in worst if d["partners"][j] > 0))
        sets.append(ps)
    return PoseSet.concat(sets), extent, scale_used


def save_cache(path, lig, pool):
    np.savez(path, conf_coords=lig.conf_coords, conf_keys=np.array(list(lig.conf_e)),
             conf_vals=np.array(list(lig.conf_e.values())), pool_coords=pool.coords, pool_conf=pool.conf,
             pool_score=pool.score, pool_site=pool.site)


def load_cache(path):
    z = np.load(path)
    conf_e = {int(k): float(v) for k, v in zip(z["conf_keys"], z["conf_vals"])}
    return z["conf_coords"], conf_e, PoseSet(z["pool_coords"], z["pool_conf"], z["pool_score"], z["pool_site"])


def get_pool(cfg, receptor, lig, runner, warnings, timings):
    path = os.path.join(cfg.out, "fast_stage.npz")
    centers = centers_for(cfg, receptor, lig)
    if cfg.reuse_fast and os.path.exists(path):
        coords, conf_e, poses = load_cache(path)
        lig = replace(lig, conf_coords=coords, conf_e=conf_e, e_ref=min(conf_e.values()))
        log(f"reusing fast stage from {path}: {len(poses)} poses, {len(conf_e)} conformers")
        return lig, Pool(poses, centers, ligand_extent(lig))
    lig, seconds = prepare_conformers(cfg, lig, runner)
    timings["prerelax"] = seconds
    t = time.time()
    poses, extent, scale = fast_stage(cfg, receptor, lig, centers, warnings)
    timings["fast"] = time.time() - t
    save_cache(path, lig, poses)
    if len(poses) == 0:
        raise RuntimeError("no clash-free poses found; enlarge --search-radius or relax the clash thresholds")
    return lig, Pool(poses, centers, extent, scale)


def select_poses(cfg, receptor, lig, pool):
    pool_poses = pool.poses
    order = np.argsort(pool_poses.score)
    groups = dist = None
    if cfg.selection == "cluster":
        poses = pool_poses.take([int(i) for i in order[:cfg.cluster_pool]])
        dist = pairwise_rmsd(poses.coords[:, lig.heavy, :], lig.perms)
        groups = cluster_poses(dist, poses.score, lig.family_radius)
        first = representatives(groups, cfg.n_qm)
        log(f"clustered {len(poses)} poses into {len(groups)} groups (radius {lig.family_radius:.2f} A); "
            f"{len(first)} representatives go to QM")
    else:
        keep = diverse(pool_poses.coords[:, lig.heavy, :], order, cfg.diversity_rmsd, cfg.n_qm)
        poses = pool_poses.take(keep)
        first = list(range(len(poses)))
    usable = sum(1 for p in first
                 if receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms) is not None)
    if usable == 0:
        raise RuntimeError("no eligible receptor unit or residue lies within the shell of any pose")
    log(f"selected {len(first)} poses for QM")
    return Selection(poses, first, groups, dist, flex=bool(lig.rotors))
