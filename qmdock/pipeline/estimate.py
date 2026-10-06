import time

import numpy as np

from qmdock.qm.jobs import energy_job, numbers_of
from qmdock.qm.engine import run_job
from qmdock.qm.pool import log


def report_dry_run(cfg, receptor, sel, lig, workers):
    poses, pids, n_lig = sel.poses, sel.first, lig.n_atoms
    pockets, complexes = {}, []
    for p in pids:
        cl = receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms)
        if cl is not None:
            pockets[cl.key] = cl
            complexes.append(cl.n_atoms)
    if not pockets:
        log("dry run: no pose has a receptor cluster")
        return
    smallest = min(pockets.values(), key=lambda c: c.n_atoms)
    t0 = time.time()
    res = run_job(energy_job(("cal", 0), numbers_of(smallest.elements()), smallest.coords(), smallest.charge, cfg))
    t_cal = res["seconds"] if res["ok"] else time.time() - t0
    n_cal = smallest.n_atoms

    def cost(n):
        return t_cal * (n / n_cal) ** 3.5

    sizes = sorted(c.n_atoms for c in pockets.values())
    pocket_t = sum(cost(n) for n in sizes)
    complex_t = sum(cost(n + n_lig) for n in complexes)
    extra = 0.0
    if cfg.selection == "cluster":
        extra = cfg.cluster_expand * cfg.expand_members * cost(float(np.mean(sizes)) + n_lig) / workers
    stage1 = (pocket_t + complex_t) / workers + extra
    avg_complex = float(np.mean(sizes)) + n_lig
    relax_t = -(-cfg.relax_top // workers) * cfg.relax_steps * 3 * cost(avg_complex) if cfg.relax_top else 0.0
    big = receptor.build_cluster(poses.coords[pids[0]], cfg.shell, cfg.final_max_atoms)
    final_t = 0.0
    if big is not None and cfg.final_max_atoms > cfg.max_cluster_atoms and cfg.final_top:
        final_t = cost(big.n_atoms) + -(-cfg.final_top // workers) * cost(big.n_atoms + n_lig)
    log(f"dry run: {len(pids)} poses, {len(pockets)} distinct pockets, pocket sizes {sizes[:8]}"
        f"{' ...' if len(sizes) > 8 else ''} atoms (+{n_lig} ligand)")
    log(f"  calibration: {n_cal}-atom pocket took {t_cal:.1f}s on one core")
    log(f"  estimated wall time with {workers} workers (rough, +-2x): stage 1 {stage1 / 60:.1f} min, "
        f"relaxation {relax_t / 60:.1f} min, stage 3 {final_t / 60:.1f} min, total {(stage1 + relax_t + final_t) / 60:.1f} min")
