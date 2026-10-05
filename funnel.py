import json
from dataclasses import replace
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import numpy as np
import pandas as pd
from rdkit import Chem

from config import HARTREE_TO_KCAL, VERSION
from ligand import build_conformers, load_ligand
from metals import ATOMIC_NUMBER
from pdb_io import write_xyz
from clustering import (auto_radius, cluster_poses, pairwise_rmsd, representatives, spread_members, sym_rmsd,
                        symmetry_permutations)
from qm import init_worker, preflight, run_job
from receptor import find_surface_sites, load_receptor


def log(msg):
    print(msg, flush=True)


def default_workers():
    cores = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 1e6
        return max(1, min(cores, int(avail // 1.5)))
    except (OSError, StopIteration):
        return cores


class Runner:
    def __init__(self, workers, fn=run_job):
        self.workers = workers
        self.fn = fn
        self.pool = None

    def _new_pool(self, n):
        return ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=init_worker)

    def __enter__(self):
        if self.workers > 1:
            self.pool = self._new_pool(self.workers)
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.shutdown(cancel_futures=True)

    def isolated(self, job):
        with self._new_pool(1) as ex:
            try:
                return ex.submit(self.fn, job).result()
            except BrokenProcessPool:
                return {"ok": False, "error": "worker process crashed (segfault or killed)",
                        "key": job["key"], "seconds": 0.0}

    def map(self, jobs, label):
        if not jobs:
            return {}
        t0 = time.time()
        results = {}
        step = 1 if len(jobs) <= 100 else max(1, len(jobs) // 20)

        def record(r):
            results[r["key"]] = r
            if len(results) % step == 0 or len(results) == len(jobs):
                log(f"  {label}: {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s, last job {r['seconds']:.0f}s)")

        if self.pool is None:
            for job in jobs:
                record(self.fn(job))
            return results
        try:
            for r in self.pool.map(self.fn, jobs, chunksize=1):
                record(r)
        except BrokenProcessPool:
            log("  a worker process died (segfault or memory kill); rerunning unfinished jobs one per process")
            self.pool.shutdown(cancel_futures=True)
            self.pool = self._new_pool(self.workers)
            for job in [j for j in jobs if j["key"] not in results]:
                record(self.isolated(job))
        crashed = sum(1 for r in results.values() if not r["ok"] and "crashed" in r.get("error", ""))
        if crashed:
            log(f"  {label}: {crashed} job(s) crashed the QM engine")
        return results


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


def pick_device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def heavy_rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def report_dry_run(cfg, receptor, poses, pids, lig_numbers, lig_charge, workers):
    n_lig = len(lig_numbers)
    pockets = {}
    for p in pids:
        cl = receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms)
        if cl is not None:
            pockets[cl.key] = cl
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
    complex_t = sum(cost(pockets_n + n_lig) for pockets_n in
                    [receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms).n_atoms
                     for p in pids if receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms)])
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


def dock_one(cfg, mol):
    timings = {}
    warnings = []
    os.makedirs(cfg.out, exist_ok=True)
    workers = cfg.workers or default_workers()
    log(f"CPU workers: {workers}")
    from fastscore import PoseSet, Scorer, diverse, search

    t = time.time()
    receptor = load_receptor(cfg.receptor, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    for (formula, status), n in sorted(receptor.summary().items()):
        log(f"receptor unit {formula}: {n} x {status}")
    lig_elements = [a.GetSymbol() for a in mol.GetAtoms()]
    lig_numbers = numbers_of(lig_elements)
    heavy = np.array([e != "H" for e in lig_elements])
    lig_charge = Chem.GetFormalCharge(mol) if cfg.ligand_charge is None else cfg.ligand_charge
    if (int(lig_numbers.sum()) - lig_charge) % 2:
        raise ValueError("ligand has an odd electron count; set --ligand-charge to its real protonation state")
    perms = symmetry_permutations(mol)
    if perms.shape[1] != int(heavy.sum()):
        perms = np.arange(int(heavy.sum()))[None, :]
    family_radius = cfg.cluster_radius or auto_radius(int(heavy.sum()))
    log(f"ligand: {len(lig_elements)} atoms, charge {lig_charge}" + (f", solvent {cfg.solvent} ({cfg.solvation_model})" if cfg.solvent else ", gas phase"))
    timings["load"] = time.time() - t

    t = time.time()
    cache_path = os.path.join(cfg.out, "fast_stage.npz")
    cached = cfg.reuse_fast and os.path.exists(cache_path)
    if cached:
        z = np.load(cache_path)
        conf_e = {int(k): float(v) for k, v in zip(z["conf_keys"], z["conf_vals"])}
        e_ref = min(conf_e.values())
        pool_poses = PoseSet(z["pool_coords"], z["pool_conf"], z["pool_score"], z["pool_site"])
        log(f"reusing fast stage from {cache_path}: {len(pool_poses)} poses, {len(conf_e)} conformers")
    else:
        t = time.time()
        confs = build_conformers(mol, cfg.n_conformers, cfg.seed, cfg.conformer_window, cfg.conformer_prune)
        log(f"conformers: {len(confs.coords)}")
        timings["conformers"] = time.time() - t

        input_pos = mol.GetConformer().GetPositions()
        centers = list(cfg.centers)
        if not centers and cfg.auto_sites:
            centers = find_surface_sites(receptor, cfg.auto_sites, cfg.search_radius * 1.5, cfg.seed)
        if not centers:
            centers = [tuple(input_pos.mean(axis=0))]
            log(f"no site given, using ligand.sdf centroid {np.round(centers[0], 2)}")

        t = time.time()
        n_conf = len(confs.coords)
        conf_e = {}
        if cfg.prerelax_steps > 0:
            with Runner(workers) as runner:
                jobs = []
                for c in range(n_conf):
                    job = relax_job(("conf", c), lig_numbers, confs.coords[c], np.arange(len(lig_numbers)),
                                    lig_charge, cfg)
                    job["steps"] = cfg.prerelax_steps
                    jobs.append(job)
                res = runner.map(jobs, "conformer GFN2 pre-relaxation")
        else:
            with Runner(workers) as runner:
                res = runner.map([energy_job(("conf", c), lig_numbers, confs.coords[c], lig_charge, cfg)
                                  for c in range(n_conf)], "conformer energies")
        for c in range(n_conf):
            r = res[("conf", c)]
            if r["ok"]:
                conf_e[c] = r["energy"]
                if "pos" in r:
                    confs.coords[c] = r["pos"]
        if not conf_e:
            errors = sorted({r.get("error", "") for r in res.values() if not r["ok"]})
            raise RuntimeError(f"no conformer converged in GFN2: {errors[:3]}; run diagnose.py")
        e_ref = min(conf_e.values())
        log(f"ligand reference energy {e_ref:.6f} Ha ({len(conf_e)}/{n_conf} conformers converged)")
        n_conf_built = n_conf
        timings["prerelax"] = time.time() - t

        t = time.time()
        device = pick_device(cfg.device)
        extent = float(np.linalg.norm(input_pos - input_pos.mean(axis=0), axis=1).max())
        log(f"fast stage on {device}, {len(centers)} site(s)")
        sets = []
        for i, c in enumerate(centers):
            env = receptor.environment(c, cfg.search_radius + extent + cfg.env_margin)
            scorer = Scorer(receptor.elements[env], receptor.coords[env], lig_elements, device, cfg)
            ps = search(scorer, confs.coords, heavy, np.array(c), cfg, cfg.seed + i, i)
            for factor in (0.9, 0.8, 0.7):
                if len(ps):
                    break
                log(f"  no clash-free pose; retrying with clash thresholds scaled by {factor}")
                soft = replace(cfg, clash_heavy=cfg.clash_heavy * factor, clash_hydrogen=cfg.clash_hydrogen * factor)
                scorer = Scorer(receptor.elements[env], receptor.coords[env], lig_elements, device, soft)
                ps = search(scorer, confs.coords, heavy, np.array(c), soft, cfg.seed + i, i)
                if len(ps):
                    warnings.append(f"site {i}: poses found only with clash thresholds scaled by {factor}")
            log(f"  site {i} {np.round(c, 1)}: {len(env)} env atoms, {len(ps)} clash-free diverse poses")
            d = ps.diag
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
        pool_poses = PoseSet.concat(sets)
        np.savez(cache_path, conf_coords=confs.coords, conf_keys=np.array(list(conf_e)),
                 conf_vals=np.array(list(conf_e.values())), pool_coords=pool_poses.coords,
                 pool_conf=pool_poses.conf, pool_score=pool_poses.score, pool_site=pool_poses.site)
    if len(pool_poses) == 0:
        raise RuntimeError("no clash-free poses found; enlarge --search-radius or relax the clash thresholds")
    order = np.argsort(pool_poses.score)
    groups = None
    if cfg.selection == "cluster":
        poses = pool_poses.take([int(i) for i in order[:cfg.cluster_pool]])
        dist = pairwise_rmsd(poses.coords[:, heavy, :], perms)
        groups = cluster_poses(dist, poses.score, family_radius)
        first = representatives(groups, cfg.n_qm)
        log(f"clustered {len(poses)} poses into {len(groups)} groups (radius {family_radius:.2f} A); "
            f"{len(first)} representatives go to QM")
    else:
        keep = diverse(pool_poses.coords[:, heavy, :], order, cfg.diversity_rmsd, cfg.n_qm)
        poses = pool_poses.take(keep)
        first = list(range(len(poses)))
    timings["fast"] = time.time() - t
    usable = sum(1 for p in first
                 if receptor.build_cluster(poses.coords[p], cfg.shell, cfg.max_cluster_atoms) is not None)
    if usable == 0:
        raise RuntimeError("no eligible receptor unit or residue lies within the shell of any pose")
    log(f"selected {len(first)} poses for QM")
    if cfg.dry_run:
        report_dry_run(cfg, receptor, poses, first, lig_numbers, lig_charge, workers)
        return None

    t = time.time()
    clusters = {}
    pocket_energy = {}

    def evaluate(runner, pids, coord_of, lig_e_of, max_atoms):
        infos, need = {}, {}
        for p in pids:
            cl = receptor.build_cluster(coord_of[p], cfg.shell, max_atoms)
            infos[p] = cl
            if cl is not None and cl.electrons() % 2 == 0 and cl.key not in pocket_energy:
                need[cl.key] = cl
        jobs = [energy_job(("pocket", k), numbers_of(cl.elements()), cl.coords(), cl.charge, cfg)
                for k, cl in need.items()]
        res = runner.map(jobs, f"pocket energies (<= {max_atoms} atoms)")
        for k, cl in need.items():
            r = res[("pocket", k)]
            pocket_energy[k] = r["energy"] if r["ok"] else None
            clusters[k] = cl
        jobs = []
        for p in pids:
            cl = infos[p]
            if cl is not None and pocket_energy.get(cl.key) is not None:
                numbers = np.concatenate([numbers_of(cl.elements()), lig_numbers])
                jobs.append(energy_job(("complex", p), numbers, np.vstack([cl.coords(), coord_of[p]]),
                                       cl.charge + lig_charge, cfg))
        res = runner.map(jobs, f"complex energies (<= {max_atoms} atoms)")
        rows = {}
        for p in pids:
            cl = infos[p]
            r = res.get(("complex", p))
            row = {"cluster": cl, "skipped_units": cl.skipped if cl else 0,
                   "truncated_units": cl.truncated if cl else 0}
            if cl is not None and cl.electrons() % 2:
                row["status"] = "odd_electron"
            elif r is None or not r["ok"]:
                row["status"] = "no_pocket" if r is None else "scf_failed"
            else:
                e_pocket = pocket_energy[cl.key]
                row.update(status="ok", scf_attempt=r["attempt"], n_atoms=cl.n_atoms + len(lig_numbers),
                           n_residues=cl.n_residues, cluster_charge=cl.charge,
                           e_int=(r["energy"] - e_pocket - lig_e_of[p]) * HARTREE_TO_KCAL,
                           strain=(lig_e_of[p] - e_ref) * HARTREE_TO_KCAL,
                           e_bind=(r["energy"] - e_pocket - e_ref) * HARTREE_TO_KCAL, e_complex=r["energy"])
            rows[p] = row
        return rows

    P = len(poses)
    coords = {p: poses.coords[p] for p in range(P)}
    with Runner(workers) as runner:
        lig_e = {p: conf_e[int(poses.conf[p])] for p in range(P) if int(poses.conf[p]) in conf_e}
        stage1 = evaluate(runner, [p for p in first if p in lig_e], coords, lig_e, cfg.max_cluster_atoms)
        cluster_of, expanded = {}, set()
        if groups:
            cluster_of = {m: i for i, g in enumerate(groups) for m in g["members"]}
            best_of = {}
            for i, g in enumerate(groups):
                vals = [stage1[m]["e_bind"] for m in g["members"] if stage1.get(m, {}).get("status") == "ok"]
                if vals:
                    best_of[i] = min(vals)
            if best_of:
                lowest = min(best_of.values())
                expanded = set([i for i, e in sorted(best_of.items(), key=lambda kv: kv[1])
                                if e - lowest <= cfg.expand_window][:cfg.cluster_expand])
                extra = []
                for i in sorted(expanded):
                    done = [m for m in groups[i]["members"] if m in stage1]
                    extra += spread_members(dist, done, groups[i]["members"], cfg.expand_members)
                extra = [p for p in extra if p in lig_e]
                log(f"expanding {len(expanded)} best cluster(s) with {len(extra)} more poses")
                if extra:
                    stage1.update(evaluate(runner, extra, coords, lig_e, cfg.max_cluster_atoms))
        ok1 = sorted([p for p, r in stage1.items() if r["status"] == "ok"], key=lambda p: stage1[p]["e_bind"])
        if not ok1:
            from collections import Counter

            counts = dict(Counter(r["status"] for r in stage1.values()))
            raise RuntimeError(f"no pose could be scored by QM: {counts}. Check that all project files are from "
                               "the same version and that the receptor is protonated (explicit hydrogens)")
        log(f"stage 1 done: {len(ok1)} scored, best {stage1[ok1[0]]['e_bind']:.2f} kcal/mol" if ok1 else "stage 1: none scored")

        relaxed = {}
        jobs = []
        for p in ok1[:cfg.relax_top]:
            cl = stage1[p]["cluster"]
            numbers = np.concatenate([numbers_of(cl.elements()), lig_numbers])
            pos = np.vstack([cl.coords(), coords[p]])
            n_rec = cl.n_atoms
            jobs.append(relax_job(("relax", p), numbers, pos, np.arange(n_rec, len(numbers)),
                                  cl.charge + lig_charge, cfg, (n_rec, len(numbers)), lig_charge))
        for key, r in runner.map(jobs, "ligand relaxation").items():
            p = key[1]
            if not r["ok"]:
                warnings.append(f"relaxation failed for pose {p}: {r['error']}")
                continue
            cl = stage1[p]["cluster"]
            n_rec = cl.n_atoms
            e_pocket = pocket_energy[cl.key]
            relaxed[p] = {"coords": r["pos"][n_rec:], "lig_e": r["lig_energy"],
                          "e_bind_relaxed": (r["energy"] - e_pocket - e_ref) * HARTREE_TO_KCAL,
                          "e_int_relaxed": (r["energy"] - e_pocket - r["lig_energy"]) * HARTREE_TO_KCAL,
                          "strain_relaxed": (r["lig_energy"] - e_ref) * HARTREE_TO_KCAL}

        order2 = sorted(ok1, key=lambda p: (p not in relaxed,
                                            relaxed[p]["e_bind_relaxed"] if p in relaxed else stage1[p]["e_bind"]))
        cand = order2[:cfg.final_top]
        coords_f = {p: relaxed[p]["coords"] if p in relaxed else coords[p] for p in cand}
        lig_f = {p: relaxed[p]["lig_e"] if p in relaxed else lig_e[p] for p in cand}
        stage3 = {}
        if cfg.final_max_atoms > cfg.max_cluster_atoms and cand:
            stage3 = evaluate(runner, cand, coords_f, lig_f, cfg.final_max_atoms)
    timings["qm"] = time.time() - t

    if groups:
        table = []
        for i, g in enumerate(groups):
            vals = [stage1[m]["e_bind"] for m in g["members"] if stage1.get(m, {}).get("status") == "ok"]
            table.append({"cluster": i, "size": len(g["members"]), "n_scored": len(vals),
                          "best_fast_score": float(poses.score[g["best"]]), "medoid_pose": g["medoid"],
                          "best_pose": g["best"], "best_qm_e_bind": min(vals) if vals else np.nan,
                          "expanded": i in expanded})
        pd.DataFrame(table).to_csv(os.path.join(cfg.out, "clusters.csv"), index=False)

    rows = []
    for p in range(P):
        s1 = stage1.get(p, {"status": "not_evaluated"})
        row = {"pose": p, "site": int(poses.site[p]), "conformer": int(poses.conf[p]),
               "fast_score": float(poses.score[p]), "status": s1["status"]}
        if groups:
            row["cluster"] = cluster_of.get(p)
        for k in ("n_atoms", "skipped_units", "truncated_units", "scf_attempt", "e_int", "strain", "e_bind"):
            if k in s1:
                row[k] = s1[k]
        if p in relaxed:
            row.update({k: v for k, v in relaxed[p].items() if k != "coords" and k != "lig_e"})
        s3 = stage3.get(p)
        if s3 and s3["status"] == "ok":
            row.update(e_int_final=s3["e_int"], strain_final=s3["strain"], e_bind_final=s3["e_bind"],
                       n_atoms_final=s3["n_atoms"], truncated_final=s3["truncated_units"])
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("e_bind_relaxed", "e_bind_final"):
        if col not in df:
            df[col] = np.nan
    ok = df[df.status == "ok"].copy()
    ok["level"] = np.where(ok.e_bind_final.notna(), 3, np.where(ok.e_bind_relaxed.notna(), 2, 1))
    ok["e_final"] = ok.e_bind_final.fillna(ok.e_bind_relaxed).fillna(ok.e_bind)
    ok = ok.sort_values(["level", "e_final"], ascending=[False, True])
    ranked = pd.concat([ok, df[df.status != "ok"]])
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)

    final_coords = {p: relaxed[p]["coords"] if p in relaxed else coords[p] for p in range(P)}
    top = int(ranked.iloc[0].pose)
    ranked["rmsd_to_top"] = [sym_rmsd(final_coords[int(p)][heavy], final_coords[top][heavy], perms)
                             for p in ranked.pose]
    ranked.to_csv(os.path.join(cfg.out, "results.csv"), index=False)

    writer = Chem.SDWriter(os.path.join(cfg.out, "poses.sdf"))
    for _, row in ranked[ranked.status == "ok"].iterrows():
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        conf = Chem.Conformer(len(lig_elements))
        for i, xyz in enumerate(final_coords[int(row.pose)]):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        m.AddConformer(conf)
        m.SetProp("_Name", f"pose_{int(row.pose)}")
        for key in ("final_rank", "e_final", "e_int", "strain", "fast_score", "rmsd_to_top"):
            m.SetProp(key, f"{float(row[key]):.4f}")
        writer.write(m)
    writer.close()

    best = ranked.iloc[0]
    best_cl = (stage3.get(top) if stage3.get(top, {}).get("status") == "ok" else stage1[top])["cluster"]
    write_xyz(os.path.join(cfg.out, "best_complex.xyz"), best_cl.elements() + lig_elements,
              np.vstack([best_cl.coords(), final_coords[top]]),
              f"best pose {top} e_final {best.e_final:.2f} kcal/mol")

    if cfg.export_dft:
        top_ids = [int(p) for p in ranked[(ranked.status == "ok") & (ranked.level == best.level)].pose[:cfg.export_dft]]
        for p in top_ids:
            cl = (stage3.get(p) if stage3.get(p, {}).get("status") == "ok" else stage1[p])["cluster"]
            d = os.path.join(cfg.out, "dft_export", f"pose_{p}")
            os.makedirs(d, exist_ok=True)
            write_xyz(os.path.join(d, "complex.xyz"), cl.elements() + lig_elements,
                      np.vstack([cl.coords(), final_coords[p]]), f"pose {p}")
            write_xyz(os.path.join(d, "pocket.xyz"), cl.elements(), cl.coords(), f"pocket for pose {p}")
            write_xyz(os.path.join(d, "ligand.xyz"), lig_elements, final_coords[p], f"ligand pose {p}")
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"pose": p, "charge_complex": cl.charge + lig_charge, "charge_pocket": cl.charge,
                           "charge_ligand": lig_charge, "xtb_e_final_kcal": float(ranked[ranked.pose == p].e_final.iloc[0])},
                          f, indent=2)
        log(f"exported {len(top_ids)} pose(s) for DFT to {os.path.join(cfg.out, 'dft_export')}")

    distinct = ranked[(ranked.status == "ok") & (ranked.level == best.level) & (ranked.rmsd_to_top > family_radius)]
    gap = float(distinct.e_final.min() - best.e_final) if len(distinct) else None
    summary = {"best_pose": top, "best_e_final_kcal": float(best.e_final), "best_level": int(best.level),
               "gap_to_next_family_kcal": gap, "n_poses_qm": len(stage1), "n_failed": int((df.status == "scf_failed").sum()),
               "timings_s": timings, "warnings": warnings, "ligand_charge": lig_charge,
               "solvent": cfg.solvent, "selection": cfg.selection, "n_clusters": len(groups) if groups else None,
               "n_poses_evaluated": len(stage1)}
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"best pose {top}: {best.e_final:.2f} kcal/mol (level {int(best.level)}), gap to next family {gap}")
    return ranked


def run(cfg):
    from ligand import ligand_states

    log(f"qmdock {VERSION}")
    preflight(cfg.solvent, cfg.solvation_model)
    base = load_ligand(cfg.ligand)
    if not cfg.centers and not cfg.auto_sites:
        cfg = replace(cfg, centers=[tuple(base.GetConformer().GetPositions().mean(axis=0))])
    states = ligand_states(base, cfg.protonation, cfg.ph_min, cfg.ph_max, cfg.max_states, cfg.seed)
    if len(states) == 1:
        return dock_one(cfg, states[0]["mol"])
    os.makedirs(cfg.out, exist_ok=True)
    for i, st in enumerate(states):
        log(f"state {i}: {st['smiles']} (charge {st['charge']}, {st['label']})")
    rows = []
    for i, st in enumerate(states):
        sub = replace(cfg, out=os.path.join(cfg.out, f"state_{i}"), ligand_charge=None)
        row = {"state": i, "smiles": st["smiles"], "charge": st["charge"], "label": st["label"], "dir": sub.out}
        log(f"=== protonation state {i}/{len(states) - 1}: {st['smiles']} ===")
        try:
            dock_one(sub, st["mol"])
            if cfg.dry_run:
                rows.append(row)
                continue
            with open(os.path.join(sub.out, "summary.json")) as f:
                summary = json.load(f)
            row.update(best_pose=summary["best_pose"], best_e_final_kcal=summary["best_e_final_kcal"],
                       best_level=summary["best_level"], gap_to_next_family_kcal=summary["gap_to_next_family_kcal"])
        except (RuntimeError, ValueError) as exc:
            row["error"] = str(exc)[:300]
            log(f"  state {i} failed: {row['error']}")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(cfg.out, "states.csv"), index=False)
    log("energies of different protonation states are not comparable (different numbers of protons); "
        "rank poses within a state and compare states by geometry or with pKa corrections")
    return table
