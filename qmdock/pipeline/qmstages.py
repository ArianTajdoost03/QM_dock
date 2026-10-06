import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from qmdock.config import HARTREE_TO_KCAL
from qmdock.qm.jobs import energy_job, numbers_of, relax_job
from qmdock.qm.pool import log
from qmdock.search.clustering import spread_members


@dataclass
class QMResult:
    stage1: dict
    relaxed: dict
    stage3: dict
    expanded: set
    cluster_of: dict
    coords: dict
    lig_e: dict
    seconds: float
    warnings: list = field(default_factory=list)


class Session:
    def __init__(self, cfg, receptor, runner):
        self.cfg = cfg
        self.receptor = receptor
        self.runner = runner
        self.pocket_energy = {}
        self.clusters = {}

    def evaluate(self, lig, pids, coord_of, lig_e_of, max_atoms):
        cfg, receptor = self.cfg, self.receptor
        infos, need = {}, {}
        for p in pids:
            cl = receptor.build_cluster(coord_of[p], cfg.shell, max_atoms)
            infos[p] = cl
            if cl is not None and cl.electrons() % 2 == 0 and cl.key not in self.pocket_energy:
                need[cl.key] = cl
        jobs = [energy_job(("pocket", k), numbers_of(cl.elements()), cl.coords(), cl.charge, cfg)
                for k, cl in need.items()]
        res = self.runner.map(jobs, f"pocket energies (<= {max_atoms} atoms)")
        for k, cl in need.items():
            r = res[("pocket", k)]
            self.pocket_energy[k] = r["energy"] if r["ok"] else None
            self.clusters[k] = cl
        jobs = []
        for p in pids:
            cl = infos[p]
            if cl is not None and self.pocket_energy.get(cl.key) is not None:
                numbers = np.concatenate([numbers_of(cl.elements()), lig.numbers])
                jobs.append(energy_job(("complex", p), numbers, np.vstack([cl.coords(), coord_of[p]]),
                                       cl.charge + lig.charge, cfg))
        res = self.runner.map(jobs, f"complex energies (<= {max_atoms} atoms)")
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
                e_pocket = self.pocket_energy[cl.key]
                row.update(status="ok", scf_attempt=r["attempt"], n_atoms=cl.n_atoms + lig.n_atoms,
                           n_residues=cl.n_residues, cluster_charge=cl.charge,
                           e_int=(r["energy"] - e_pocket - lig_e_of[p]) * HARTREE_TO_KCAL,
                           strain=(lig_e_of[p] - lig.e_ref) * HARTREE_TO_KCAL,
                           e_bind=(r["energy"] - e_pocket - lig.e_ref) * HARTREE_TO_KCAL, e_complex=r["energy"])
            rows[p] = row
        return rows

    def expand_clusters(self, lig, sel, stage1, coords, lig_e):
        cfg = self.cfg
        groups = sel.groups
        cluster_of = {m: i for i, g in enumerate(groups) for m in g["members"]}
        best_of = {}
        for i, g in enumerate(groups):
            vals = [stage1[m]["e_bind"] for m in g["members"] if stage1.get(m, {}).get("status") == "ok"]
            if vals:
                best_of[i] = min(vals)
        expanded = set()
        if best_of:
            lowest = min(best_of.values())
            expanded = set([i for i, e in sorted(best_of.items(), key=lambda kv: kv[1])
                            if e - lowest <= cfg.expand_window][:cfg.cluster_expand])
            extra = []
            for i in sorted(expanded):
                done = [m for m in groups[i]["members"] if m in stage1]
                extra += spread_members(sel.dist, done, groups[i]["members"], cfg.expand_members)
            extra = [p for p in extra if p in lig_e]
            log(f"expanding {len(expanded)} best cluster(s) with {len(extra)} more poses")
            if extra:
                stage1.update(self.evaluate(lig, extra, coords, lig_e, cfg.max_cluster_atoms))
        return cluster_of, expanded

    def run(self, lig, sel):
        cfg = self.cfg
        t = time.time()
        warnings = []
        poses = sel.poses
        P = len(poses)
        coords = {p: poses.coords[p] for p in range(P)}
        lig_e = {p: lig.conf_e[int(poses.conf[p])] for p in range(P)
                 if int(poses.conf[p]) in lig.conf_e and p not in sel.rejected}
        stage1 = self.evaluate(lig, [p for p in sel.first if p in lig_e], coords, lig_e, cfg.max_cluster_atoms)
        cluster_of, expanded = {}, set()
        if sel.groups:
            cluster_of, expanded = self.expand_clusters(lig, sel, stage1, coords, lig_e)
        ok1 = sorted([p for p, r in stage1.items() if r["status"] == "ok"], key=lambda p: stage1[p]["e_bind"])
        if not ok1:
            counts = dict(Counter(r["status"] for r in stage1.values()))
            raise RuntimeError(f"no pose could be scored by QM: {counts}. Check that all project files are from "
                               "the same version and that the receptor is protonated (explicit hydrogens)")
        log(f"stage 1 done: {len(ok1)} scored, best {stage1[ok1[0]]['e_bind']:.2f} kcal/mol")

        relaxed = {}
        jobs = []
        for p in ok1[:cfg.relax_top]:
            cl = stage1[p]["cluster"]
            numbers = np.concatenate([numbers_of(cl.elements()), lig.numbers])
            pos = np.vstack([cl.coords(), coords[p]])
            n_rec = cl.n_atoms
            jobs.append(relax_job(("relax", p), numbers, pos, np.arange(n_rec, len(numbers)),
                                  cl.charge + lig.charge, cfg, (n_rec, len(numbers)), lig.charge))
        for key, r in self.runner.map(jobs, "ligand relaxation").items():
            p = key[1]
            if not r["ok"]:
                warnings.append(f"relaxation failed for pose {p}: {r['error']}")
                continue
            cl = stage1[p]["cluster"]
            e_pocket = self.pocket_energy[cl.key]
            relaxed[p] = {"coords": r["pos"][cl.n_atoms:], "lig_e": r["lig_energy"],
                          "e_bind_relaxed": (r["energy"] - e_pocket - lig.e_ref) * HARTREE_TO_KCAL,
                          "e_int_relaxed": (r["energy"] - e_pocket - r["lig_energy"]) * HARTREE_TO_KCAL,
                          "strain_relaxed": (r["lig_energy"] - lig.e_ref) * HARTREE_TO_KCAL}

        order2 = sorted(ok1, key=lambda p: (p not in relaxed,
                                            relaxed[p]["e_bind_relaxed"] if p in relaxed else stage1[p]["e_bind"]))
        cand = order2[:cfg.final_top]
        coords_f = {p: relaxed[p]["coords"] if p in relaxed else coords[p] for p in cand}
        lig_f = {p: relaxed[p]["lig_e"] if p in relaxed else lig_e[p] for p in cand}
        stage3 = {}
        if cfg.final_max_atoms > cfg.max_cluster_atoms and cand:
            stage3 = self.evaluate(lig, cand, coords_f, lig_f, cfg.final_max_atoms)
        return QMResult(stage1, relaxed, stage3, expanded, cluster_of, coords, lig_e, time.time() - t, warnings)
