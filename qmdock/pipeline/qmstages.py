import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from qmdock.config import COST_EXPONENT, HARTREE_TO_KCAL
from qmdock.qm.jobs import energy_job, numbers_of, relax_job, strain_job
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


def pocket_members(cluster):
    return {("u", u) for u in cluster.key[0]} | {("r", r) for r in cluster.key[1]}


def pocket_cost(n_pocket, n_ligand):
    return (n_pocket + n_ligand) ** COST_EXPONENT + n_pocket ** COST_EXPONENT


def assign_pockets(receptor, cfg, pids, coord_of, cap, known, n_ligand):
    own = {p: receptor.build_cluster(coord_of[p], cfg.shell, cap, cfg.charged_shell) for p in pids}
    infos = dict(own)
    if not cfg.share_pockets:
        return infos
    groups = []
    for p in pids:
        if own[p] is None:
            continue
        alone = pocket_cost(own[p].n_atoms, n_ligand)
        best = None
        for g in groups:
            union = receptor.build_cluster(np.vstack([coord_of[q] for q in g["poses"] + [p]]),
                                           cfg.shell, cap, cfg.charged_shell)
            if union is None or union.electrons() % 2:
                continue
            shared = (len(g["poses"]) + 1) * (union.n_atoms + n_ligand) ** COST_EXPONENT + union.n_atoms ** COST_EXPONENT
            gain = g["separate"] + alone - shared
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, g, union)
        if best is None:
            groups.append({"poses": [p], "cluster": own[p], "separate": alone})
        else:
            _, g, union = best
            g["poses"].append(p)
            g["cluster"] = union
            g["separate"] += alone
    for g in groups:
        shared, k = g["cluster"], len(g["poses"])
        wanted = pocket_members(shared)
        saving = shared.n_atoms ** COST_EXPONENT
        reusable = [(k * ((c.n_atoms + n_ligand) ** COST_EXPONENT - (shared.n_atoms + n_ligand) ** COST_EXPONENT), c)
                    for c in known.values() if c.electrons() % 2 == 0 and wanted <= pocket_members(c)]
        reusable = [(extra, c) for extra, c in reusable if extra < saving]
        if reusable:
            shared = min(reusable, key=lambda item: item[0])[1]
        for p in g["poses"]:
            infos[p] = shared
    return infos


class Session:
    def __init__(self, cfg, receptor, runner):
        self.cfg = cfg
        self.receptor = receptor
        self.runner = runner
        self.pocket_energy = {}
        self.clusters = {}
        self.pocket_ids = {}

    def evaluate(self, lig, pids, coord_of, lig_e_of, max_atoms, lig_rel_of=None):
        cfg = self.cfg
        lig_rel_of = lig_e_of if lig_rel_of is None else lig_rel_of
        known = {k: c for k, c in self.clusters.items() if self.pocket_energy.get(k) is not None}
        infos = assign_pockets(self.receptor, cfg, pids, coord_of, max_atoms, known, lig.n_atoms)
        need = {}
        for p in pids:
            cl = infos[p]
            if cl is not None and cl.electrons() % 2 == 0 and cl.key not in self.pocket_energy:
                need[cl.key] = cl
        shared_by = Counter(cl.key for cl in infos.values() if cl is not None)
        jobs = [energy_job(("pocket", k), numbers_of(cl.elements()), cl.coords(), cl.charge, cfg)
                for k, cl in need.items()]
        res = self.runner.map(jobs, f"pocket energies ({len(need)} new pockets for {len(pids)} poses)")
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
        res = self.runner.map(jobs, f"complex energies ({len(jobs)} poses)")
        rows = {}
        for p in pids:
            cl = infos[p]
            r = res.get(("complex", p))
            row = {"cluster": cl, "skipped_units": cl.skipped if cl else 0,
                   "truncated_units": cl.truncated if cl else 0,
                   "pocket_id": self.pocket_ids.setdefault(cl.key, len(self.pocket_ids)) if cl else None,
                   "pocket_poses": shared_by[cl.key] if cl else 0}
            if cl is not None and cl.electrons() % 2:
                row["status"] = "odd_electron"
            elif r is None or not r["ok"]:
                row["status"] = "no_pocket" if r is None else "scf_failed"
            else:
                e_pocket = self.pocket_energy[cl.key]
                e_int = (r["energy"] - e_pocket - lig_e_of[p]) * HARTREE_TO_KCAL
                strain = (lig_rel_of[p] - lig.e_ref) * HARTREE_TO_KCAL
                strain_raw = (lig_e_of[p] - lig.e_ref) * HARTREE_TO_KCAL
                row.update(status="ok", scf_attempt=r["attempt"], n_atoms=cl.n_atoms + lig.n_atoms,
                           n_residues=cl.n_residues, cluster_charge=cl.charge, e_int=e_int, strain=strain,
                           strain_raw=strain_raw, e_bind=e_int + strain, e_bind_raw=e_int + strain_raw,
                           e_complex=r["energy"])
            rows[p] = row
        return rows

    def ligand_energies(self, lig, sel, pids, coords, lig_e, lig_rel):
        cfg = self.cfg
        heavy = np.where(lig.heavy)[0]
        if sel.flex:
            need = [p for p in pids if p not in lig_rel]
            jobs = [strain_job(("lig", p), lig.numbers, coords[p], heavy, lig.charge, cfg) for p in need]
            for key, r in self.runner.map(jobs, "ligand strain (restrained relaxation)").items():
                if r["ok"]:
                    lig_e[key[1]] = r["e_start"]
                    lig_rel[key[1]] = r["energy"]
            return
        conformer = {p: int(sel.poses.conf[p]) for p in pids}
        need = sorted({c for c in conformer.values() if c in lig.conf_e and c not in lig.conf_rel})
        jobs = [strain_job(("lig", c), lig.numbers, lig.conf_coords[c], heavy, lig.charge, cfg) for c in need]
        for key, r in self.runner.map(jobs, "ligand strain (restrained relaxation)").items():
            if r["ok"]:
                lig.conf_rel[key[1]] = r["energy"]
        for p, c in conformer.items():
            if c in lig.conf_rel:
                lig_e[p] = lig.conf_e[c]
                lig_rel[p] = lig.conf_rel[c]

    def relax_cluster(self, cluster, ligand_coords):
        cap = self.cfg.relax_cap
        if cap is None or cluster.n_atoms <= cap:
            return cluster
        small = self.receptor.build_cluster(ligand_coords, self.cfg.shell, cap, self.cfg.charged_shell)
        return cluster if small is None or small.electrons() % 2 else small

    def valid(self, sel, lig, p):
        return p not in sel.rejected and (sel.flex or int(sel.poses.conf[p]) in lig.conf_e)

    def expand_clusters(self, lig, sel, stage1, coords, lig_e, lig_rel):
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
            extra = [p for p in extra if self.valid(sel, lig, p)]
            self.ligand_energies(lig, sel, extra, coords, lig_e, lig_rel)
            extra = [p for p in extra if p in lig_rel]
            log(f"expanding {len(expanded)} best cluster(s) with {len(extra)} more poses")
            if extra:
                stage1.update(self.evaluate(lig, extra, coords, lig_e, cfg.cluster_cap, lig_rel))
        return cluster_of, expanded

    def run(self, lig, sel):
        cfg = self.cfg
        t = time.time()
        warnings = []
        poses = sel.poses
        P = len(poses)
        coords = {p: poses.coords[p] for p in range(P)}
        lig_e, lig_rel = {}, {}
        wanted = [p for p in (sel.first if sel.flex else range(P)) if self.valid(sel, lig, p)]
        self.ligand_energies(lig, sel, wanted, coords, lig_e, lig_rel)
        stage1 = self.evaluate(lig, [p for p in sel.first if p in lig_rel], coords, lig_e, cfg.cluster_cap, lig_rel)
        cluster_of, expanded = {}, set()
        if sel.groups:
            cluster_of, expanded = self.expand_clusters(lig, sel, stage1, coords, lig_e, lig_rel)
        ok1 = sorted([p for p, r in stage1.items() if r["status"] == "ok"], key=lambda p: stage1[p]["e_bind"])
        if not ok1:
            counts = dict(Counter(r["status"] for r in stage1.values()))
            raise RuntimeError(f"no pose could be scored by QM: {counts}. Check that the receptor "
                               "is protonated (explicit hydrogens)")
        log(f"stage 1 done: {len(ok1)} scored, best {stage1[ok1[0]]['e_bind']:.2f} kcal/mol")

        relaxed, jobs, atoms = {}, [], {}
        for p in ok1[:cfg.relax_top]:
            cl = self.relax_cluster(stage1[p]["cluster"], coords[p])
            numbers = np.concatenate([numbers_of(cl.elements()), lig.numbers])
            pos = np.vstack([cl.coords(), coords[p]])
            atoms[p] = cl.n_atoms
            jobs.append(relax_job(("relax", p), numbers, pos, np.arange(cl.n_atoms, len(numbers)),
                                  cl.charge + lig.charge, cfg, (cl.n_atoms, len(numbers)), lig.charge))
        for key, r in self.runner.map(jobs, "ligand relaxation").items():
            p = key[1]
            if not r["ok"]:
                warnings.append(f"relaxation failed for pose {p}: {r['error']}")
                continue
            new = r["pos"][atoms[p]:]
            shift = float(np.sqrt(((new[lig.heavy] - coords[p][lig.heavy]) ** 2).sum(axis=1).mean()))
            if shift > cfg.relax_max_shift:
                warnings.append(f"pose {p} moved {shift:.2f} A during relaxation (limit {cfg.relax_max_shift}); "
                                "relaxed geometry discarded")
                continue
            e_bind = stage1[p]["e_bind_raw"] + (r["energy"] - r["e_start"]) * HARTREE_TO_KCAL
            strain = (r["lig_energy"] - lig.e_ref) * HARTREE_TO_KCAL
            relaxed[p] = {"coords": new, "lig_e": r["lig_energy"], "e_bind_relaxed": e_bind,
                          "e_int_relaxed": e_bind - strain, "strain_relaxed": strain, "relax_shift": shift,
                          "relax_steps_used": r["steps"], "relax_atoms": atoms[p]}

        order2 = sorted(ok1, key=lambda p: (p not in relaxed,
                                            relaxed[p]["e_bind_relaxed"] if p in relaxed else stage1[p]["e_bind"]))
        cand = [p for p in order2[:cfg.final_top] if p in relaxed]
        coords_f = {p: relaxed[p]["coords"] for p in cand}
        lig_f = {p: relaxed[p]["lig_e"] for p in cand}
        stage3 = {}
        if cand:
            stage3 = self.evaluate(lig, cand, coords_f, lig_f, cfg.final_cap)
        return QMResult(stage1, relaxed, stage3, expanded, cluster_of, coords, lig_e, time.time() - t, warnings)
