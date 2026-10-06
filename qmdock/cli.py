import argparse
import faulthandler

from qmdock.config import Config


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser(description="QM-ranked docking on crystal or metalloprotein surfaces")
    p.add_argument("--receptor", required=True)
    p.add_argument("--ligand", required=True)
    p.add_argument("--out", default="run")
    p.add_argument("--center", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"))
    p.add_argument("--auto-sites", type=int, default=0)
    p.add_argument("--search-radius", type=float, default=8.0)
    p.add_argument("--ligand-charge", type=int)
    p.add_argument("--n-conformers", type=int, default=100)
    p.add_argument("--n-random", type=int, default=50000)
    p.add_argument("--n-optimize", type=int, default=1000)
    p.add_argument("--prerelax-steps", type=int, default=100)
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--n-qm", type=int, default=30)
    p.add_argument("--shell", type=float, default=4.0)
    p.add_argument("--max-cluster-atoms", type=int, default=150)
    p.add_argument("--final-top", type=int, default=3)
    p.add_argument("--final-max-atoms", type=int, default=300)
    p.add_argument("--relax-top", type=int, default=3)
    p.add_argument("--relax-steps", type=int, default=5)
    p.add_argument("--uhf", type=int, default=0)
    p.add_argument("--keep-waters", action="store_true")
    p.add_argument("--protonation", choices=("none", "pka", "all"), default="none")
    p.add_argument("--ph-min", type=float, default=6.4)
    p.add_argument("--ph-max", type=float, default=8.4)
    p.add_argument("--max-states", type=int, default=4)
    p.add_argument("--require-metal", action="store_true")
    p.add_argument("--solvent")
    p.add_argument("--solvation-model", choices=("alpb", "gbsa"), default="alpb")
    p.add_argument("--selection", choices=("greedy", "cluster"), default="greedy")
    p.add_argument("--cluster-radius", type=float, default=0.0)
    p.add_argument("--cluster-expand", type=int, default=3)
    p.add_argument("--expand-members", type=int, default=5)
    p.add_argument("--expand-window", type=float, default=3.0)
    p.add_argument("--cluster-pool", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reuse-fast", action="store_true")
    p.add_argument("--mutation-rounds", type=int, default=0)
    p.add_argument("--mutation-parents", type=int, default=100)
    p.add_argument("--mutation-children", type=int, default=10)
    p.add_argument("--joint-states", action="store_true")
    p.add_argument("--transfer-rmsd", type=float, default=1.0)
    p.add_argument("--export-dft", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    cfg = Config(receptor=a.receptor, ligand=a.ligand, out=a.out,
                 centers=[tuple(c) for c in a.center] if a.center else [], auto_sites=a.auto_sites,
                 search_radius=a.search_radius, ligand_charge=a.ligand_charge,
                 n_conformers=a.n_conformers, n_random=a.n_random, n_optimize=a.n_optimize, opt_steps=a.opt_steps, prerelax_steps=a.prerelax_steps,
                 n_qm=a.n_qm, shell=a.shell, max_cluster_atoms=a.max_cluster_atoms, final_top=a.final_top,
                 final_max_atoms=a.final_max_atoms, relax_top=a.relax_top, relax_steps=a.relax_steps,
                 uhf=a.uhf, require_metal=a.require_metal, keep_waters=a.keep_waters, protonation=a.protonation, ph_min=a.ph_min,
                 ph_max=a.ph_max, max_states=a.max_states, solvent=a.solvent,
                 solvation_model=a.solvation_model, selection=a.selection, cluster_radius=a.cluster_radius,
                 cluster_expand=a.cluster_expand, expand_members=a.expand_members,
                 expand_window=a.expand_window, cluster_pool=a.cluster_pool, dry_run=a.dry_run,
                 reuse_fast=a.reuse_fast, export_dft=a.export_dft, joint_states=a.joint_states, transfer_rmsd=a.transfer_rmsd, mutation_rounds=a.mutation_rounds,
                 mutation_parents=a.mutation_parents, mutation_children=a.mutation_children, workers=a.workers, device=a.device,
                 seed=a.seed)
    from qmdock.pipeline.run import run

    run(cfg)
