import argparse
import faulthandler

from qmdock.chem.spin import parse_uhf
from qmdock.config import Config


def build_parser(description, with_ligand=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--receptor", required=True)
    if with_ligand:
        p.add_argument("--ligand", required=True)
    p.add_argument("--out", default="run")
    p.add_argument("--center", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"))
    p.add_argument("--auto-sites", type=int, default=0)
    p.add_argument("--search-radius", type=float, default=8.0)
    p.add_argument("--radius-tolerance", type=float, default=0.5)
    p.add_argument("--void-buriedness", type=float, default=0.6)
    p.add_argument("--no-chemical-score", dest="chemical_score", action="store_false")
    p.add_argument("--hbond-weight", type=float, default=0.587)
    p.add_argument("--hydrophobic-weight", type=float, default=0.0351)
    p.add_argument("--elec-weight", type=float, default=0.0)
    p.add_argument("--ligand-charge", type=int)
    p.add_argument("--n-conformers", type=int, default=100)
    p.add_argument("--n-random", type=int, default=50000)
    p.add_argument("--n-optimize", type=int, default=1000)
    p.add_argument("--prerelax-steps", type=int, default=100)
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--n-qm", type=int, default=30)
    p.add_argument("--shell", type=float, default=4.0)
    p.add_argument("--charged-shell", type=float, default=6.0)
    p.add_argument("--no-share-pockets", dest="share_pockets", action="store_false")
    p.add_argument("--max-cluster-atoms", type=int, default=0)
    p.add_argument("--final-top", type=int, default=6)
    p.add_argument("--final-max-atoms", type=int, default=0)
    p.add_argument("--relax-top", type=int, default=6)
    p.add_argument("--relax-steps", type=int, default=30)
    p.add_argument("--relax-max-atoms", type=int, default=200)
    p.add_argument("--relax-max-shift", type=float, default=1.5)
    p.add_argument("--ref-conformers", type=int, default=5)
    p.add_argument("--ref-steps", type=int, default=400)
    p.add_argument("--strain-restraint", type=float, default=0.2)
    p.add_argument("--strain-steps", type=int, default=100)
    p.add_argument("--keep-waters", action="store_true")
    p.add_argument("--no-add-hydrogens", dest="add_hydrogens", action="store_false")
    p.add_argument("--receptor-ph", type=float, default=7.4)
    p.add_argument("--protonation", choices=("none", "pka", "all"), default="none")
    p.add_argument("--ph-min", type=float, default=6.4)
    p.add_argument("--ph-max", type=float, default=8.4)
    p.add_argument("--max-states", type=int, default=4)
    p.add_argument("--uhf", type=parse_uhf, default="auto")
    p.add_argument("--relax-select", choices=("balanced", "e_bind"), default="balanced")
    p.add_argument("--pocket-mode", choices=("pose", "site"), default="pose")
    p.add_argument("--site-max-atoms", type=int, default=0)
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--relax-accuracy", type=float, default=1.0)
    p.add_argument("--state-ph", type=float, default=7.4)
    p.add_argument("--state-window", type=float, default=5.0)
    p.add_argument("--no-pka-penalty", action="store_true")
    p.add_argument("--pka", action="append", default=[])
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
    p.add_argument("--torsions", action="store_true")
    p.add_argument("--max-torsions", type=int, default=12)
    p.add_argument("--torsion-fraction", type=float, default=0.5)
    p.add_argument("--export-dft", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--sequential-states", action="store_true")
    p.add_argument("--grid-spacing", type=float, default=0.25)
    p.add_argument("--grid-shortlist", type=float, default=2.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    return p


def config_from_args(a, ligand):
    from qmdock.chem.pka import parse_overrides

    try:
        parse_overrides(a.pka)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if a.accuracy <= 0 or a.relax_accuracy <= 0:
        raise SystemExit("--accuracy and --relax-accuracy must be positive")
    return Config(receptor=a.receptor, ligand=ligand, out=a.out,
                 centers=[tuple(c) for c in a.center] if a.center else [], auto_sites=a.auto_sites,
                 search_radius=a.search_radius, radius_tolerance=a.radius_tolerance,
                 void_buriedness=a.void_buriedness, chemical_score=a.chemical_score,
                 hbond_weight=a.hbond_weight, hydrophobic_weight=a.hydrophobic_weight, elec_weight=a.elec_weight, ligand_charge=a.ligand_charge,
                 n_conformers=a.n_conformers, n_random=a.n_random, n_optimize=a.n_optimize, opt_steps=a.opt_steps, prerelax_steps=a.prerelax_steps,
                 n_qm=a.n_qm, shell=a.shell, charged_shell=a.charged_shell,
                 share_pockets=a.share_pockets, max_cluster_atoms=a.max_cluster_atoms, final_top=a.final_top,
                 final_max_atoms=a.final_max_atoms, relax_top=a.relax_top, relax_steps=a.relax_steps,
                 relax_max_atoms=a.relax_max_atoms, relax_max_shift=a.relax_max_shift,
                 ref_conformers=a.ref_conformers, ref_steps=a.ref_steps,
                 strain_restraint=a.strain_restraint, strain_steps=a.strain_steps,
                 uhf=a.uhf, require_metal=a.require_metal, keep_waters=a.keep_waters, add_hydrogens=a.add_hydrogens,
                 receptor_ph=a.receptor_ph, protonation=a.protonation, ph_min=a.ph_min,
                 ph_max=a.ph_max, max_states=a.max_states, state_ph=a.state_ph, state_window=a.state_window, accuracy=a.accuracy, relax_select=a.relax_select, pocket_mode=a.pocket_mode,
                 site_max_atoms=a.site_max_atoms, relax_accuracy=a.relax_accuracy,
                 pka_penalty=not a.no_pka_penalty, pka_overrides=a.pka, solvent=a.solvent,
                 solvation_model=a.solvation_model, selection=a.selection, cluster_radius=a.cluster_radius,
                 cluster_expand=a.cluster_expand, expand_members=a.expand_members,
                 expand_window=a.expand_window, cluster_pool=a.cluster_pool, dry_run=a.dry_run,
                 reuse_fast=a.reuse_fast, export_dft=a.export_dft, torsions=a.torsions, max_torsions=a.max_torsions,
                 torsion_fraction=a.torsion_fraction, joint_states=a.joint_states, transfer_rmsd=a.transfer_rmsd, mutation_rounds=a.mutation_rounds,
                 mutation_parents=a.mutation_parents, mutation_children=a.mutation_children, workers=a.workers, device=a.device, grid_spacing=a.grid_spacing, parallel_states=not a.sequential_states,
                 grid_shortlist=a.grid_shortlist,
                 seed=a.seed)


def main():
    faulthandler.enable()
    a = build_parser("QM-ranked docking on crystal or metalloprotein surfaces").parse_args()
    from qmdock.pipeline.run import run

    run(config_from_args(a, a.ligand))
