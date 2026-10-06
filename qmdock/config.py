from dataclasses import dataclass, field

HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANG = 0.529177210903
VERSION = "0.7"


@dataclass
class Config:
    receptor: str
    ligand: str
    out: str = "run"
    centers: list = field(default_factory=list)
    auto_sites: int = 0
    search_radius: float = 8.0
    env_margin: float = 6.0
    ligand_charge: int = None
    n_conformers: int = 100
    conformer_window: float = 15.0
    conformer_prune: float = 0.5
    n_random: int = 50000
    n_optimize: int = 1000
    opt_steps: int = 100
    clash_heavy: float = 0.78
    clash_hydrogen: float = 0.65
    coordination_min: float = 1.8
    diversity_rmsd: float = 2.0
    n_qm: int = 30
    shell: float = 4.0
    max_cluster_atoms: int = 150
    final_top: int = 3
    final_max_atoms: int = 300
    min_unit_atoms: int = 10
    require_metal: bool = False
    keep_waters: bool = False
    protonation: str = "none"
    ph_min: float = 6.4
    ph_max: float = 8.4
    max_states: int = 4
    uhf: int = 0
    relax_top: int = 3
    relax_steps: int = 5
    relax_fmax: float = 0.002
    prerelax_steps: int = 50
    accuracy: float = 1.0
    solvent: str = None
    solvation_model: str = "alpb"
    selection: str = "greedy"
    cluster_radius: float = 0.0
    cluster_expand: int = 3
    expand_members: int = 5
    expand_window: float = 3.0
    cluster_pool: int = 300
    dry_run: bool = False
    reuse_fast: bool = False
    export_dft: int = 0
    joint_states: bool = False
    transfer_rmsd: float = 1.0
    mutation_rounds: int = 0
    mutation_parents: int = 100
    mutation_children: int = 10
    mutation_steps: int = 40
    mutation_rotation: float = 0.35
    mutation_shift: float = 0.6
    mutation_swap: float = 0.3
    mutation_dedupe: float = 0.3
    workers: int = 0
    device: str = "auto"
    seed: int = 7
    pair_budget: float = 6e7
