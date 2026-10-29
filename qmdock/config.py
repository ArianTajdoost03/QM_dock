from dataclasses import dataclass, field

POSES_FILE = "poses.sdf"
RESULTS_FILE = "results.csv"
SUMMARY_FILE = "summary.json"
CLUSTERS_FILE = "clusters.csv"
STATES_FILE = "states.csv"
BEST_COMPLEX_FILE = "best_complex.xyz"
CONFIG_FILE = "config.json"
POOL_FILE = "fast_stage.npz"
PROTONATED_RECEPTOR_FILE = "receptor_h.pdb"
STATE_PREFIX = "state_"
OUTPUT_FILES = (POSES_FILE, RESULTS_FILE, SUMMARY_FILE, CLUSTERS_FILE, STATES_FILE, BEST_COMPLEX_FILE, CONFIG_FILE,
                PROTONATED_RECEPTOR_FILE)

HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANG = 0.529177210903
COST_EXPONENT = 2.8
LARGE_RADIUS = 10.0


@dataclass
class Config:
    receptor: str
    ligand: str
    out: str = "run"
    centers: list = field(default_factory=list)
    auto_sites: int = 0
    search_radius: float = 8.0
    radius_tolerance: float = 0.5
    void_buriedness: float = 0.6
    chemical_score: bool = True
    hbond_weight: float = 0.587
    hydrophobic_weight: float = 0.0351
    elec_weight: float = 0.0
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
    charged_shell: float = 6.0
    share_pockets: bool = True
    max_cluster_atoms: int = 0
    final_top: int = 6
    final_max_atoms: int = 0
    min_unit_atoms: int = 10
    require_metal: bool = False
    keep_waters: bool = False
    add_hydrogens: bool = True
    receptor_ph: float = 7.4
    protonation: str = "none"
    ph_min: float = 6.4
    ph_max: float = 8.4
    max_states: int = 4
    state_ph: float = 7.4
    state_window: float = 5.0
    pka_penalty: bool = True
    pka_overrides: list = field(default_factory=list)
    uhf: int | str = "auto"
    relax_select: str = "balanced"
    pocket_mode: str = "pose"
    site_max_atoms: int = 0
    relax_top: int = 6
    relax_steps: int = 30
    relax_fmax: float = 0.002
    relax_max_atoms: int = 200
    relax_max_shift: float = 1.5
    prerelax_steps: int = 50
    ref_conformers: int = 5
    ref_steps: int = 400
    strain_restraint: float = 0.2
    strain_steps: int = 100
    accuracy: float = 1.0
    relax_accuracy: float = 1.0
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
    torsions: bool = False
    max_torsions: int = 12
    torsion_fraction: float = 0.5
    torsion_lr: float = 0.15
    torsion_penalty: float = 0.03
    mutation_torsion: float = 0.5
    joint_states: bool = False
    parallel_states: bool = True
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
    grid_spacing: float = 0.25
    grid_far_spacing: float = 1.0
    grid_max_nodes: int = 5000000
    grid_shortlist: float = 2.0

    @property
    def cluster_cap(self):
        return self.max_cluster_atoms or None

    @property
    def final_cap(self):
        return self.final_max_atoms or None

    @property
    def relax_cap(self):
        return self.relax_max_atoms or None
