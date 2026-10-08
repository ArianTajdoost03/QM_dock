import faulthandler
import sys

from qmdock.cli import build_parser, config_from_args


def main():
    faulthandler.enable()
    p = build_parser("QM-ranked screening of an SDF library against one pocket", with_ligand=False)
    p.add_argument("--ligands", required=True, help="one SDF file with all ligands")
    p.add_argument("--resume", action="store_true", help="skip ligands already finished in --out")
    p.add_argument("--retry-failed", action="store_true", help="with --resume, redo ligands that failed")
    p.add_argument("--max-heavy-atoms", type=int, default=60)
    p.add_argument("--rank-by", choices=("energy", "per-atom"), default="energy")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(out="screen_out", n_conformers=30, n_random=20000, n_optimize=300, opt_steps=60, n_qm=6,
                   relax_top=1, relax_steps=3, final_top=0, protonation="pka", max_states=3, joint_states=True)
    a = p.parse_args()
    if a.dry_run:
        p.error("--dry-run is not supported in screening")
    if not a.center and not a.auto_sites:
        p.error("a pocket is required: give --center X Y Z (or --auto-sites)")
    from qmdock.screening.runner import ScreenOptions, screen

    cfg = config_from_args(a, a.ligands)
    opts = ScreenOptions(a.ligands, a.resume, a.retry_failed, a.max_heavy_atoms, a.rank_by, a.top)
    try:
        _, interrupted = screen(cfg, opts)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(130 if interrupted else 0)
