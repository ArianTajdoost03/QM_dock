from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from qmdock.chem.receptor import load_receptor
from qmdock.cli import build_parser, config_from_args
from qmdock.config import Config
from qmdock.search.buried import buriedness, inside_radius, seed_threshold

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2"


def hollow_sphere(radius=6.0, n=600):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def wall(spacing=1.5, half=15.0):
    g = np.arange(-half, half + 1e-6, spacing)
    return np.array([[x, y, 0.0] for x in g for y in g])


def test_a_point_inside_a_cavity_is_fully_buried_and_a_point_in_open_space_is_not():
    atoms = hollow_sphere()
    depth = buriedness(np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]), atoms)
    assert depth[0] > 0.95 and depth[1] == 0.0


def test_a_point_above_a_flat_surface_is_only_partly_buried():
    depth = buriedness(np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 9.0]]), wall())
    assert 0.2 < depth[0] < 0.45 and depth[1] < depth[0]


def test_buriedness_handles_empty_input():
    assert buriedness(np.zeros((0, 3)), wall()).shape == (0,)
    assert (buriedness(np.array([[0.0, 0.0, 0.0]]), np.zeros((0, 3))) == 0).all()


def test_the_seed_threshold_relaxes_for_a_shallow_site_and_never_exceeds_the_request():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    assert abs(seed_threshold(np.array([0.3, 0.3, 0.9]), points, [0.0, 0.0, 0.0], 0.6) - 0.2) < 1e-9
    assert seed_threshold(np.array([0.9, 0.9, 0.9]), points, [0.0, 0.0, 0.0], 0.6) == 0.6
    assert seed_threshold(np.array([0.9, 0.9, 0.9]), points, [50.0, 0.0, 0.0], 0.6) == 0.6
    assert seed_threshold(np.array([0.05, 0.05, 0.9]), points, [0.0, 0.0, 0.0], 0.6) == 0.0


def test_poses_whose_centroid_leaves_the_search_radius_are_flagged():
    poses = np.zeros((3, 4, 3))
    poses[1, :, 0] = 8.4
    poses[2, :, 0] = 9.0
    assert inside_radius(poses, [0.0, 0.0, 0.0], 8.0, 0.5).tolist() == [True, True, False]


def test_buriedness_enriches_the_seed_points_near_the_true_site_on_carbonic_anhydrase():
    rec = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False))), addCoords=True)
    coords = ligand.GetConformer().GetPositions()
    heavy = np.array([a.GetAtomicNum() > 1 for a in ligand.GetAtoms()])
    centre = coords[heavy].mean(axis=0)
    atoms = rec.coords[rec.elements != "H"]
    radius = 12.0
    g = np.arange(-radius, radius + 1e-6, 0.5)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    points = centre + grid[np.linalg.norm(grid, axis=1) <= radius]
    nearby = atoms[cKDTree(atoms).query_ball_point(centre, radius + 13.0)]
    voids = points[cKDTree(nearby).query(points)[0] >= 2.6]
    depth = buriedness(voids, nearby)
    threshold = seed_threshold(depth, voids, centre, 0.6)
    keep = depth >= threshold
    close = np.linalg.norm(voids - centre, axis=1) <= 3.0
    assert buriedness(centre[None], nearby)[0] > 0.6 and threshold == 0.6
    assert keep.sum() < 0.5 * len(voids)
    assert close[keep].mean() > 2.0 * close.mean()
    assert close[keep].sum() > 0.9 * close.sum()


def test_the_new_options_have_defaults_and_can_be_changed():
    cfg = Config(receptor="r", ligand="l")
    assert cfg.void_buriedness == 0.6 and cfg.radius_tolerance == 0.5
    args = build_parser("dock.py").parse_args(["--receptor", "r.pdb", "--ligand", "l.sdf", "--center", "0", "0", "0",
                                                "--void-buriedness", "0", "--radius-tolerance", "1.0"])
    changed = config_from_args(args, "l.sdf")
    assert changed.void_buriedness == 0 and changed.radius_tolerance == 1.0
