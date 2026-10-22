import numpy as np
from scipy.spatial import cKDTree

RAYS = 32
RAY_LENGTH = 10.0
RAY_STEP = 1.0
ATOM_RADIUS = 1.8
VOXEL = 1.0
CHUNK = 1000


def ray_directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def buriedness(points, atoms, rays=RAYS, length=RAY_LENGTH):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    atoms = np.asarray(atoms, dtype=float).reshape(-1, 3)
    if len(points) == 0 or len(atoms) == 0:
        return np.zeros(len(points))
    origin = points.min(axis=0) - length - VOXEL
    shape = np.ceil((points.max(axis=0) + length + VOXEL - origin) / VOXEL).astype(int) + 1
    axes = [origin[k] + VOXEL * np.arange(shape[k]) for k in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied = (cKDTree(atoms).query(centers, distance_upper_bound=ATOM_RADIUS)[0] <= ATOM_RADIUS).reshape(shape)
    steps = np.arange(RAY_STEP, length + 1e-9, RAY_STEP)
    directions = ray_directions(rays)
    depth = np.empty(len(points))
    for start in range(0, len(points), CHUNK):
        block = points[start:start + CHUNK]
        sample = block[:, None, None, :] + directions[None, :, None, :] * steps[None, None, :, None]
        index = np.clip(np.rint((sample - origin) / VOXEL).astype(int), 0, shape - 1)
        hit = occupied[index[..., 0], index[..., 1], index[..., 2]]
        depth[start:start + CHUNK] = hit.any(axis=2).mean(axis=1)
    return depth


def inside_radius(coords, center, radius, tolerance):
    centroid = np.asarray(coords).mean(axis=1)
    return np.linalg.norm(centroid - np.asarray(center, dtype=float), axis=1) <= radius + tolerance


def seed_threshold(depth, points, center, wanted, radius=3.0, margin=0.1):
    near = np.linalg.norm(points - np.asarray(center, dtype=float), axis=1) <= radius
    if not near.any():
        return wanted
    return max(0.0, min(wanted, float(np.median(depth[near])) - margin))
