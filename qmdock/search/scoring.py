import torch

from qmdock.chem.elements import DONOR_ELEMENTS, METALS, VDW_RADII


class Scorer:
    def __init__(self, env_elements, env_coords, lig_elements, device, cfg):
        self.device = device
        self.env = torch.as_tensor(env_coords, dtype=torch.float32, device=device)
        lig_r = torch.tensor([VDW_RADII.get(e, 1.7) for e in lig_elements], device=device)
        env_r = torch.tensor([VDW_RADII.get(e, 1.7) for e in env_elements], device=device)
        self.rsum = lig_r[:, None] + env_r[None, :]
        lig_h = torch.tensor([e == "H" for e in lig_elements], device=device)
        env_h = torch.tensor([e == "H" for e in env_elements], device=device)
        lig_donor = torch.tensor([e in DONOR_ELEMENTS for e in lig_elements], device=device)
        env_metal = torch.tensor([e in METALS for e in env_elements], device=device)
        self.coord = lig_donor[:, None] & env_metal[None, :]
        scale = torch.where(lig_h[:, None] | env_h[None, :], cfg.clash_hydrogen, cfg.clash_heavy)
        self.dmin = torch.where(self.coord, torch.tensor(cfg.coordination_min, device=device),
                                scale * self.rsum)
        self.n_lig = len(lig_elements)
        self.n_env = len(env_elements)
        self.budget = cfg.pair_budget
        self.lh = torch.where(~lig_h)[0]
        self.eh = torch.where(~env_h)[0]
        sub = (self.lh[:, None], self.eh[None, :])
        self.rsum_h, self.coord_h, self.dmin_h = self.rsum[sub], self.coord[sub], self.dmin[sub]
        self.env_heavy = self.env[self.eh]

    def chunk(self, grad=False, full=True):
        pairs = self.n_lig * self.n_env if full else len(self.lh) * len(self.eh)
        return max(1, int(self.budget // (pairs * (6 if grad else 1))))

    def clash_partners(self, X):
        d = torch.cdist(X, self.env, compute_mode="donot_use_mm_for_euclid_dist")
        return (d < self.dmin).any(dim=1).sum(dim=0)

    def evaluate(self, X, full=True):
        if full:
            pos, env, rsum, coord_mask, dmin = X, self.env, self.rsum, self.coord, self.dmin
        else:
            pos, env = X[:, self.lh], self.env_heavy
            rsum, coord_mask, dmin = self.rsum_h, self.coord_h, self.dmin_h
        d = torch.cdist(pos, env, compute_mode="donot_use_mm_for_euclid_dist")
        s = d - rsum
        steric = (-0.0356 * torch.exp(-(s / 0.5) ** 2) - 0.00516 * torch.exp(-((s - 3.0) / 2.0) ** 2)
                  + 0.84 * torch.clamp(s, max=0.0) ** 2)
        coord = -1.0 * torch.exp(-((d - 2.1) / 0.35) ** 2) + 5.0 * torch.relu(1.8 - d) ** 2
        e = torch.where(coord_mask, coord, steric)
        clash = (d < dmin).flatten(1).any(dim=1)
        return e.flatten(1).sum(dim=1), clash


def free_points(scorer, center, radius, spacing=0.5, clearance=2.6):
    g = torch.arange(-radius, radius + 1e-6, spacing, device=scorer.device)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    pts = center + grid[grid.norm(dim=1) <= radius]
    d = torch.cat([torch.cdist(pts[i:i + 4096], scorer.env_heavy).min(dim=1).values
                   for i in range(0, len(pts), 4096)])
    return pts[d >= clearance]
