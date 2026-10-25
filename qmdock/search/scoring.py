from types import SimpleNamespace

import torch

from qmdock.chem.elements import DONOR_ELEMENTS, METALS, VDW_RADII
from qmdock.search.terms import pair_energy


class Scorer:
    def __init__(self, env_elements, env_coords, lig_elements, device, cfg, lig_types=None, env_types=None):
        self.device = device
        self.env = torch.as_tensor(env_coords, dtype=torch.float32, device=device)
        lig_r = torch.tensor([VDW_RADII.get(e, 1.7) for e in lig_elements], device=device)
        env_r = torch.tensor([VDW_RADII.get(e, 1.7) for e in env_elements], device=device)
        self.rsum = lig_r[:, None] + env_r[None, :]
        lig_h = torch.tensor([e == "H" for e in lig_elements], device=device)
        env_h = torch.tensor([e == "H" for e in env_elements], device=device)
        lig_donor = torch.tensor([e in DONOR_ELEMENTS for e in lig_elements], device=device)
        env_metal = torch.tensor([e in METALS for e in env_elements], device=device)
        typed = lig_types is not None and env_types is not None
        weight = torch.as_tensor(lig_types.metal_weight, dtype=torch.float32, device=device) if typed else lig_donor.float()
        self.coord = (weight > 0)[:, None] & env_metal[None, :]
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
        self.pair_full = self._pair(cfg, weight, lig_types if typed else None, env_types if typed else None,
                                    env_metal, None, None)
        self.pair_heavy = self._pair(cfg, weight, lig_types if typed else None, env_types if typed else None,
                                     env_metal, self.lh, self.eh)
        terms = (self.pair_full.hbond, self.pair_full.hydrophobic, self.pair_full.charges)
        self.load = 1.0 + 0.5 * sum(t is not None for t in terms)

    def _pair(self, cfg, weight, lig_types, env_types, env_metal, lig_idx, env_idx):
        def lv(v):
            return v if lig_idx is None else v[lig_idx]

        def ev(v):
            return v if env_idx is None else v[env_idx]

        def mat(m):
            return m if lig_idx is None else m[lig_idx[:, None], env_idx[None, :]]

        pair = SimpleNamespace(rsum=mat(self.rsum), coord=mat(self.coord), dmin=mat(self.dmin), metal_weight=lv(weight),
                               metal_cols=None, metal_free=None, hbond=None, hydrophobic=None, charges=None,
                               w_hbond=cfg.hbond_weight, w_hydrophobic=cfg.hydrophobic_weight, w_elec=cfg.elec_weight)
        metals = torch.where(ev(env_metal))[0]
        if len(metals):
            pair.metal_cols = metals
        if lig_types is None:
            return pair
        as_bool = lambda a: torch.as_tensor(a, dtype=torch.bool, device=self.device)
        as_float = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        if len(metals):
            pair.metal_free = ev(as_float(env_types.metal_free))[metals]
        if cfg.hbond_weight:
            donor_l, acceptor_l = lv(as_bool(lig_types.donor)), lv(as_bool(lig_types.acceptor))
            donor_e, acceptor_e = ev(as_bool(env_types.donor)), ev(as_bool(env_types.acceptor))
            pair.hbond = ((donor_l[:, None] & acceptor_e[None, :]) | (acceptor_l[:, None] & donor_e[None, :])).float()
        if cfg.hydrophobic_weight:
            pair.hydrophobic = (lv(as_bool(lig_types.hydrophobic))[:, None]
                                & ev(as_bool(env_types.hydrophobic))[None, :]).float()
        if cfg.elec_weight:
            pair.charges = lv(as_float(lig_types.charge))[:, None] * ev(as_float(env_types.charge))[None, :]
        return pair

    def chunk(self, grad=False, full=True):
        pairs = self.n_lig * self.n_env if full else len(self.lh) * len(self.eh)
        return max(1, int(self.budget // (pairs * (6 if grad else 1) * self.load)))

    def clash_partners(self, X):
        d = torch.cdist(X, self.env, compute_mode="donot_use_mm_for_euclid_dist")
        return (d < self.dmin).any(dim=1).sum(dim=0)

    def evaluate(self, X, full=True):
        if full:
            pos, env, pair = X, self.env, self.pair_full
        else:
            pos, env, pair = X[:, self.lh], self.env_heavy, self.pair_heavy
        d = torch.cdist(pos, env, compute_mode="donot_use_mm_for_euclid_dist")
        clash = (d < pair.dmin).flatten(1).any(dim=1)
        return pair_energy(torch, d, pair), clash


def free_points(scorer, center, radius, spacing=0.5, clearance=2.6):
    g = torch.arange(-radius, radius + 1e-6, spacing, device=scorer.device)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    pts = center + grid[grid.norm(dim=1) <= radius]
    d = torch.cat([torch.cdist(pts[i:i + 4096], scorer.env_heavy).min(dim=1).values
                   for i in range(0, len(pts), 4096)])
    return pts[d >= clearance]
