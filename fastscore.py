import numpy as np
import torch

from metals import DONOR_ELEMENTS, METALS, VDW_RADII


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


def adam_step(p, g, m, v, k, lr, b1=0.9, b2=0.999, eps=1e-8):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    p.sub_(lr * (m / (1 - b1 ** k)) / ((v / (1 - b2 ** k)).sqrt() + eps))


class PoseSet:
    def __init__(self, coords, conf, score, site, diag=None):
        self.diag = diag or {}
        self.coords = coords
        self.conf = conf
        self.score = score
        self.site = site

    def __len__(self):
        return len(self.conf)

    def take(self, idx):
        idx = np.asarray(idx, dtype=int)
        return PoseSet(self.coords[idx], self.conf[idx], self.score[idx], self.site[idx])

    @staticmethod
    def concat(sets):
        return PoseSet(np.concatenate([s.coords for s in sets]),
                       np.concatenate([s.conf for s in sets]),
                       np.concatenate([s.score for s in sets]),
                       np.concatenate([s.site for s in sets]))


def random_rotations(n, gen, device):
    q = torch.randn(n, 4, generator=gen, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(n, 3, 3)


def rodrigues(w):
    theta = (w.pow(2).sum(1, keepdim=True) + 1e-12).sqrt()
    k = w / theta
    K = torch.zeros(len(w), 3, 3, device=w.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(theta)[:, :, None], torch.cos(theta)[:, :, None]
    return torch.eye(3, device=w.device) + s * K + (1 - c) * (K @ K)


def place(C, idx, R, t):
    return torch.einsum("bij,bnj->bni", R, C[idx]) + t[:, None, :]


def diverse(coords, order, cutoff, limit):
    kept = []
    for i in order:
        if kept:
            d = np.sqrt(((coords[kept] - coords[i]) ** 2).sum(-1).mean(-1))
            if d.min() < cutoff:
                continue
        kept.append(int(i))
        if len(kept) >= limit:
            break
    return kept


def free_points(scorer, center, radius, spacing=0.5, clearance=2.6):
    g = torch.arange(-radius, radius + 1e-6, spacing, device=scorer.device)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    pts = center + grid[grid.norm(dim=1) <= radius]
    d = torch.cat([torch.cdist(pts[i:i + 4096], scorer.env_heavy).min(dim=1).values
                   for i in range(0, len(pts), 4096)])
    return pts[d >= clearance]


def search(scorer, conf_coords, heavy, center, cfg, seed, site):
    device = scorer.device
    gen = torch.Generator(device=device).manual_seed(seed)
    centered = conf_coords - conf_coords.mean(axis=1, keepdims=True)
    C = torch.as_tensor(centered, dtype=torch.float32, device=device)
    ctr = torch.as_tensor(center, dtype=torch.float32, device=device)
    n = cfg.n_random
    free = 0
    voids = free_points(scorer, ctr, cfg.search_radius)
    scores = np.empty(n, dtype=np.float32)
    conf = np.empty(n, dtype=np.int64)
    Rs = np.empty((n, 3, 3), dtype=np.float32)
    ts = np.empty((n, 3), dtype=np.float32)
    step = scorer.chunk(False, False)
    with torch.no_grad():
        for s in range(0, n, step):
            b = min(step, n - s)
            idx = torch.randint(len(C), (b,), generator=gen, device=device)
            R = random_rotations(b, gen, device)
            v = torch.randn(b, 3, generator=gen, device=device)
            v = v / v.norm(dim=1, keepdim=True)
            r = cfg.search_radius * torch.rand(b, 1, generator=gen, device=device) ** (1 / 3)
            t = ctr + v * r
            if len(voids):
                pick = voids[torch.randint(len(voids), (b,), generator=gen, device=device)]
                jitter = 0.25 * torch.randn(b, 3, generator=gen, device=device)
                t = torch.where(torch.rand(b, 1, generator=gen, device=device) < 0.75, pick + jitter, t)
            sc, cl = scorer.evaluate(place(C, idx, R, t), full=False)
            free += int((~cl).sum())
            scores[s:s + b] = sc.cpu().numpy()
            conf[s:s + b] = idx.cpu().numpy()
            Rs[s:s + b] = R.cpu().numpy()
            ts[s:s + b] = t.cpu().numpy()
    top = np.argsort(scores)[:cfg.n_optimize]
    conf_t = torch.as_tensor(conf[top], device=device)
    R0 = torch.as_tensor(Rs[top], device=device)
    t0 = torch.as_tensor(ts[top], device=device)
    final_R, final_t = [], []
    step = scorer.chunk(True, False)
    for s in range(0, len(top), step):
        sl = slice(s, s + step)
        idx, R0b, t0b = conf_t[sl], R0[sl], t0[sl]
        w = 1e-3 * torch.randn(len(idx), 3, generator=gen, device=device)
        dt = torch.zeros(len(idx), 3, device=device)
        mw, vw, mt, vt = (torch.zeros_like(w), torch.zeros_like(w), torch.zeros_like(dt), torch.zeros_like(dt))
        decay = 0.5 ** (1 / max(1, cfg.opt_steps // 4))
        for k in range(1, cfg.opt_steps + 1):
            w.requires_grad_(True)
            dt.requires_grad_(True)
            X = place(C, idx, rodrigues(w) @ R0b, t0b + dt)
            sc, _ = scorer.evaluate(X, full=False)
            excess = torch.relu((X.mean(1) - ctr).norm(dim=1) - cfg.search_radius)
            gw, gt = torch.autograd.grad((sc + 10.0 * excess ** 2).sum(), [w, dt])
            w, dt = w.detach(), dt.detach()
            adam_step(w, gw, mw, vw, k, 0.05 * decay ** k)
            adam_step(dt, gt, mt, vt, k, 0.2 * decay ** k)
        with torch.no_grad():
            final_R.append(rodrigues(w) @ R0b)
            final_t.append(t0b + dt)
    with torch.no_grad():
        Rf, tf = torch.cat(final_R), torch.cat(final_t)
        Xf, sf, cf = [], [], []
        partners = torch.zeros(scorer.n_env, device=device)
        step = scorer.chunk(False, True)
        for s in range(0, len(top), step):
            X = place(C, conf_t[s:s + step], Rf[s:s + step], tf[s:s + step])
            sc, _ = scorer.evaluate(X, full=False)
            _, clash = scorer.evaluate(X, full=True)
            partners += scorer.clash_partners(X[clash]).float() if bool(clash.any()) else 0
            Xf.append(X.cpu().numpy())
            sf.append(sc.cpu().numpy())
            cf.append(clash.cpu().numpy())
    X, sc, clash = np.concatenate(Xf), np.concatenate(sf), np.concatenate(cf)
    ok = np.where(~clash)[0]
    order = ok[np.argsort(sc[ok])]
    if cfg.selection == "cluster":
        keep = [int(i) for i in order[:cfg.cluster_pool]]
    else:
        keep = diverse(X[:, heavy, :], order, cfg.diversity_rmsd, cfg.n_qm * 3)
    diag = {"random_clash_free": free / n, "optimized": len(top), "optimized_clash_free": int(len(ok)),
            "partners": partners.cpu().numpy()}
    return PoseSet(X[keep].astype(float), conf[top][keep], sc[keep].astype(float),
                   np.full(len(keep), site, dtype=int), diag)
