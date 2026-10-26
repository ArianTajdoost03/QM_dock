import math

import torch
import torch.nn.functional as F

from qmdock.search import terms
from qmdock.search.terms import ramp, relu

NEAR_REACH = 2.5
BLOCK_ELEMENTS = 12_000_000
PAIR_ELEMENTS = 6_000_000


def near_values(d, rsum, coord, hbond, hydrophobic, w_hbond, w_hydrophobic):
    s = d[None, :] - rsum
    steric = -0.0356 * torch.exp(-(s / 0.5) ** 2) + 0.84 * relu(-s) ** 2
    energy = torch.where(coord, torch.zeros_like(steric), steric)
    if hbond is not None:
        energy = energy - w_hbond * hbond * ramp((terms.HBOND_FAR - d[None, :]) / (terms.HBOND_FAR - terms.HBOND_NEAR))
    if hydrophobic is not None:
        energy = energy - w_hydrophobic * hydrophobic * ramp(
            (terms.HYDROPHOBIC_FAR - s) / (terms.HYDROPHOBIC_FAR - terms.HYDROPHOBIC_NEAR))
    return torch.where(d[None, :] < rsum + NEAR_REACH, energy, torch.zeros_like(energy))


def far_values(d, rsum, coord):
    s = d[:, None, :] - rsum[None]
    tail = -0.00516 * torch.exp(-((s - 3.0) / 2.0) ** 2)
    return torch.where(coord[None], torch.zeros_like(tail), tail).sum(2)


def potential_values(d, charge):
    inverse = 1.0 / (relu(d - terms.ELEC_MIN) + terms.ELEC_MIN) - 1.0 / terms.ELEC_CUTOFF
    return (charge[None, :] * inverse * ramp(terms.ELEC_CUTOFF - d)).sum(1)


class Volume:
    def __init__(self, center, half_width, spacing, max_nodes, channels, device):
        if (2 * half_width / spacing + 1) ** 3 > max_nodes:
            spacing = 2 * half_width / (max_nodes ** (1 / 3) - 1)
        self.count = int(math.ceil(2 * half_width / spacing)) + 1
        self.spacing = 2 * half_width / (self.count - 1)
        self.center = center
        self.half_width = half_width
        self.origin = center - half_width
        self.scale = 2.0 / ((self.count - 1) * self.spacing)
        self.device = device
        self.data = torch.zeros(channels, self.count, self.count, self.count, device=device)

    def node_blocks(self, size):
        total = self.count ** 3
        flat = torch.arange(total, device=self.device)
        for start in range(0, total, size):
            idx = flat[start:start + size]
            ix, iy, iz = idx // (self.count * self.count), (idx // self.count) % self.count, idx % self.count
            pts = self.origin + torch.stack([ix, iy, iz], dim=1).float() * self.spacing
            inside = ((pts - self.center) ** 2).sum(1) <= self.half_width ** 2
            sel = torch.where(inside)[0]
            if len(sel):
                yield pts[sel], (ix[sel], iy[sel], iz[sel])

    def cube_blocks(self, side):
        m = max(1, int(side / self.spacing))
        axis = torch.arange(self.count, device=self.device)
        for x in range(0, self.count, m):
            for y in range(0, self.count, m):
                for z in range(0, self.count, m):
                    gx, gy, gz = torch.meshgrid(axis[x:x + m], axis[y:y + m], axis[z:z + m], indexing="ij")
                    ix, iy, iz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
                    pts = self.origin + torch.stack([ix, iy, iz], dim=1).float() * self.spacing
                    sel = torch.where(((pts - self.center) ** 2).sum(1) <= self.half_width ** 2)[0]
                    if len(sel):
                        yield pts[sel], (ix[sel], iy[sel], iz[sel])

    def finish(self):
        self.data = self.data.permute(0, 3, 2, 1).contiguous()[None]

    def sample(self, pos):
        b, n, _ = pos.shape
        g = (pos - self.origin) * self.scale - 1.0
        out = F.grid_sample(self.data, g.reshape(1, b * n, 1, 1, 3), mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        return out.reshape(self.data.shape[1], b, n)


def unique_rows(matrix):
    _, inverse = torch.unique(matrix, dim=0, return_inverse=True)
    first = torch.zeros(int(inverse.max()) + 1, dtype=torch.long, device=matrix.device)
    first.scatter_(0, inverse, torch.arange(len(inverse), device=matrix.device))
    return inverse, first


class EnergyGrid:
    def __init__(self, scorer, center, half_width, spacing, far_spacing, max_nodes):
        device = scorer.device
        pair = scorer.pair_heavy
        center = torch.as_tensor(center, dtype=torch.float32, device=device)
        env = scorer.env_heavy
        features = [pair.rsum, pair.coord.float()]
        if pair.hbond is not None:
            features.append(pair.hbond)
        if pair.hydrophobic is not None:
            features.append(pair.hydrophobic)
        self.type_of, first = unique_rows(torch.cat(features, dim=1))
        self.n_types = len(first)
        rsum, coord = pair.rsum[first], pair.coord[first]
        hbond = pair.hbond[first] if pair.hbond is not None else None
        hydrophobic = pair.hydrophobic[first] if pair.hydrophobic is not None else None
        self.far_of, far_first = unique_rows(torch.cat([pair.rsum, pair.coord.float()], dim=1))
        self.n_far = len(far_first)
        far_rsum, far_coord = pair.rsum[far_first], pair.coord[far_first]
        use_charge = scorer.charge_e is not None and bool(pair.w_elec)
        self.charge_l = scorer.charge_l if use_charge else None
        self.w_elec = pair.w_elec
        self.near = Volume(center, half_width, spacing, max_nodes, self.n_types, device)
        self.far = Volume(center, half_width, far_spacing, max_nodes, self.n_far + (1 if use_charge else 0), device)
        cut = float(rsum.max()) + NEAR_REACH + 1e-3
        piece = max(1, PAIR_ELEMENTS // self.n_types)
        env_index = torch.arange(len(env), device=device)
        for pts, (ix, iy, iz) in self.near.cube_blocks(cut):
            low, high = pts.min(0).values - cut, pts.max(0).values + cut
            near_env = env[((env >= low) & (env <= high)).all(1)]
            if len(near_env) == 0:
                continue
            rows, cols = torch.nonzero(torch.cdist(pts, near_env, compute_mode="donot_use_mm_for_euclid_dist") < cut, as_tuple=True)
            acc = torch.zeros(len(pts), self.n_types, device=device)
            for lo in range(0, len(rows), piece):
                r, c = rows[lo:lo + piece], cols[lo:lo + piece]
                dd = (pts[r] - near_env[c]).norm(dim=1)
                gc = env_index[((env >= low) & (env <= high)).all(1)][c]
                val = near_values(dd, rsum[:, gc], coord[:, gc], hbond[:, gc] if hbond is not None else None,
                                  hydrophobic[:, gc] if hydrophobic is not None else None,
                                  pair.w_hbond, pair.w_hydrophobic)
                acc.index_add_(0, r, val.T)
            self.near.data[:, ix, iy, iz] = acc.T
        for pts, (ix, iy, iz) in self.far.node_blocks(max(1, BLOCK_ELEMENTS // (len(env) * self.n_far))):
            d = torch.cdist(pts, env, compute_mode="donot_use_mm_for_euclid_dist")
            self.far.data[:self.n_far, ix, iy, iz] = far_values(d, far_rsum, far_coord).T
            if use_charge:
                self.far.data[self.n_far, ix, iy, iz] = potential_values(d, scorer.charge_e)
        self.near.finish()
        self.far.finish()
        self.n_channels = self.n_types + self.n_far
        self.pair = pair
        self.metal_xyz = env[pair.metal_cols] if pair.metal_cols is not None else None

    def __call__(self, pos):
        b, n, _ = pos.shape
        near = self.near.sample(pos)
        far = self.far.sample(pos)
        energy = torch.gather(near, 0, self.type_of[None, None, :].expand(1, b, n))[0].sum(1)
        energy = energy + torch.gather(far[:self.n_far], 0, self.far_of[None, None, :].expand(1, b, n))[0].sum(1)
        if self.charge_l is not None:
            energy = energy + self.w_elec * (self.charge_l[None, :] * far[self.n_far]).sum(1)
        pair = self.pair
        if self.metal_xyz is not None:
            d = torch.cdist(pos, self.metal_xyz, compute_mode="donot_use_mm_for_euclid_dist")
            energy = energy + torch.where(pair.coord[:, pair.metal_cols][None], 5.0 * relu(1.8 - d) ** 2,
                                          torch.zeros_like(d)).reshape(b, -1).sum(1)
            reach = pair.metal_weight[None, :, None] * torch.exp(-((d - terms.METAL_DISTANCE) / terms.METAL_WIDTH) ** 2)
            total = reach.sum(1)
            if pair.metal_free is None:
                energy = energy - total.sum(1)
            else:
                free = pair.metal_free[None, :]
                energy = energy - (free * torch.tanh(total / (free + 1e-6))).sum(1)
        return energy
