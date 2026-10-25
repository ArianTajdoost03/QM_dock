HBOND_NEAR = 2.8
HBOND_FAR = 3.5
HYDROPHOBIC_NEAR = 0.9
HYDROPHOBIC_FAR = 1.9
ELEC_CUTOFF = 8.0
ELEC_MIN = 2.0
METAL_DISTANCE = 2.1
METAL_WIDTH = 0.35


def relu(x):
    return (x + abs(x)) * 0.5


def ramp(x):
    return relu(x) - relu(x - 1.0)


def pair_energy(xp, d, pair):
    n = d.shape[0]
    s = d - pair.rsum
    steric = (-0.0356 * xp.exp(-(s / 0.5) ** 2) - 0.00516 * xp.exp(-((s - 3.0) / 2.0) ** 2)
              + 0.84 * relu(-s) ** 2)
    contact = 5.0 * relu(1.8 - d) ** 2
    energy = xp.where(pair.coord, contact, steric).reshape(n, -1).sum(1)
    if pair.metal_cols is not None:
        reach = pair.metal_weight[None, :, None] * xp.exp(-((d[:, :, pair.metal_cols] - METAL_DISTANCE) / METAL_WIDTH) ** 2)
        total = reach.sum(1)
        if pair.metal_free is None:
            energy = energy - total.sum(1)
        else:
            free = pair.metal_free[None, :]
            energy = energy - (free * xp.tanh(total / (free + 1e-6))).sum(1)
    if pair.hbond is not None:
        energy = energy - pair.w_hbond * (pair.hbond * ramp((HBOND_FAR - d) / (HBOND_FAR - HBOND_NEAR))).reshape(n, -1).sum(1)
    if pair.hydrophobic is not None:
        energy = energy - pair.w_hydrophobic * (
            pair.hydrophobic * ramp((HYDROPHOBIC_FAR - s) / (HYDROPHOBIC_FAR - HYDROPHOBIC_NEAR))).reshape(n, -1).sum(1)
    if pair.charges is not None:
        inverse = 1.0 / (relu(d - ELEC_MIN) + ELEC_MIN) - 1.0 / ELEC_CUTOFF
        energy = energy + pair.w_elec * (pair.charges * inverse * ramp(ELEC_CUTOFF - d)).reshape(n, -1).sum(1)
    return energy
