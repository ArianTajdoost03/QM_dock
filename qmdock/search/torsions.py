import math

import torch
from rdkit import Chem

from qmdock.chem.elements import VDW_RADII


class TorsionModel:
    def __init__(self, rotors, mol, device, soft=0.75, hard=0.7):
        self.K = len(rotors)
        self.device = device
        self.a = [r[0] for r in rotors]
        self.b = [r[1] for r in rotors]
        n = mol.GetNumAtoms()
        masks = torch.zeros(self.K, n, dtype=torch.bool)
        for k, r in enumerate(rotors):
            masks[k, list(r[2])] = True
        self.masks = masks.to(device)[:, None, :, None]
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        heavy = [i for i, e in enumerate(elements) if e != "H"]
        topo = Chem.GetDistanceMatrix(mol)
        pairs = [(i, j) for x, i in enumerate(heavy) for j in heavy[x + 1:] if topo[i][j] >= 4]
        self.pairs = len(pairs)
        if pairs:
            self.pi = torch.tensor([p[0] for p in pairs], device=device)
            self.pj = torch.tensor([p[1] for p in pairs], device=device)
            radii = torch.tensor([VDW_RADII.get(elements[i], 1.7) + VDW_RADII.get(elements[j], 1.7)
                                  for i, j in pairs], device=device)
            self.soft, self.hard = soft * radii, hard * radii

    def apply(self, X, delta):
        for k in range(self.K):
            pivot = X[:, self.a[k], :]
            axis = X[:, self.b[k], :] - pivot
            n = (axis / (axis.norm(dim=1, keepdim=True) + 1e-9))[:, None, :]
            rel = X - pivot[:, None, :]
            ang = delta[:, k][:, None, None]
            c, s = torch.cos(ang), torch.sin(ang)
            rot = rel * c + torch.cross(n.expand_as(rel), rel, dim=-1) * s + n * (rel * n).sum(-1, keepdim=True) * (1 - c)
            X = torch.where(self.masks[k], pivot[:, None, :] + rot, X)
        return X

    def intra(self, X):
        if not self.pairs:
            zero = torch.zeros(len(X), device=X.device)
            return zero, zero.bool()
        d = (X[:, self.pi, :] - X[:, self.pj, :]).norm(dim=-1)
        return 5.0 * torch.relu(self.soft - d).pow(2).sum(1), (d < self.hard).any(1)

    def random_deltas(self, b, gen, fraction):
        d = (torch.rand(b, self.K, generator=gen, device=self.device) * 2 - 1) * math.pi
        keep = torch.rand(b, 1, generator=gen, device=self.device) < fraction
        return torch.where(keep, d, torch.zeros_like(d))
