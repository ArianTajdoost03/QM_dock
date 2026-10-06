import numpy as np


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
