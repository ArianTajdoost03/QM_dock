import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from scipy.stats import spearmanr


def recovery(results_csv, poses_sdf, reference_sdf, threshold):
    ref = Chem.RemoveHs(next(iter(Chem.SDMolSupplier(reference_sdf, removeHs=False))))
    rows = []
    for m in Chem.SDMolSupplier(poses_sdf, removeHs=False):
        rmsd = rdMolAlign.CalcRMS(Chem.RemoveHs(m), ref)
        rows.append({"pose": int(m.GetProp("_Name").split("_")[1]), "rmsd_ref": rmsd})
    df = pd.read_csv(results_csv).merge(pd.DataFrame(rows), on="pose")
    df = df[df.status == "ok"].sort_values("final_rank")
    hits = df[df.rmsd_ref <= threshold]
    out = {
        "n_poses": len(df),
        "top1_rmsd": float(df.rmsd_ref.iloc[0]),
        "best_rmsd_any": float(df.rmsd_ref.min()),
        "first_hit_rank": int(hits.final_rank.iloc[0]) if len(hits) else None,
        "top1_hit": bool(len(hits) and hits.final_rank.iloc[0] == df.final_rank.iloc[0]),
    }
    if len(df) > 3:
        rho, p = spearmanr(df.e_final, df.rmsd_ref)
        out["spearman_energy_vs_rmsd"] = float(rho)
        out["spearman_p"] = float(p)
    if len(hits) and len(df) > len(hits):
        out["energy_gap_hit_vs_decoys_kcal"] = float(df[df.rmsd_ref > threshold].e_final.min() - hits.e_final.min())
    return out


def scan(complex_xyz, n_ligand, offsets, uhf=0, charge=0):
    from metals import ATOMIC_NUMBER
    from qm import solve

    lines = open(complex_xyz).read().splitlines()[2:]
    el = [l.split()[0] for l in lines]
    xyz = np.array([[float(v) for v in l.split()[1:4]] for l in lines])
    numbers = np.array([ATOMIC_NUMBER[e] for e in el], dtype=np.int32)
    n_rec = len(el) - n_ligand
    lig = xyz[n_rec:]
    axis = lig.mean(axis=0) - xyz[:n_rec].mean(axis=0)
    axis /= np.linalg.norm(axis)
    e_lig = solve(numbers[n_rec:], lig, 0, uhf)[0]
    e_rec = solve(numbers[:n_rec], xyz[:n_rec], charge, uhf)[0]
    rows = []
    for off in offsets:
        pos = xyz.copy()
        pos[n_rec:] = lig + off * axis
        e = solve(numbers, pos, charge, uhf)[0]
        rows.append({"offset_A": off, "e_int_kcal": (e - e_rec - e_lig) * 627.509474})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recovery")
    r.add_argument("--results", required=True)
    r.add_argument("--poses", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--threshold", type=float, default=2.0)
    s = sub.add_parser("scan")
    s.add_argument("--complex", required=True)
    s.add_argument("--n-ligand", type=int, required=True)
    s.add_argument("--offsets", type=float, nargs="+", default=[-0.4, 0.0, 0.4, 0.8, 1.5, 3.0])
    a = p.parse_args()
    if a.cmd == "recovery":
        for k, v in recovery(a.results, a.poses, a.reference, a.threshold).items():
            print(f"{k}: {v}")
    else:
        print(scan(a.complex, a.n_ligand, a.offsets).to_string(index=False))


if __name__ == "__main__":
    main()
