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


def main():
    p = argparse.ArgumentParser(description="Compare a finished docking run with a known ligand pose")
    p.add_argument("--results", required=True, help="results.csv of the run")
    p.add_argument("--poses", required=True, help="poses.sdf of the run")
    p.add_argument("--reference", required=True, help="crystal ligand SDF in the receptor frame")
    p.add_argument("--threshold", type=float, default=2.0, help="RMSD in A that counts as success")
    a = p.parse_args()
    for k, v in recovery(a.results, a.poses, a.reference, a.threshold).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
