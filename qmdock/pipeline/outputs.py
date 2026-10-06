import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from qmdock.chem.pdb_io import write_xyz
from qmdock.qm.pool import log
from qmdock.search.clustering import sym_rmsd


def pose_table(sel, res):
    poses = sel.poses
    rows = []
    for p in range(len(poses)):
        s1 = res.stage1.get(p, {"status": "not_evaluated"})
        status = "transfer_rejected" if p in sel.rejected else s1["status"]
        row = {"pose": p, "site": int(poses.site[p]), "conformer": int(poses.conf[p]),
               "fast_score": float(poses.score[p]), "status": status}
        if sel.groups:
            row["cluster"] = res.cluster_of.get(p)
        for k in ("n_atoms", "skipped_units", "truncated_units", "scf_attempt", "e_int", "strain", "e_bind"):
            if k in s1:
                row[k] = s1[k]
        if p in res.relaxed:
            row.update({k: v for k, v in res.relaxed[p].items() if k not in ("coords", "lig_e")})
        s3 = res.stage3.get(p)
        if s3 and s3["status"] == "ok":
            row.update(e_int_final=s3["e_int"], strain_final=s3["strain"], e_bind_final=s3["e_bind"],
                       n_atoms_final=s3["n_atoms"], truncated_final=s3["truncated_units"])
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("e_bind_relaxed", "e_bind_final"):
        if col not in df:
            df[col] = np.nan
    ok = df[df.status == "ok"].copy()
    ok["level"] = np.where(ok.e_bind_final.notna(), 3, np.where(ok.e_bind_relaxed.notna(), 2, 1))
    ok["e_final"] = ok.e_bind_final.fillna(ok.e_bind_relaxed).fillna(ok.e_bind)
    ok = ok.sort_values(["level", "e_final"], ascending=[False, True])
    ranked = pd.concat([ok, df[df.status != "ok"]])
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)
    return df, ranked


def write_cluster_table(out, sel, res):
    table = []
    for i, g in enumerate(sel.groups):
        vals = [res.stage1[m]["e_bind"] for m in g["members"] if res.stage1.get(m, {}).get("status") == "ok"]
        table.append({"cluster": i, "size": len(g["members"]), "n_scored": len(vals),
                      "best_fast_score": float(sel.poses.score[g["best"]]), "medoid_pose": g["medoid"],
                      "best_pose": g["best"], "best_qm_e_bind": min(vals) if vals else np.nan,
                      "expanded": i in res.expanded})
    pd.DataFrame(table).to_csv(os.path.join(out, "clusters.csv"), index=False)


def export_dft(cfg, out, lig, res, ranked, final_coords, best):
    top_ids = [int(p) for p in ranked[(ranked.status == "ok") & (ranked.level == best.level)].pose[:cfg.export_dft]]
    for p in top_ids:
        cl = (res.stage3.get(p) if res.stage3.get(p, {}).get("status") == "ok" else res.stage1[p])["cluster"]
        d = os.path.join(out, "dft_export", f"pose_{p}")
        os.makedirs(d, exist_ok=True)
        write_xyz(os.path.join(d, "complex.xyz"), cl.elements() + lig.elements,
                  np.vstack([cl.coords(), final_coords[p]]), f"pose {p}")
        write_xyz(os.path.join(d, "pocket.xyz"), cl.elements(), cl.coords(), f"pocket for pose {p}")
        write_xyz(os.path.join(d, "ligand.xyz"), lig.elements, final_coords[p], f"ligand pose {p}")
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"pose": p, "charge_complex": cl.charge + lig.charge, "charge_pocket": cl.charge,
                       "charge_ligand": lig.charge, "xtb_e_final_kcal": float(ranked[ranked.pose == p].e_final.iloc[0])},
                      f, indent=2)
    log(f"exported {len(top_ids)} pose(s) for DFT to {os.path.join(out, 'dft_export')}")


def write_results(cfg, out, lig, sel, res, timings, warnings):
    os.makedirs(out, exist_ok=True)
    if sel.groups:
        write_cluster_table(out, sel, res)
    df, ranked = pose_table(sel, res)
    P = len(sel.poses)
    final_coords = {p: res.relaxed[p]["coords"] if p in res.relaxed else res.coords[p] for p in range(P)}
    top = int(ranked.iloc[0].pose)
    ranked["rmsd_to_top"] = [sym_rmsd(final_coords[int(p)][lig.heavy], final_coords[top][lig.heavy], lig.perms)
                             for p in ranked.pose]
    ranked.to_csv(os.path.join(out, "results.csv"), index=False)

    writer = Chem.SDWriter(os.path.join(out, "poses.sdf"))
    for _, row in ranked[ranked.status == "ok"].iterrows():
        m = Chem.Mol(lig.mol)
        m.RemoveAllConformers()
        conf = Chem.Conformer(lig.n_atoms)
        for i, xyz in enumerate(final_coords[int(row.pose)]):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        m.AddConformer(conf)
        m.SetProp("_Name", f"pose_{int(row.pose)}")
        for key in ("final_rank", "e_final", "e_int", "strain", "fast_score", "rmsd_to_top"):
            m.SetProp(key, f"{float(row[key]):.4f}")
        writer.write(m)
    writer.close()

    best = ranked.iloc[0]
    best_cl = (res.stage3.get(top) if res.stage3.get(top, {}).get("status") == "ok" else res.stage1[top])["cluster"]
    write_xyz(os.path.join(out, "best_complex.xyz"), best_cl.elements() + lig.elements,
              np.vstack([best_cl.coords(), final_coords[top]]), f"best pose {top} e_final {best.e_final:.2f} kcal/mol")
    if cfg.export_dft:
        export_dft(cfg, out, lig, res, ranked, final_coords, best)

    distinct = ranked[(ranked.status == "ok") & (ranked.level == best.level) & (ranked.rmsd_to_top > lig.family_radius)]
    gap = float(distinct.e_final.min() - best.e_final) if len(distinct) else None
    timings = dict(timings)
    timings["qm"] = res.seconds
    summary = {"best_pose": top, "best_e_final_kcal": float(best.e_final), "best_level": int(best.level),
               "gap_to_next_family_kcal": gap, "n_poses_qm": len(res.stage1),
               "n_failed": int((df.status == "scf_failed").sum()), "timings_s": timings,
               "warnings": list(warnings) + list(res.warnings), "ligand_charge": lig.charge,
               "solvent": cfg.solvent, "selection": cfg.selection,
               "n_clusters": len(sel.groups) if sel.groups else None, "n_poses_evaluated": len(res.stage1),
               "n_transfer_rejected": len(sel.rejected)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"best pose {top}: {best.e_final:.2f} kcal/mol (level {int(best.level)}), gap to next family {gap}")
    return ranked, summary
