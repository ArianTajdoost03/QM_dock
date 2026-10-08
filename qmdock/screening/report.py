import os

import pandas as pd
from rdkit import Chem

COLUMNS = ["rank", "name", "index", "status", "e_final_kcal", "e_per_heavy_atom", "level", "best_state",
           "best_state_smiles", "charge", "gap_to_next_family_kcal", "n_states", "n_states_ok", "heavy_atoms",
           "input_smiles", "seconds", "notes", "error", "dir"]


def choose_best_state(rows):
    ok = [r for r in rows if "error" not in r and r.get("best_e_final_kcal") is not None]
    if not ok:
        return None
    top = max(r["best_level"] for r in ok)
    return min((r for r in ok if r["best_level"] == top), key=lambda r: r["best_e_final_kcal"])


def ligand_result(rec, rows, seconds, ldir):
    result = {"name": rec.name, "index": rec.index, "heavy_atoms": rec.heavy_atoms, "input_smiles": rec.input_smiles,
              "notes": "; ".join(rec.notes), "seconds": round(seconds, 1), "dir": ldir,
              "n_states": len(rows), "n_states_ok": sum(1 for r in rows if "error" not in r)}
    best = choose_best_state(rows)
    if best is None:
        errors = [r["error"] for r in rows if "error" in r]
        result.update(status="failed", error=(errors[0] if errors else "no protonation state could be scored"))
        return result
    result.update(status="ok", e_final_kcal=best["best_e_final_kcal"], level=best["best_level"],
                  best_state=best["state"], best_state_smiles=best["smiles"], charge=best["charge"],
                  gap_to_next_family_kcal=best.get("gap_to_next_family_kcal"), pose_file=os.path.join(best["dir"], "poses.sdf"),
                  e_per_heavy_atom=(best["best_e_final_kcal"] / rec.heavy_atoms if rec.heavy_atoms else None))
    return result


def ranked_table(results, rank_by="energy"):
    df = pd.DataFrame(results)
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    key = "e_per_heavy_atom" if rank_by == "per-atom" else "e_final_kcal"
    ok = df[df.status == "ok"].sort_values(key, ascending=True)
    rest = df[df.status != "ok"].sort_values("index")
    out = pd.concat([ok, rest])
    out["rank"] = [i + 1 if s == "ok" else None for i, s in enumerate(out.status)]
    return out[COLUMNS]


def write_table(path, results, rank_by="energy"):
    ranked_table(results, rank_by).to_csv(path, index=False)


def write_best_poses(path, results, rank_by="energy"):
    table = ranked_table(results, rank_by)
    writer = Chem.SDWriter(path)
    written = 0
    for _, row in table[table.status == "ok"].iterrows():
        pose_file = next((r["pose_file"] for r in results if r["name"] == row["name"] and r.get("pose_file")), None)
        if not pose_file or not os.path.isfile(pose_file):
            continue
        mol = next(iter(Chem.SDMolSupplier(pose_file, removeHs=False)), None)
        if mol is None:
            continue
        mol.SetProp("_Name", str(row["name"]))
        mol.SetProp("screen_rank", str(int(row["rank"])))
        mol.SetProp("e_final_kcal", f"{row['e_final_kcal']:.3f}")
        mol.SetProp("state_smiles", str(row["best_state_smiles"]))
        writer.write(mol)
        written += 1
    writer.close()
    return written
