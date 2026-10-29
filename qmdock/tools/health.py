import argparse
import glob
import os

import numpy as np
import pandas as pd

from qmdock.config import RESULTS_FILE

STRAIN_WARNING = 30.0


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def summarize(df):
    ok = df[df["status"] == "ok"] if "status" in df else df
    out = {"poses": int(len(df)), "scored": int(len(ok))}
    if "status" in df:
        out["status"] = {k: int(v) for k, v in df["status"].value_counts().items()}
    if len(ok) == 0:
        return out
    for col in ("e_int", "strain", "strain_raw", "e_bind"):
        if col in ok:
            out[f"{col}_mean"] = float(ok[col].mean())
            out[f"{col}_sd"] = float(ok[col].std(ddof=0))
    if {"e_bind", "strain"} <= set(ok.columns):
        out["corr_ebind_strain"] = float(ok["e_bind"].corr(ok["strain"])) if len(ok) > 2 else float("nan")
        out["strained_fraction"] = float((ok["strain"] > STRAIN_WARNING).mean())
    if {"e_bind", "e_int"} <= set(ok.columns):
        out["corr_ebind_eint"] = float(ok["e_bind"].corr(ok["e_int"])) if len(ok) > 2 else float("nan")
    out["strain_dominates"] = bool("strain_sd" in out and "e_int_sd" in out and out["strain_sd"] > out["e_int_sd"])
    if "scf_attempt" in ok:
        out["scf_retry_fraction"] = float((ok["scf_attempt"] > 0).mean())
    if "n_atoms" in ok:
        out["n_atoms_mean"] = float(ok["n_atoms"].mean())
        out["n_atoms_max"] = int(ok["n_atoms"].max())
    if "e_bind_relaxed" in ok:
        rel = ok[ok["e_bind_relaxed"].notna()]
        out["relaxed"] = int(len(rel))
        if len(rel):
            out["relax_shift_mean"] = float((rel["e_bind_relaxed"] - rel["e_bind"]).mean())
            out["relax_shift_sd"] = float((rel["e_bind_relaxed"] - rel["e_bind"]).std(ddof=0))
            out["stage1_vs_relaxed_spearman"] = spearman(rel["e_bind"], rel["e_bind_relaxed"])
            if "relax_converged" in rel:
                out["relax_converged"] = int(rel["relax_converged"].astype(bool).sum())
    return out


def advice(s):
    notes = []
    if s.get("strain_dominates"):
        notes.append("strain varies more than the interaction energy, so stage-1 ranking mostly reflects strain")
    if s.get("strained_fraction", 0) > 0.25:
        notes.append(f"{100 * s['strained_fraction']:.0f}% of poses have strain above {STRAIN_WARNING:.0f} kcal/mol")
    if s.get("corr_ebind_strain", 0) > 0.9:
        notes.append("e_bind is almost perfectly correlated with strain")
    if s.get("scf_retry_fraction", 0) > 0.2:
        notes.append("many SCF jobs needed retries, which inflates run time")
    if s.get("relaxed") and s.get("relax_converged", s["relaxed"]) < s["relaxed"]:
        notes.append("some relaxations did not converge; consider more --relax-steps")
    rho = s.get("stage1_vs_relaxed_spearman")
    if rho is not None and not np.isnan(rho) and rho < 0.5:
        notes.append(f"stage-1 order barely predicts the relaxed order (Spearman {rho:.2f}); relax more poses")
    return notes


def format_report(name, s):
    lines = [f"{name}: {s['scored']} scored of {s['poses']} poses  {s.get('status', {})}"]
    for col in ("e_int", "strain", "strain_raw", "e_bind"):
        if f"{col}_mean" in s:
            lines.append(f"  {col:11s} mean {s[col + '_mean']:8.2f}  sd {s[col + '_sd']:7.2f}")
    if "corr_ebind_strain" in s:
        lines.append(f"  corr(e_bind, strain) {s['corr_ebind_strain']:.2f}   corr(e_bind, e_int) {s.get('corr_ebind_eint', float('nan')):.2f}")
    if s.get("relaxed"):
        lines.append(f"  relaxed {s['relaxed']} poses, shift mean {s['relax_shift_mean']:.2f} sd {s['relax_shift_sd']:.2f} "
                     f"kcal/mol, stage-1 vs relaxed Spearman {s['stage1_vs_relaxed_spearman']:.2f}")
    if "n_atoms_mean" in s:
        lines.append(f"  complex size mean {s['n_atoms_mean']:.0f} atoms, max {s['n_atoms_max']}")
    for note in advice(s):
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


def find_results(path):
    if os.path.isfile(path):
        return [path]
    found = sorted(glob.glob(os.path.join(path, "state_*", RESULTS_FILE)))
    return found or [p for p in [os.path.join(path, RESULTS_FILE)] if os.path.exists(p)]


def main():
    p = argparse.ArgumentParser(description="Diagnose energy quality of finished qmdock runs")
    p.add_argument("paths", nargs="+", help="run directories or results.csv files")
    a = p.parse_args()
    for path in a.paths:
        for csv in find_results(path):
            print(format_report(csv, summarize(pd.read_csv(csv))))


if __name__ == "__main__":
    main()
