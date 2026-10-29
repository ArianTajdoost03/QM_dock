import argparse
import csv
import json
import os
import re

import numpy as np
import pandas as pd

HARTREE_TO_KCAL = 627.509474
ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def read_xyz(path):
    lines = open(path).read().splitlines()[2:]
    return [l.split()[:4] for l in lines if l.strip()]


def write_orca(path, xyz, charge, mult, keywords, nprocs):
    with open(path, "w") as f:
        f.write(f"! {keywords}\n%pal nprocs {nprocs} end\n* xyz {charge} {mult}\n")
        for el, x, y, z in xyz:
            f.write(f"{el} {x} {y} {z}\n")
        f.write("*\n")


def make_inputs(root, keywords, nprocs, mult_complex, mult_pocket, mult_ligand):
    made = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        manifest = os.path.join(d, "manifest.json")
        if not os.path.exists(manifest):
            continue
        info = json.load(open(manifest))
        open_shell = int(info.get("uhf", 0)) + 1
        for part, charge, mult in (("complex", info["charge_complex"], mult_complex or open_shell),
                                   ("pocket", info["charge_pocket"], mult_pocket or open_shell),
                                   ("ligand", info["charge_ligand"], mult_ligand or 1)):
            out = os.path.join(d, f"{part}.inp")
            write_orca(out, read_xyz(os.path.join(d, f"{part}.xyz")), charge, mult, keywords, nprocs)
            made.append(out)
    return made


def last_energy(path):
    if not os.path.exists(path):
        return None
    found = ENERGY.findall(open(path, errors="ignore").read())
    return float(found[-1]) if found else None


def collect(root):
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        info = json.load(open(os.path.join(d, "manifest.json")))
        e = {part: last_energy(os.path.join(d, f"{part}.out")) for part in ("complex", "pocket", "ligand")}
        row = {"pose": info["pose"], "xtb_e_final_kcal": info["xtb_e_final_kcal"],
               "xtb_e_int_kcal": info.get("xtb_e_int_kcal"), "uhf": info.get("uhf", 0),
               **{f"E_{k}_Ha": v for k, v in e.items()}}
        row["dft_e_int_kcal"] = ((e["complex"] - e["pocket"] - e["ligand"]) * HARTREE_TO_KCAL
                                 if all(v is not None for v in e.values()) else None)
        rows.append(row)
    rows.sort(key=lambda r: (r["dft_e_int_kcal"] is None, r["dft_e_int_kcal"] or 0.0))
    return rows


def pair_metrics(xtb, dft):
    xtb, dft = np.asarray(xtb, dtype=float), np.asarray(dft, dtype=float)
    out = {"n": int(len(xtb))}
    if len(xtb) == 0:
        return out
    out["mean_offset"] = float((xtb - dft).mean())
    out["mae"] = float(np.abs(xtb - dft).mean())
    out["rmse"] = float(np.sqrt(((xtb - dft) ** 2).mean()))
    if len(xtb) >= 3 and xtb.std() > 0 and dft.std() > 0:
        out["pearson"] = float(np.corrcoef(xtb, dft)[0, 1])
        out["spearman"] = float(pd.Series(xtb).rank().corr(pd.Series(dft).rank()))
        out["slope"] = float(np.polyfit(dft, xtb, 1)[0])
    return out


def compare_files(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["run"] = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["xtb_e_int_kcal", "dft_e_int_kcal"])
    result = {"overall": pair_metrics(df["xtb_e_int_kcal"], df["dft_e_int_kcal"]), "runs": {}}
    agree = []
    for run, g in df.groupby("run"):
        result["runs"][run] = pair_metrics(g["xtb_e_int_kcal"], g["dft_e_int_kcal"])
        if len(g) >= 2:
            agree.append(g["xtb_e_int_kcal"].idxmin() == g["dft_e_int_kcal"].idxmin())
    result["top1_agreement"] = float(np.mean(agree)) if agree else float("nan")
    result["runs_compared"] = len(agree)
    return result


def format_comparison(result):
    o = result["overall"]
    lines = [f"{o['n']} poses with both energies; GFN2 minus DFT interaction energy:"]
    if o["n"]:
        lines.append(f"  mean offset {o['mean_offset']:.2f}, MAE {o['mae']:.2f}, RMSE {o['rmse']:.2f} kcal/mol")
    if "pearson" in o:
        lines.append(f"  Pearson {o['pearson']:.2f}, Spearman {o['spearman']:.2f}, slope {o['slope']:.2f}")
    lines.append(f"  both methods pick the same best pose in {100 * result['top1_agreement']:.0f}% of "
                 f"{result['runs_compared']} run(s)" if result["runs_compared"] else "  fewer than 2 poses per run, no ranking check")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="ORCA inputs and result collection for DFT rescoring of exported poses")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("inputs")
    m.add_argument("dir")
    m.add_argument("--keywords", default="r2SCAN-3c TightSCF")
    m.add_argument("--nprocs", type=int, default=8)
    m.add_argument("--mult-complex", type=int, default=None)
    m.add_argument("--mult-pocket", type=int, default=None)
    m.add_argument("--mult-ligand", type=int, default=None)
    c = sub.add_parser("collect")
    c.add_argument("dir")
    k = sub.add_parser("compare")
    k.add_argument("results", nargs="+", help="dft_results.csv files, one per run")
    a = p.parse_args()
    if a.cmd == "compare":
        print(format_comparison(compare_files(a.results)))
    elif a.cmd == "inputs":
        for path in make_inputs(a.dir, a.keywords, a.nprocs, a.mult_complex, a.mult_pocket, a.mult_ligand):
            print(path)
    else:
        rows = collect(a.dir)
        out = os.path.join(a.dir, "dft_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(r)
        print(f"written {out}")


if __name__ == "__main__":
    main()
