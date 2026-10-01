import argparse
import csv
import json
import os
import re

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
        for part, charge, mult in (("complex", info["charge_complex"], mult_complex),
                                   ("pocket", info["charge_pocket"], mult_pocket),
                                   ("ligand", info["charge_ligand"], mult_ligand)):
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
        row = {"pose": info["pose"], "xtb_e_final_kcal": info["xtb_e_final_kcal"], **{f"E_{k}_Ha": v for k, v in e.items()}}
        row["dft_e_int_kcal"] = ((e["complex"] - e["pocket"] - e["ligand"]) * HARTREE_TO_KCAL
                                 if all(v is not None for v in e.values()) else None)
        rows.append(row)
    rows.sort(key=lambda r: (r["dft_e_int_kcal"] is None, r["dft_e_int_kcal"] or 0.0))
    return rows


def main():
    p = argparse.ArgumentParser(description="ORCA inputs and result collection for DFT rescoring of exported poses")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("inputs")
    m.add_argument("dir")
    m.add_argument("--keywords", default="r2SCAN-3c TightSCF")
    m.add_argument("--nprocs", type=int, default=8)
    m.add_argument("--mult-complex", type=int, default=1)
    m.add_argument("--mult-pocket", type=int, default=1)
    m.add_argument("--mult-ligand", type=int, default=1)
    c = sub.add_parser("collect")
    c.add_argument("dir")
    a = p.parse_args()
    if a.cmd == "inputs":
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
