import argparse
import csv
import os

from bench_common import find_complexes, read_ids
from qmdock.chem.elements import METALS
from qmdock.chem.receptor import hydrogen_warning, load_receptor
from run_benchmark import fresh_ligand

FIELDS = ["id", "status", "problems", "ligand_heavy", "receptor_atoms", "receptor_h_per_heavy", "metals"]


def check_complex(cid, protein, ligand, require_hydrogens, seed):
    row = {k: "" for k in FIELDS}
    row["id"] = cid
    problems, failed = [], False
    try:
        receptor = load_receptor(protein)
        row.update(receptor_atoms=len(receptor.elements), receptor_h_per_heavy=round(receptor.hydrogen_fraction(), 3),
                   metals=" ".join(sorted({e for e in receptor.elements if e in METALS})))
        if require_hydrogens and hydrogen_warning(receptor):
            problems.append("receptor_without_hydrogens")
    except Exception as exc:
        problems.append("receptor_failed:%s" % exc)
        failed = True
    try:
        fresh, _, info = fresh_ligand(ligand, cid, seed)
        row["ligand_heavy"] = info["n_heavy"]
    except Exception as exc:
        problems.append("ligand_failed:%s" % exc)
        failed = True
    row["problems"] = ";".join(problems)
    row["status"] = "fail" if failed else ("warn" if problems else "ok")
    return row


def main():
    parser = argparse.ArgumentParser(description="Check receptors and ligands with qmdock's own loaders before docking")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--out", default=".")
    parser.add_argument("--require-hydrogens", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    found = find_complexes(args.data)
    ids = read_ids(args.ids) if args.ids else list(found)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for n, cid in enumerate(ids, 1):
        if cid not in found:
            row = {k: "" for k in FIELDS}
            row.update(id=cid, status="fail", problems="files_not_found")
        else:
            row = check_complex(cid, *found[cid], args.require_hydrogens, args.seed)
        rows.append(row)
        if row["status"] != "ok":
            print("[%d/%d] %s %s %s" % (n, len(ids), cid, row["status"], row["problems"]), flush=True)
    path = os.path.join(args.out, "preflight.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tally = {s: sum(1 for r in rows if r["status"] == s) for s in ("ok", "warn", "fail")}
    print("checked %d complexes: %d ok, %d warn, %d fail -> %s" % (len(rows), tally["ok"], tally["warn"], tally["fail"], path))
    bare = sum(1 for r in rows if r["receptor_h_per_heavy"] != "" and r["receptor_h_per_heavy"] < 0.3)
    if bare:
        try:
            import openmm
            import pdbfixer
        except ImportError:
            print("%d receptors have no hydrogens and pdbfixer/openmm are not installed, so qmdock cannot add them" % bare)


if __name__ == "__main__":
    main()
