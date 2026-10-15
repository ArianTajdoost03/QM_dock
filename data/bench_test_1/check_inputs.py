import argparse
import csv
import os
import sys

from rdkit import Chem

from bench_common import find_complexes, read_crystal_ligand, read_ids
from run_benchmark import fresh_ligand

SUPPORTED = {"H", "Li", "B", "C", "N", "O", "F", "Na", "Mg", "Si", "P", "S", "Cl", "K", "Ca", "Mn", "Fe",
             "Co", "Ni", "Cu", "Zn", "Se", "Br", "Ag", "Cd", "I", "Hg"}
FIELDS = ["id", "status", "problems", "ligand_heavy", "ligand_repaired_nh", "aromatic_atoms_lost",
          "receptor_atoms", "receptor_h_per_heavy", "receptor_unsupported", "metals"]


def element_of(line):
    symbol = line[76:78].strip()
    if not symbol:
        symbol = "".join(c for c in line[12:16].strip() if c.isalpha())[:2]
    return symbol.capitalize()


def scan_receptor(path):
    counts = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                symbol = element_of(line)
                symbol = "H" if symbol == "D" else symbol
                counts[symbol] = counts.get(symbol, 0) + 1
    hydrogens = counts.get("H", 0)
    heavy = sum(v for k, v in counts.items() if k != "H")
    unsupported = sorted(k for k in counts if k not in SUPPORTED)
    metals = sorted(k for k in counts if k in ("Fe", "Zn", "Mn", "Cu", "Mg", "Ca", "Ni", "Co", "Cd", "Ag", "Hg"))
    return heavy + hydrogens, hydrogens / max(1, heavy), unsupported, metals


def aromatic_atoms(mol):
    return sum(1 for a in mol.GetAtoms()
               if a.GetIsAromatic() or any(b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds()))


def check_complex(cid, protein, ligand, min_hydrogens, seed):
    row = {k: "" for k in FIELDS}
    row["id"] = cid
    problems, failed = [], False
    try:
        atoms, ratio, unsupported, metals = scan_receptor(protein)
        row.update(receptor_atoms=atoms, receptor_h_per_heavy=round(ratio, 3),
                   receptor_unsupported=" ".join(unsupported), metals=" ".join(metals))
        if ratio < min_hydrogens:
            problems.append("receptor_without_hydrogens")
        if unsupported:
            problems.append("receptor_unsupported_element")
            failed = True
    except Exception as exc:
        problems.append("receptor_unreadable:%s" % exc)
        failed = True
    try:
        mol, repaired = read_crystal_ligand(ligand)
        raw = Chem.MolFromMolFile(ligand, removeHs=False, sanitize=False)
        lost = aromatic_atoms(raw) - aromatic_atoms(mol) if raw is not None else 0
        row.update(ligand_heavy=sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
                   ligand_repaired_nh=repaired, aromatic_atoms_lost=max(0, lost))
        if repaired:
            problems.append("ligand_repaired_aromatic_nh")
        if lost > 0:
            problems.append("ligand_aromaticity_lost_on_read")
        bad = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in SUPPORTED})
        if bad:
            problems.append("ligand_unsupported_element:" + "".join(bad))
            failed = True
        else:
            fresh_ligand(ligand, cid, seed)
    except Exception as exc:
        problems.append("ligand_failed:%s" % exc)
        failed = True
    row["problems"] = ";".join(problems)
    row["status"] = "fail" if failed else ("warn" if problems else "ok")
    return row


def main():
    parser = argparse.ArgumentParser(description="Check receptors and ligands before docking")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--out", default=".")
    parser.add_argument("--min-hydrogens", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
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
            row = check_complex(cid, *found[cid], args.min_hydrogens, args.seed)
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
    if tally["warn"] or tally["fail"]:
        print("re-run split_complexes.py to regenerate ligands flagged as repaired or aromaticity_lost", file=sys.stderr)


if __name__ == "__main__":
    main()
