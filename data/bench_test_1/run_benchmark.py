import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from bench_common import (find_complexes, load_table, read_crystal_ligand, read_ids, read_json, read_options,
                          write_json)

HERE = os.path.dirname(os.path.abspath(__file__))


def fresh_ligand(crystal_path, name, seed):
    mol, repaired = read_crystal_ligand(crystal_path)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
    heavy = Chem.RemoveHs(mol)
    Chem.AssignStereochemistryFrom3D(heavy)
    smiles = Chem.MolToSmiles(heavy)
    built = Chem.MolFromSmiles(smiles)
    if built is None:
        raise ValueError("SMILES could not be rebuilt: %s" % smiles)
    fresh = Chem.AddHs(built)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(fresh, params) < 0:
        raise ValueError("3D embedding failed")
    if AllChem.MMFFHasAllMoleculeParams(fresh):
        AllChem.MMFFOptimizeMolecule(fresh, maxIters=2000)
    fresh.SetProp("_Name", name)
    info = {
        "smiles": smiles,
        "n_heavy": heavy.GetNumAtoms(),
        "n_rot": int(rdMolDescriptors.CalcNumRotatableBonds(heavy)),
        "charge": int(Chem.GetFormalCharge(heavy)),
        "repaired_aromatic_nh": repaired,
    }
    return fresh, xyz.mean(axis=0), info


def clean_run_dir(run_dir):
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if os.path.isdir(path) and name.startswith("state_"):
            shutil.rmtree(path)
        elif name in ("poses.sdf", "results.csv", "summary.json", "states.csv", "best_complex.xyz", "clusters.csv", "run_info.json"):
            os.remove(path)


def dock_one(task):
    cid, receptor, ligand, args, tokens = task
    run_dir = os.path.join(args.out, cid)
    os.makedirs(run_dir, exist_ok=True)
    info_path = os.path.join(run_dir, "run_info.json")
    previous = read_json(info_path)
    if not args.force and previous and previous.get("returncode") == 0:
        return cid, "skipped_done", float(previous.get("seconds", 0.0))
    clean_run_dir(run_dir)
    try:
        fresh, center, info = fresh_ligand(ligand, cid, args.seed)
    except Exception as exc:
        write_json(info_path, {"returncode": None, "error": "prep: %s" % exc, "seconds": 0.0})
        return cid, "prep_failed: %s" % exc, 0.0
    ligand_path = os.path.join(run_dir, "input_ligand.sdf")
    writer = Chem.SDWriter(ligand_path)
    writer.write(fresh)
    writer.close()
    write_json(os.path.join(run_dir, "prep.json"), dict(
        info, id=cid, receptor=receptor, crystal_ligand=ligand, center=[float(v) for v in center]))
    command = [
        args.python, os.path.join(args.qmdock, "dock.py"),
        "--receptor", receptor, "--ligand", os.path.abspath(ligand_path),
        "--center", *["%.3f" % v for v in center],
        "--out", os.path.abspath(run_dir), *tokens,
    ]
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in command))
        return cid, "dry_run", 0.0
    start = time.time()
    returncode = None
    error = None
    with open(os.path.join(run_dir, "dock.log"), "w") as log:
        try:
            returncode = subprocess.run(
                command, cwd=args.qmdock, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout or None).returncode
        except subprocess.TimeoutExpired:
            error = "timeout after %d s" % args.timeout
    seconds = time.time() - start
    write_json(info_path, {"returncode": returncode, "error": error, "seconds": seconds, "command": command})
    status = "ok" if returncode == 0 else (error or "exit code %s" % returncode)
    return cid, status, seconds


def main():
    parser = argparse.ArgumentParser(description="Redock every complex with qmdock")
    parser.add_argument("--data", required=True)
    parser.add_argument("--qmdock", required=True)
    parser.add_argument("--out", default="runs/default")
    parser.add_argument("--options", default=os.path.join(HERE, "dock_options.txt"))
    parser.add_argument("--opts")
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-single-bond-ligands", action="store_true")
    args = parser.parse_args()

    args.data = os.path.abspath(args.data)
    args.qmdock = os.path.abspath(args.qmdock)
    args.out = os.path.abspath(args.out)
    tokens = read_options(path=None if args.opts else args.options, text=args.opts)

    complexes = find_complexes(args.data)
    if args.ids:
        wanted = read_ids(args.ids)
        missing = [i for i in wanted if i not in complexes]
        if missing:
            print("%d ids not found in data folder (first few: %s)" % (len(missing), ", ".join(missing[:5])))
        complexes = {i: complexes[i] for i in wanted if i in complexes}

    summary = {}
    for folder in (args.data, os.path.dirname(args.data)):
        summary = load_table(os.path.join(folder, "summary.csv"))
        if summary:
            break
    if summary and not args.allow_single_bond_ligands:
        skipped = [i for i in complexes if summary.get(i, {}).get("ligand_bonds") not in (None, "", "bond_orders_from_template")]
        for i in skipped:
            del complexes[i]
        if skipped:
            print("skipped %d ligands without bond orders: %s" % (len(skipped), ", ".join(skipped)))
    elif not summary:
        print("summary.csv from the split step not found; cannot check ligand bond orders")

    if args.limit:
        complexes = dict(list(complexes.items())[:args.limit])
    if not complexes:
        sys.exit("no complexes to dock")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "options_used.txt"), "w") as fh:
        fh.write(" ".join(shlex.quote(t) for t in tokens) + "\n")

    tasks = [(cid, rec, lig, args, tokens) for cid, (rec, lig) in complexes.items()]
    print("docking %d complexes with options: %s" % (len(tasks), " ".join(tokens)))
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for n, (cid, status, seconds) in enumerate(pool.map(dock_one, tasks), 1):
            print("[%d/%d] %s %s (%.0f s)" % (n, len(tasks), cid, status, seconds), flush=True)
            rows.append((cid, status, seconds))
    with open(os.path.join(args.out, "status.csv"), "w") as fh:
        fh.write("id,status,seconds\n")
        for cid, status, seconds in rows:
            fh.write("%s,%s,%.1f\n" % (cid, status.replace(",", ";"), seconds))
    with open(os.path.join(args.out, "centers.csv"), "w") as fh:
        fh.write("id,x,y,z\n")
        for cid, _, _ in rows:
            prep = read_json(os.path.join(args.out, cid, "prep.json"))
            if prep:
                fh.write("%s,%.3f,%.3f,%.3f\n" % ((cid,) + tuple(prep["center"])))
    good = sum(s in ("ok", "skipped_done", "dry_run") for _, s, _ in rows)
    print("finished: %d ok, %d failed" % (good, len(rows) - good))


if __name__ == "__main__":
    main()
