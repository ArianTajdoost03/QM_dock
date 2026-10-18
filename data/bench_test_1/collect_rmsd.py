import argparse
import glob
import math
import os
import re

import pandas as pd

from bench_common import find_complexes, load_table, read_ids, read_json
from qmdock.config import POSES_FILE, STATE_PREFIX
from rmsd import best_rmsd, load_mols

TOP_N = (3, 5, 10)


def state_dirs(run_dir):
    if os.path.exists(os.path.join(run_dir, POSES_FILE)):
        return [run_dir]
    found = [d for d in glob.glob(os.path.join(run_dir, STATE_PREFIX + "*")) if os.path.exists(os.path.join(d, POSES_FILE))]
    return sorted(found, key=lambda d: int(re.search(re.escape(STATE_PREFIX) + r"(\d+)$", d).group(1)))


def energy_of(mol):
    try:
        return float(mol.GetProp("e_final"))
    except (KeyError, ValueError):
        return math.nan


def score_state(ref, state_dir):
    rows = []
    for rank, mol in enumerate(load_mols(os.path.join(state_dir, POSES_FILE)), 1):
        try:
            value = best_rmsd(ref, mol)[0]
        except ValueError:
            value = math.nan
        rows.append({"rank": rank, "e_final": energy_of(mol), "rmsd": float(value)})
    return rows


def finite_min(values):
    values = [v for v in values if not math.isnan(v)]
    return min(values) if values else math.nan


def evaluate(cid, run_dir, crystal_fallback, reference, select):
    row = {"id": cid, "metal": reference.get(cid, {}).get("metal", ""), "logKd": math.nan, "status": ""}
    try:
        row["logKd"] = float(reference[cid]["logKd"])
    except (KeyError, ValueError):
        pass
    pose_records = []
    if not os.path.isdir(run_dir):
        row["status"] = "not_run"
        return row, pose_records
    info = read_json(os.path.join(run_dir, "run_info.json")) or {}
    prep = read_json(os.path.join(run_dir, "prep.json")) or {}
    row.update(n_heavy=prep.get("n_heavy"), n_rot=prep.get("n_rot"), seconds=info.get("seconds"), returncode=info.get("returncode"))
    states = state_dirs(run_dir)
    if not states:
        row["status"] = "failed" if info.get("returncode") not in (0, None) or info.get("error") else "no_poses"
        return row, pose_records
    crystal = prep.get("crystal_ligand")
    if not crystal or not os.path.exists(crystal):
        crystal = crystal_fallback
    ref = load_mols(crystal)[0]
    per_state = []
    for state_dir in states:
        name = os.path.basename(state_dir) if state_dir != run_dir else STATE_PREFIX + "0"
        poses = score_state(ref, state_dir)
        per_state.append((name, poses))
        for p in poses:
            pose_records.append(dict(id=cid, state=name, **p))
    if all(math.isnan(p["rmsd"]) for _, poses in per_state for p in poses):
        row["status"] = "rmsd_error"
        return row, pose_records
    if select == "pooled":
        merged = [(name_, p) for name_, poses_ in per_state for p in poses_]
        merged.sort(key=lambda item: item[1]["e_final"] if not math.isnan(item[1]["e_final"]) else math.inf)
        name, first = merged[0]
        ranked = [p for _, p in merged]
        row.update(selected_state=name, top1_rank_in_state=first["rank"])
    else:
        if select == "energy":
            tops = [poses_[0]["e_final"] if poses_ else math.nan for _, poses_ in per_state]
            keyed = [(t if not math.isnan(t) else math.inf, i) for i, t in enumerate(tops)]
            chosen = min(keyed)[1]
        else:
            chosen = 0
        name, ranked = per_state[chosen]
        row.update(selected_state=name, top1_rank_in_state=1)
    row.update(
        status="ok", n_states=len(per_state),
        n_poses=sum(len(p) for _, p in per_state),
        top1_rmsd=ranked[0]["rmsd"], top1_e_final=ranked[0]["e_final"],
    )
    row["top1_e_per_heavy"] = row["top1_e_final"] / row["n_heavy"] if row.get("n_heavy") else math.nan
    for n in TOP_N:
        row["rmsd_top%d_min" % n] = finite_min([p["rmsd"] for p in ranked[:n]])
    best = None
    for state_name, state_poses in per_state:
        for p in state_poses:
            if not math.isnan(p["rmsd"]) and (best is None or p["rmsd"] < best[0]):
                best = (p["rmsd"], state_name, p["rank"])
    row.update(best_pose_rmsd=best[0], best_pose_state=best[1], best_pose_rank=best[2])
    row["oracle_state_top1_rmsd"] = finite_min([poses_[0]["rmsd"] for _, poses_ in per_state if poses_])
    return row, pose_records


def main():
    parser = argparse.ArgumentParser(description="RMSD of every docked complex against the crystal ligand")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_table.csv"))
    parser.add_argument("--ids")
    parser.add_argument("--state-select", choices=("energy", "first", "pooled"), default="energy")
    parser.add_argument("--out")
    args = parser.parse_args()

    runs = os.path.abspath(args.runs)
    out = os.path.abspath(args.out) if args.out else runs
    os.makedirs(out, exist_ok=True)
    reference = load_table(args.reference)
    crystals = {cid: lig for cid, (_, lig) in find_complexes(args.data).items()}
    if args.ids:
        ids = read_ids(args.ids)
    else:
        ids = sorted(set(crystals) & {d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d))})

    rows, poses = [], []
    for n, cid in enumerate(ids, 1):
        row, records = evaluate(cid, os.path.join(runs, cid), crystals.get(cid), reference, args.state_select)
        rows.append(row)
        poses.extend(records)
        print("[%d/%d] %s %s %s" % (n, len(ids), cid, row["status"], "%.2f" % row["top1_rmsd"] if "top1_rmsd" in row else ""), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "rmsd_results.csv"), index=False)
    pd.DataFrame(poses).to_csv(os.path.join(out, "pose_rmsd.csv"), index=False)
    ok = table[table.status == "ok"]
    print("complexes: %d, scored: %d" % (len(table), len(ok)))
    if len(ok):
        print("top-1 success (<=2 A): %.1f%% of scored, %.1f%% of all" % (
            100 * (ok.top1_rmsd <= 2).mean(), 100 * (ok.top1_rmsd <= 2).sum() / len(table)))
        print("best-pose success (<=2 A): %.1f%% of scored, %.1f%% of all" % (
            100 * (ok.best_pose_rmsd <= 2).mean(), 100 * (ok.best_pose_rmsd <= 2).sum() / len(table)))
    print("wrote %s" % os.path.join(out, "rmsd_results.csv"))


if __name__ == "__main__":
    main()
