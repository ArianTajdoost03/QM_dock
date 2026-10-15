import argparse
import os

import numpy as np
import pandas as pd

from bench_common import find_complexes, read_ids, read_json
from collect_rmsd import state_dirs
from rmsd import load_mols, skeleton


def pool_rmsds(ref, mol, coords, max_matches=2000):
    rmol, rxyz = skeleton(ref)
    pmol, _ = skeleton(mol)
    matches = np.array(pmol.GetSubstructMatches(rmol, uniquify=False, useChirality=False, maxMatches=max_matches))
    if matches.size == 0:
        raise ValueError("heavy-atom connectivity differs between the molecules")
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    H = coords[:, heavy, :]
    out = np.empty(len(coords))
    step = max(1, int(2e7 // (len(matches) * H.shape[1] * 3)))
    for start in range(0, len(H), step):
        Q = H[start:start + step][:, matches, :]
        out[start:start + step] = np.sqrt(((Q - rxyz) ** 2).sum(-1).mean(-1)).min(axis=1)
    return out


def pool_groups(run_dir):
    shared = os.path.join(run_dir, "fast_stage.npz")
    groups = {}
    for state in state_dirs(run_dir):
        own = os.path.join(state, "fast_stage.npz")
        path = own if os.path.exists(own) else shared
        if os.path.exists(path):
            groups.setdefault(path, []).append(state)
    return groups


def reference_for(path, states, n_atoms):
    for state in states:
        mols = load_mols(os.path.join(state, "poses.sdf"))
        if mols[0].GetNumAtoms() == n_atoms:
            return mols[0]
    return None


def attribute(cid, run_dir, crystal, threshold):
    row = {"id": cid, "status": "", "diagnosis": ""}
    if not os.path.isdir(run_dir):
        row["status"] = "not_run"
        return row
    groups = pool_groups(run_dir)
    if not groups:
        row["status"] = "no_pool_cache"
        return row
    prep = read_json(os.path.join(run_dir, "prep.json")) or {}
    path_ref = prep.get("crystal_ligand")
    if not path_ref or not os.path.exists(path_ref):
        path_ref = crystal
    ref = load_mols(path_ref)[0]
    best = None
    for path, states in groups.items():
        z = np.load(path)
        coords, score = z["pool_coords"], z["pool_score"]
        mol = reference_for(path, states, coords.shape[1])
        if mol is None:
            continue
        rmsd = pool_rmsds(ref, mol, coords)
        fast_rank = np.empty(len(score), int)
        fast_rank[np.argsort(score)] = np.arange(1, len(score) + 1)
        native = rmsd <= threshold
        entry = {"pool_size": len(score), "pool_best_rmsd": float(rmsd.min()),
                 "pool_best_fast_rank": int(fast_rank[np.argmin(rmsd)]), "n_pool_native": int(native.sum()),
                 "best_native_fast_rank": int(fast_rank[native].min()) if native.any() else None}
        tables = []
        for state in states:
            results = os.path.join(state, "results.csv")
            if os.path.exists(results):
                table = pd.read_csv(results)
                table["state"] = os.path.basename(state) if state != run_dir else "state_0"
                tables.append(table)
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        entry["pool_rmsd"] = rmsd
        entry["table"] = table
        if best is None or entry["pool_best_rmsd"] < best["pool_best_rmsd"]:
            best = entry
    if best is None:
        row["status"] = "no_reference_mol"
        return row
    table = best.pop("table")
    rmsd = best.pop("pool_rmsd")
    row.update(best)
    row["status"] = "ok"
    if table.empty or "pool_index" not in table or table.status.eq("ok").sum() == 0:
        row["diagnosis"] = "search" if row["pool_best_rmsd"] > threshold else "pool_has_native"
        row["status"] = "no_qm_results" if table.empty else "no_pool_index"
        return row
    ok = table[table.status == "ok"].copy()
    ok["pool_rmsd"] = rmsd[ok.pool_index.astype(int).to_numpy()]
    row["n_qm_evaluated"] = len(ok)
    native_qm = ok[ok.pool_rmsd <= threshold]
    row["n_native_qm"] = len(native_qm)
    top = ok.loc[ok.e_final.idxmin()]
    row["top1_pool_rmsd"] = float(top.pool_rmsd)
    row["top1_state"] = top.state
    state_tops = [g.loc[g.e_final.idxmin()] for _, g in ok.groupby("state")]
    row["oracle_state_top1_pool_rmsd"] = float(min(t.pool_rmsd for t in state_tops))
    if row["pool_best_rmsd"] > threshold:
        row["diagnosis"] = "search"
    elif native_qm.empty:
        row["diagnosis"] = "selection"
    elif top.pool_rmsd <= threshold:
        row["diagnosis"] = "ok"
    elif row["oracle_state_top1_pool_rmsd"] <= threshold:
        row["diagnosis"] = "state_selection"
    else:
        row["diagnosis"] = "qm_ranking"
    return row


def main():
    parser = argparse.ArgumentParser(description="Attribute each docking failure to search, selection or QM ranking")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    runs = os.path.abspath(args.runs)
    crystals = {cid: lig for cid, (_, lig) in find_complexes(args.data).items()}
    ids = read_ids(args.ids) if args.ids else sorted(d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d)))
    rows = []
    for n, cid in enumerate(ids, 1):
        try:
            row = attribute(cid, os.path.join(runs, cid), crystals.get(cid), args.threshold)
        except Exception as exc:
            row = {"id": cid, "status": "error:%s" % exc, "diagnosis": ""}
        rows.append(row)
        print("[%d/%d] %s %s %s" % (n, len(ids), cid, row["status"], row["diagnosis"]), flush=True)
    table = pd.DataFrame(rows)
    out = os.path.abspath(args.out) if args.out else os.path.join(runs, "attribution.csv")
    table.to_csv(out, index=False)
    scored = table[table.diagnosis != ""]
    print("\nattributed %d of %d complexes -> %s" % (len(scored), len(table), out))
    for name, count in scored.diagnosis.value_counts().items():
        print("  %-16s %4d  (%.0f%%)" % (name, count, 100 * count / len(scored)))
    print("  search: no pose within %.1f A was generated | selection: native pose in pool but never sent to QM" % args.threshold)
    print("  qm_ranking: native pose evaluated but not ranked first | state_selection: another protonation state had it first")


if __name__ == "__main__":
    main()
