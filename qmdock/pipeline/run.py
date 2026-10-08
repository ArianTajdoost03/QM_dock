import json
import os
import time
from dataclasses import replace

import pandas as pd

from qmdock.chem.ligand import load_ligand
from qmdock.chem.protonation import ligand_states
from qmdock.chem.receptor import load_receptor
from qmdock.config import VERSION
from qmdock.pipeline.estimate import report_dry_run
from qmdock.pipeline.fast import get_pool, select_poses
from qmdock.pipeline.joint import transfer_selection
from qmdock.pipeline.ligandprep import describe_ligand, prepare_conformers
from qmdock.pipeline.outputs import write_results
from qmdock.pipeline.qmstages import Session
from qmdock.qm.engine import preflight
from qmdock.qm.pool import Runner, default_workers, log


def dock_state(cfg, mol, receptor, runner, session):
    timings, warnings = {}, []
    os.makedirs(cfg.out, exist_ok=True)
    lig = describe_ligand(cfg, mol)
    lig, pool = get_pool(cfg, receptor, lig, runner, warnings, timings)
    sel = select_poses(cfg, receptor, lig, pool)
    if cfg.dry_run:
        report_dry_run(cfg, receptor, sel, lig, runner.workers)
        return None
    res = session.run(lig, sel)
    return write_results(cfg, cfg.out, lig, sel, res, timings, warnings)[0]


def state_row(i, st, sub):
    return {"state": i, "smiles": st["smiles"], "charge": st["charge"], "label": st["label"], "dir": sub.out}


def add_summary(row, out):
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    row.update(best_pose=summary["best_pose"], best_e_final_kcal=summary["best_e_final_kcal"],
               best_level=summary["best_level"], gap_to_next_family_kcal=summary["gap_to_next_family_kcal"],
               n_transfer_rejected=summary.get("n_transfer_rejected"))


def run_sequential(cfg, states, receptor, runner, session):
    rows = []
    for i, st in enumerate(states):
        sub = replace(cfg, out=os.path.join(cfg.out, f"state_{i}"), ligand_charge=None)
        row = state_row(i, st, sub)
        log(f"=== protonation state {i}/{len(states) - 1}: {st['smiles']} ===")
        try:
            dock_state(sub, st["mol"], receptor, runner, session)
            if not cfg.dry_run:
                add_summary(row, sub.out)
        except (RuntimeError, ValueError) as exc:
            row["error"] = str(exc)[:300]
            log(f"  state {i} failed: {row['error']}")
        rows.append(row)
    return rows


def run_joint(cfg, states, receptor, runner, session):
    base_cfg = replace(cfg, ligand_charge=None)
    ligs = [describe_ligand(base_cfg, st["mol"]) for st in states]
    t_idx = max(range(len(states)), key=lambda i: states[i]["mol"].GetNumAtoms())
    log(f"joint search on state {t_idx} ({states[t_idx]['smiles']}), the most protonated; "
        "the pose search runs once and poses are transferred to the other states")
    timings, warnings = {}, []
    os.makedirs(cfg.out, exist_ok=True)
    lig_t, pool = get_pool(base_cfg, receptor, ligs[t_idx], runner, warnings, timings)
    sel_t = select_poses(base_cfg, receptor, lig_t, pool)
    if cfg.dry_run:
        report_dry_run(base_cfg, receptor, sel_t, lig_t, runner.workers)
        return [state_row(i, st, replace(cfg, out=os.path.join(cfg.out, f"state_{i}"))) for i, st in enumerate(states)]
    rows = []
    for i, st in enumerate(states):
        sub = replace(cfg, out=os.path.join(cfg.out, f"state_{i}"), ligand_charge=None)
        row = state_row(i, st, sub)
        log(f"=== protonation state {i}/{len(states) - 1}: {st['smiles']} ===")
        try:
            if i == t_idx:
                lig_i, sel_i = lig_t, sel_t
            else:
                lig_i, seconds = prepare_conformers(sub, ligs[i], runner)
                sel_i, info = transfer_selection(sub, receptor, lig_t, lig_i, sel_t, pool)
                row.update(info)
            res = session.run(lig_i, sel_i)
            write_results(sub, sub.out, lig_i, sel_i, res, timings, warnings)
            add_summary(row, sub.out)
        except (RuntimeError, ValueError) as exc:
            row["error"] = str(exc)[:300]
            log(f"  state {i} failed: {row['error']}")
        rows.append(row)
    return rows


def dock_molecule(cfg, base, receptor, runner, session):
    states = ligand_states(base, cfg.protonation, cfg.ph_min, cfg.ph_max, cfg.max_states, cfg.seed)
    if len(states) == 1:
        row = state_row(0, states[0], cfg)
        dock_state(cfg, states[0]["mol"], receptor, runner, session)
        add_summary(row, cfg.out)
        return [row]
    os.makedirs(cfg.out, exist_ok=True)
    for i, st in enumerate(states):
        log(f"state {i}: {st['smiles']} (charge {st['charge']}, {st['label']})")
    return (run_joint if cfg.joint_states else run_sequential)(cfg, states, receptor, runner, session)


def run(cfg):
    log(f"qmdock {VERSION}")
    preflight(cfg.solvent, cfg.solvation_model)
    base = load_ligand(cfg.ligand)
    if not cfg.centers and not cfg.auto_sites:
        cfg = replace(cfg, centers=[tuple(base.GetConformer().GetPositions().mean(axis=0))])
    states = ligand_states(base, cfg.protonation, cfg.ph_min, cfg.ph_max, cfg.max_states, cfg.seed)
    workers = cfg.workers or default_workers()
    log(f"CPU workers: {workers}")
    t = time.time()
    receptor = load_receptor(cfg.receptor, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    for (formula, status), n in sorted(receptor.summary().items()):
        log(f"receptor unit {formula}: {n} x {status}")
    log(f"receptor loaded in {time.time() - t:.1f}s")
    with Runner(workers) as runner:
        session = Session(cfg, receptor, runner)
        if len(states) == 1:
            return dock_state(cfg, states[0]["mol"], receptor, runner, session)
        os.makedirs(cfg.out, exist_ok=True)
        for i, st in enumerate(states):
            log(f"state {i}: {st['smiles']} (charge {st['charge']}, {st['label']})")
        joint = cfg.joint_states
        rows = (run_joint if joint else run_sequential)(cfg, states, receptor, runner, session)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(cfg.out, "states.csv"), index=False)
    log("energies of different protonation states are not comparable (different numbers of protons); "
        "rank poses within a state and compare states by geometry or with pKa corrections")
    return table
