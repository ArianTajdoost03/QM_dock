import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pandas as pd

from qmdock.chem.hydrogens import load_protonated
from qmdock.chem.ligand import load_ligand
from qmdock.chem.protonation import ligand_states
from qmdock.config import CONFIG_FILE, STATE_PREFIX, STATES_FILE, SUMMARY_FILE
from qmdock.pipeline.estimate import report_dry_run
from qmdock.pipeline.fast import get_pool, select_poses
from qmdock.pipeline.joint import transfer_selection
from qmdock.pipeline.ligandprep import describe_ligand, prepare_conformers
from qmdock.pipeline.outputs import write_results
from qmdock.pipeline.qmstages import Session
from qmdock.pipeline.stateselect import apply_penalty, finalize, prepare
from qmdock.qm.engine import preflight
from qmdock.qm.pool import Runner, default_workers, log
from qmdock.ui import set_label


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
    return write_results(cfg, cfg.out, lig, sel, res, timings, warnings, receptor)[0]


def state_row(i, st, sub, penalty=None):
    row = {"state": i, "smiles": st["smiles"], "charge": st["charge"], "label": st["label"], "dir": sub.out}
    return apply_penalty(row, penalty)


def add_summary(row, out):
    with open(os.path.join(out, SUMMARY_FILE)) as f:
        summary = json.load(f)
    row.update(best_pose=summary["best_pose"], best_e_final_kcal=summary["best_e_final_kcal"],
               best_level=summary["best_level"], gap_to_next_family_kcal=summary["gap_to_next_family_kcal"],
               n_transfer_rejected=summary.get("n_transfer_rejected"))


def run_states(cfg, states, runner, work, allow_parallel=True):
    def labelled(i):
        set_label(f"[state {i}] " if parallel else "")
        try:
            return work(i, states[i])
        finally:
            set_label("")

    parallel = allow_parallel and cfg.parallel_states and len(states) > 1 and runner.workers > 1
    if not parallel:
        return [labelled(i) for i in range(len(states))]
    with ThreadPoolExecutor(len(states)) as ex:
        return list(ex.map(labelled, range(len(states))))


def run_sequential(cfg, states, receptor, runner, session):
    penalties = prepare(cfg, states)

    def work(i, st):
        sub = replace(cfg, out=os.path.join(cfg.out, f"{STATE_PREFIX}{i}"), ligand_charge=None)
        row = state_row(i, st, sub, penalties[i])
        log(f"=== protonation state {i}/{len(states) - 1}: {st['smiles']} ===")
        try:
            dock_state(sub, st["mol"], receptor, runner, session)
            if not cfg.dry_run:
                add_summary(row, sub.out)
        except (RuntimeError, ValueError) as exc:
            row["error"] = str(exc)[:300]
            log(f"  state {i} failed: {row['error']}")
        return row

    return finalize(run_states(cfg, states, runner, work, allow_parallel=False), cfg)


def run_joint(cfg, states, receptor, runner, session):
    base_cfg = replace(cfg, ligand_charge=None)
    penalties = prepare(cfg, states)
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
        return [state_row(i, st, replace(cfg, out=os.path.join(cfg.out, f"{STATE_PREFIX}{i}")), penalties[i])
                for i, st in enumerate(states)]

    def work(i, st):
        sub = replace(cfg, out=os.path.join(cfg.out, f"{STATE_PREFIX}{i}"), ligand_charge=None)
        row = state_row(i, st, sub, penalties[i])
        log(f"=== protonation state {i}/{len(states) - 1}: {st['smiles']} ===")
        try:
            if i == t_idx:
                lig_i, sel_i = lig_t, sel_t
            else:
                lig_i, seconds = prepare_conformers(sub, ligs[i], runner)
                sel_i, info = transfer_selection(sub, receptor, lig_t, lig_i, sel_t, pool)
                row.update(info)
            res = session.run(lig_i, sel_i)
            write_results(sub, sub.out, lig_i, sel_i, res, dict(timings), list(warnings), receptor)
            add_summary(row, sub.out)
        except (RuntimeError, ValueError) as exc:
            row["error"] = str(exc)[:300]
            log(f"  state {i} failed: {row['error']}")
        return row

    return finalize(run_states(cfg, states, runner, work), cfg)


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
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, CONFIG_FILE), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    preflight(cfg.solvent, cfg.solvation_model)
    base = load_ligand(cfg.ligand)
    if not cfg.centers and not cfg.auto_sites:
        cfg = replace(cfg, centers=[tuple(base.GetConformer().GetPositions().mean(axis=0))])
    states = ligand_states(base, cfg.protonation, cfg.ph_min, cfg.ph_max, cfg.max_states, cfg.seed)
    workers = cfg.workers or default_workers()
    log(f"CPU workers: {workers}")
    t = time.time()
    receptor = load_protonated(cfg, cfg.out, log)
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
    table.to_csv(os.path.join(cfg.out, STATES_FILE), index=False)
    if cfg.pka_penalty:
        log("states are compared with a pKa-based population penalty added to e_final; raw energies of "
            "different protonation states are not comparable (see states.csv: pka_penalty_kcal, best_e_corrected_kcal)")
    else:
        log("energies of different protonation states are not comparable (different numbers of protons); "
            "the pKa penalty is switched off, so the selected state is the one with the lowest raw energy")
    return table
