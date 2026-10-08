import json
import os
import time
import traceback
from dataclasses import dataclass, replace

from qmdock.chem.receptor import load_receptor
from qmdock.pipeline.qmstages import Session
from qmdock.pipeline.run import dock_molecule
from qmdock.qm.engine import preflight
from qmdock.qm.pool import Runner, default_workers
from qmdock.screening.ligands import read_ligands
from qmdock.screening.report import ligand_result, ranked_table, write_best_poses, write_table
from qmdock.ui import bar, log, log_to


@dataclass
class ScreenOptions:
    ligands: str
    resume: bool = False
    retry_failed: bool = False
    max_heavy: int = 60
    rank_by: str = "energy"
    top: int = 10


def check_inputs(cfg, opts):
    for label, path in (("receptor", cfg.receptor), ("ligand library", opts.ligands)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"{label} is empty: {path}")
    if not cfg.centers and not cfg.auto_sites:
        raise ValueError("a pocket is required: give --center X Y Z (or --auto-sites)")
    os.makedirs(cfg.out, exist_ok=True)
    probe = os.path.join(cfg.out, ".write_test")
    try:
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise OSError(f"cannot write to the output folder {cfg.out}: {exc}")


def saved_result(ldir):
    path = os.path.join(ldir, "result.json")
    if os.path.isfile(path):
        try:
            with open(path) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None
    return None


def save_result(ldir, result):
    os.makedirs(ldir, exist_ok=True)
    with open(os.path.join(ldir, "result.json"), "w") as handle:
        json.dump(result, handle, indent=2, default=str)


def process(cfg, opts, rec, receptor, runner, session):
    ldir = os.path.join(cfg.out, "ligands", f"{rec.index + 1:05d}_{rec.name}")
    if opts.resume:
        old = saved_result(ldir)
        if old is not None and (old.get("status") == "ok" or not opts.retry_failed):
            old["resumed"] = True
            return old
    base = {"name": rec.name, "index": rec.index, "heavy_atoms": rec.heavy_atoms, "input_smiles": rec.input_smiles,
            "notes": "; ".join(rec.notes), "dir": ldir}
    if rec.status != "ready":
        result = dict(base, status=rec.status, error=rec.error)
        save_result(ldir, result)
        return result
    os.makedirs(ldir, exist_ok=True)
    sub = replace(cfg, out=ldir, ligand_charge=None)
    t0 = time.time()
    rows = []
    try:
        with log_to(os.path.join(ldir, "log.txt")):
            try:
                rows = dock_molecule(sub, rec.mol, receptor, runner, session)
            except Exception as exc:
                log(traceback.format_exc())
                rows = [{"error": f"{type(exc).__name__}: {exc}"[:300]}]
    except OSError as exc:
        rows = [{"error": f"file error: {exc}"[:300]}]
    result = ligand_result(rec, rows, time.time() - t0, ldir)
    save_result(ldir, result)
    return result


def screen(cfg, opts):
    check_inputs(cfg, opts)
    records = read_ligands(opts.ligands, opts.max_heavy, cfg.seed)
    n_bad = sum(1 for r in records if r.status != "ready")
    log(f"read {len(records)} ligand records from {opts.ligands} ({n_bad} will not be docked)")
    preflight(cfg.solvent, cfg.solvation_model)
    receptor = load_receptor(cfg.receptor, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    workers = cfg.workers or default_workers()
    log(f"CPU workers: {workers}; protonation: {cfg.protonation}"
        f"{', joint states' if cfg.joint_states and cfg.protonation != 'none' else ''}")
    results, interrupted = [], False
    table_path = os.path.join(cfg.out, "screening_results.csv")
    try:
        with Runner(workers) as runner:
            session = Session(cfg, receptor, runner)
            progress = bar(len(records), "screening", unit="ligand", leave=True)
            best = None
            for rec in records:
                result = process(cfg, opts, rec, receptor, runner, session)
                results.append(result)
                if result.get("status") == "ok" and (best is None or result["e_final_kcal"] < best[1]):
                    best = (result["name"], result["e_final_kcal"])
                if result.get("status") not in ("ok", None):
                    log(f"  {result['name']}: {result['status']} - {result.get('error', '')[:120]}")
                progress.update(1)
                progress.set_postfix(ok=sum(r.get("status") == "ok" for r in results),
                                     bad=sum(r.get("status") != "ok" for r in results),
                                     best=f"{best[0]} {best[1]:.1f}" if best else "-")
                write_table(table_path, results, opts.rank_by)
            progress.close()
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted; writing the results so far (rerun with --resume to continue)")
    finally:
        if results:
            write_table(table_path, results, opts.rank_by)
            n_poses = write_best_poses(os.path.join(cfg.out, "best_poses.sdf"), results, opts.rank_by)
            log(f"wrote {table_path} and {n_poses} best poses to best_poses.sdf")
    summarize(results, opts)
    return results, interrupted


def summarize(results, opts):
    if not results:
        return
    counts = {}
    for r in results:
        counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
    log("status: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))))
    table = ranked_table(results, opts.rank_by)
    top = table[table.status == "ok"].head(opts.top)
    if len(top):
        log(f"top {len(top)} by {'energy per heavy atom' if opts.rank_by == 'per-atom' else 'energy'} "
            "(kcal/mol, best protonation state per ligand):")
        for _, r in top.iterrows():
            log(f"  {int(r['rank']):3d}  {str(r['name'])[:28]:28s} {r['e_final_kcal']:9.2f}  state {int(r['best_state'])}  "
                f"{str(r['best_state_smiles'])[:40]}")
