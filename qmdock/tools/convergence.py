import argparse
import json
import os
from dataclasses import fields, replace

import numpy as np
import pandas as pd
from rdkit import Chem

from qmdock.chem.receptor import load_receptor
from qmdock.chem.spin import cluster_uhf
from qmdock.config import CONFIG_FILE, POSES_FILE, PROTONATED_RECEPTOR_FILE, Config
from qmdock.qm.jobs import energy_job, numbers_of
from qmdock.qm.pool import Runner, default_workers, log
from qmdock.tools.health import spearman


def scan(receptor, cfg, numbers, charge, poses, shells, accuracies, runner):
    rows = []
    for accuracy in accuracies:
        conf = replace(cfg, accuracy=accuracy)
        lig = runner.map([energy_job(("lig", i), numbers, poses[i], charge, conf) for i in range(len(poses))],
                         f"ligand energies, accuracy {accuracy:g}")
        for shell in shells:
            clusters = {i: receptor.build_cluster(poses[i], shell, None, shell) for i in range(len(poses))}
            usable = {i: c for i, c in clusters.items() if c is not None and cluster_uhf(c, conf.uhf) is not None}
            pockets = {c.key: c for c in usable.values()}
            jobs = [energy_job(("pocket", k), numbers_of(c.elements()), c.coords(), c.charge, conf,
                               cluster_uhf(c, conf.uhf)) for k, c in pockets.items()]
            jobs += [energy_job(("complex", i), np.concatenate([numbers_of(c.elements()), numbers]),
                                np.vstack([c.coords(), poses[i]]), c.charge + charge, conf, cluster_uhf(c, conf.uhf))
                     for i, c in usable.items()]
            res = runner.map(jobs, f"shell {shell:g} A, accuracy {accuracy:g}")
            for i, c in usable.items():
                complex_, pocket = res[("complex", i)], res[("pocket", c.key)]
                if not (complex_["ok"] and pocket["ok"] and lig[("lig", i)]["ok"]):
                    continue
                e_int = (complex_["energy"] - pocket["energy"] - lig[("lig", i)]["energy"]) * 627.509474
                rows.append({"pose": i, "shell": shell, "accuracy": accuracy, "n_atoms": c.n_atoms + len(numbers),
                             "e_int": e_int, "seconds": complex_.get("seconds", 0.0)})
    return pd.DataFrame(rows)


def compare(df, reference=None):
    if df.empty:
        return df
    if reference is None:
        top = df[df["accuracy"] == df["accuracy"].min()]
        reference = (float(top["shell"].max()), float(top["accuracy"].min()))
    ref = df[(df["shell"] == reference[0]) & (df["accuracy"] == reference[1])].set_index("pose")
    out = []
    for (shell, accuracy), g in df.groupby(["shell", "accuracy"]):
        g = g.set_index("pose")
        common = g.index.intersection(ref.index)
        delta = (g.loc[common, "e_int"] - ref.loc[common, "e_int"]).abs()
        out.append({"shell": shell, "accuracy": accuracy, "poses": len(common),
                    "n_atoms_mean": float(g["n_atoms"].mean()), "seconds_mean": float(g["seconds"].mean()),
                    "mean_abs_dev": float(delta.mean()) if len(common) else float("nan"),
                    "max_abs_dev": float(delta.max()) if len(common) else float("nan"),
                    "spearman": spearman(g.loc[common, "e_int"], ref.loc[common, "e_int"]),
                    "time_vs_reference": float(g["seconds"].sum() / max(ref["seconds"].sum(), 1e-9))})
    table = pd.DataFrame(out)
    table["is_reference"] = (table["shell"] == reference[0]) & (table["accuracy"] == reference[1])
    return table


def load_config(state_dir):
    for d in (state_dir, os.path.dirname(os.path.abspath(state_dir))):
        path = os.path.join(d, CONFIG_FILE)
        if os.path.exists(path):
            data = json.load(open(path))
            names = {f.name for f in fields(Config)}
            return Config(**{k: v for k, v in data.items() if k in names}), d
    raise SystemExit(f"no {CONFIG_FILE} found in {state_dir} or its parent")


def main():
    p = argparse.ArgumentParser(description="How much do shell size and QM accuracy change the interaction energy?")
    p.add_argument("state_dir")
    p.add_argument("--poses", type=int, default=5)
    p.add_argument("--shells", type=float, nargs="+", default=[3.5, 4.0, 5.0, 6.0, 8.0])
    p.add_argument("--accuracies", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--receptor")
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    cfg, top = load_config(a.state_dir)
    receptor_path = a.receptor or os.path.join(top, PROTONATED_RECEPTOR_FILE)
    if not os.path.exists(receptor_path):
        receptor_path = cfg.receptor
    receptor = load_receptor(receptor_path, cfg.min_unit_atoms, cfg.require_metal, remove_waters=not cfg.keep_waters)
    mols = [m for m in Chem.SDMolSupplier(os.path.join(a.state_dir, POSES_FILE), removeHs=False) if m is not None]
    mols = mols[:a.poses]
    numbers = np.array([atom.GetAtomicNum() for atom in mols[0].GetAtoms()], dtype=np.int32)
    charge = sum(atom.GetFormalCharge() for atom in mols[0].GetAtoms())
    poses = [m.GetConformer().GetPositions() for m in mols]
    with Runner(a.workers or default_workers()) as runner:
        df = scan(receptor, cfg, numbers, charge, poses, a.shells, a.accuracies, runner)
    table = compare(df)
    df.to_csv(os.path.join(a.state_dir, "convergence_energies.csv"), index=False)
    table.to_csv(os.path.join(a.state_dir, "convergence_summary.csv"), index=False)
    log(table.to_string(index=False))


if __name__ == "__main__":
    main()
