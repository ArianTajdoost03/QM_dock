import csv
import itertools
import json
import os
import shlex

from rdkit import Chem

RESERVED_OPTIONS = ("--receptor", "--ligand", "--center", "--out")


def read_options(path=None, text=None):
    if text is None:
        with open(path) as fh:
            text = " ".join(
                line.strip() for line in fh
                if line.strip() and not line.lstrip().startswith("#")
            )
    tokens = shlex.split(text)
    clash = [t for t in tokens if t in RESERVED_OPTIONS]
    if clash:
        raise ValueError("these options are set by the runner and must not appear in the options: %s" % ", ".join(clash))
    return tokens


def find_complexes(data_dir):
    found = {}
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.endswith("_protein.pdb"):
                continue
            cid = name[:-len("_protein.pdb")]
            ligand = os.path.join(root, cid + "_ligand.sdf")
            if os.path.exists(ligand):
                found[cid.lower()] = (os.path.abspath(os.path.join(root, name)), os.path.abspath(ligand))
    return dict(sorted(found.items()))


def read_ids(path):
    with open(path) as fh:
        return [line.strip().lower() for line in fh if line.strip()]


def load_table(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {row["id"].strip().lower(): row for row in csv.DictReader(fh)}


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def write_json(path, data):
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def read_crystal_ligand(path):
    mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=True)
    if mol is not None:
        return mol, 0
    raw = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    if raw is None:
        raise ValueError("crystal ligand cannot be parsed")
    raw.UpdatePropertyCache(strict=False)
    candidates = [a.GetIdx() for a in raw.GetAtoms()
                  if a.GetSymbol() == "N" and a.GetDegree() == 2 and a.GetFormalCharge() == 0
                  and a.GetTotalNumHs() == 0
                  and (a.GetIsAromatic() or any(b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds()))]
    for size in range(1, min(len(candidates), 3) + 1):
        for combo in itertools.combinations(candidates, size):
            trial = Chem.RWMol(raw)
            for idx in combo:
                atom = trial.GetAtomWithIdx(idx)
                atom.SetNumExplicitHs(1)
                atom.SetNoImplicit(True)
            try:
                Chem.SanitizeMol(trial)
            except Exception:
                continue
            return trial.GetMol(), size
    raise ValueError("crystal ligand cannot be sanitized")
