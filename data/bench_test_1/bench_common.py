import contextlib
import csv
import io
import json
import os
import shlex
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from qmdock.cli import build_parser

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
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            build_parser("dock.py").parse_args(
                ["--receptor", "r.pdb", "--ligand", "l.sdf", "--center", "0", "0", "0", "--out", "o", *tokens])
    except SystemExit:
        raise ValueError("qmdock does not accept these options (%s)" % captured.getvalue().strip().splitlines()[-1]) from None
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


PROCESS_GB = 2.5
WORKER_GB = 1.5
MEMORY_USE = 0.9


def plan_jobs(jobs, total_workers, available_gb, fixed_workers=None):
    jobs = max(1, jobs)
    while True:
        per = fixed_workers or max(1, total_workers // jobs)
        if jobs == 1 or available_gb is None or jobs * (PROCESS_GB + per * WORKER_GB) <= available_gb * MEMORY_USE:
            return jobs, per
        jobs -= 1


def option_value(tokens, name):
    for i, t in enumerate(tokens):
        if t == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None
