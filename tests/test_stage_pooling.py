from types import SimpleNamespace

import numpy as np

from qmdock.config import Config
from qmdock.pipeline.qmstages import Session
from test_pockets import moved, setup


class RecordingRunner:
    def __init__(self, fail_pockets=False):
        self.calls = []
        self.fail_pockets = fail_pockets

    def map(self, jobs, label):
        self.calls.append([j["key"] for j in jobs])
        out = {}
        for j in jobs:
            ok = not (self.fail_pockets and j["key"][0] == "pocket")
            out[j["key"]] = {"ok": ok, "energy": -100.0, "attempt": 0, "error": "" if ok else "scf"}
        return out


def make(runner):
    rec, ligand, coords = setup()
    lig = SimpleNamespace(numbers=np.array([a.GetAtomicNum() for a in ligand.GetAtoms()]), charge=0,
                          n_atoms=ligand.GetNumAtoms(), e_ref=-50.0)
    return Session(Config(receptor="r", ligand="l"), rec, runner), lig, coords


def test_pockets_and_complexes_go_to_the_pool_in_one_call():
    runner = RecordingRunner()
    session, lig, coords = make(runner)
    poses = {0: coords, 1: moved(coords, 3.0, 40, 5)}
    rows = session.evaluate(lig, [0, 1], poses, {0: -50.0, 1: -50.0}, None)
    assert len(runner.calls) == 1
    kinds = [k[0] for k in runner.calls[0]]
    assert kinds.count("pocket") == 2 and kinds.count("complex") == 2
    assert all(r["status"] == "ok" for r in rows.values())


def test_failed_pocket_energy_marks_every_pose_of_that_pocket_as_no_pocket():
    runner = RecordingRunner(fail_pockets=True)
    session, lig, coords = make(runner)
    rows = session.evaluate(lig, [0, 1], {0: coords, 1: coords + 0.05}, {0: -50.0, 1: -50.0}, None)
    assert [rows[p]["status"] for p in (0, 1)] == ["no_pocket", "no_pocket"]
    again = RecordingRunner()
    session.runner = again
    rows = session.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    assert rows[0]["status"] == "no_pocket"
    assert [k[0] for call in again.calls for k in call] == []


def test_known_pocket_is_not_recomputed_and_only_complexes_are_submitted():
    runner = RecordingRunner()
    session, lig, coords = make(runner)
    session.evaluate(lig, [0], {0: coords}, {0: -50.0}, None)
    runner.calls.clear()
    session.evaluate(lig, [0], {0: coords + 0.02}, {0: -50.0}, None)
    assert [[k[0] for k in call] for call in runner.calls] == [["complex"]]
