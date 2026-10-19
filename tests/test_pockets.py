from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation

from qmdock.chem.receptor import load_receptor
from qmdock.config import Config
from qmdock.pipeline.qmstages import Session, assign_pockets, pocket_members

CA2 = Path(__file__).parent.parent / "data" / "carbonic anhydrase 2"


def setup():
    rec = load_receptor(str(CA2 / "1A42.pdb"), 10, False)
    ligand = Chem.AddHs(next(iter(Chem.SDMolSupplier(str(CA2 / "ligand.sdf"), removeHs=False))), addCoords=True)
    return rec, ligand, ligand.GetConformer().GetPositions()


def moved(coords, shift, degrees, seed):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3)
    direction *= shift / np.linalg.norm(direction)
    centre = coords.mean(axis=0)
    rotation = Rotation.from_rotvec(axis * np.radians(degrees)).as_matrix()
    return (coords - centre) @ rotation.T + centre + direction


def test_every_pose_keeps_all_of_its_own_contacts_inside_its_pocket():
    rec, ligand, coords = setup()
    poses = {0: coords, 1: moved(coords, 0.5, 5, 1), 2: moved(coords, 1.0, 10, 2), 3: moved(coords, 2.0, 25, 3),
             4: moved(coords, 3.0, 40, 4)}
    cfg = Config(receptor="r", ligand="l")
    infos = assign_pockets(rec, cfg, list(poses), poses, None, {}, ligand.GetNumAtoms())
    for p, coord in poses.items():
        own = rec.build_cluster(coord, cfg.shell, None, cfg.charged_shell)
        assert pocket_members(own) <= pocket_members(infos[p]) and infos[p].electrons() % 2 == 0


def test_sharing_can_be_switched_off_and_identical_poses_still_share_one_pocket():
    rec, ligand, coords = setup()
    poses = {0: coords, 1: coords + 0.05, 2: moved(coords, 3.0, 40, 5)}
    cfg = Config(receptor="r", ligand="l", share_pockets=False)
    infos = assign_pockets(rec, cfg, list(poses), poses, None, {}, ligand.GetNumAtoms())
    assert infos[0].key == infos[1].key and infos[0].key != infos[2].key


def test_an_already_computed_pocket_is_reused_when_it_is_cheaper():
    rec, ligand, coords = setup()
    cfg = Config(receptor="r", ligand="l")
    first = assign_pockets(rec, cfg, [0], {0: coords}, None, {}, ligand.GetNumAtoms())[0]
    again = assign_pockets(rec, cfg, [0], {0: coords + 0.02}, None, {first.key: first}, ligand.GetNumAtoms())[0]
    assert again.key == first.key


class InstantRunner:
    def __init__(self):
        self.keys = []

    def map(self, jobs, label):
        self.keys.extend(j["key"] for j in jobs)
        return {j["key"]: {"ok": True, "energy": -100.0, "attempt": 0} for j in jobs}


def test_pocket_energy_is_computed_once_per_distinct_pocket():
    rec, ligand, coords = setup()
    cfg = Config(receptor="r", ligand="l")
    runner = InstantRunner()
    session = Session(cfg, rec, runner)
    lig = SimpleNamespace(numbers=np.array([a.GetAtomicNum() for a in ligand.GetAtoms()]), charge=0,
                          n_atoms=ligand.GetNumAtoms(), e_ref=-50.0)
    poses = {0: coords, 1: coords + 0.05, 2: moved(coords, 3.0, 40, 5)}
    rows = session.evaluate(lig, [0, 1, 2], poses, {0: -50.0, 1: -50.0, 2: -50.0}, None)
    kinds = [k[0] for k in runner.keys]
    assert kinds.count("pocket") == len({r["pocket_id"] for r in rows.values()}) == 2
    assert kinds.count("complex") == 3
    assert rows[0]["pocket_id"] == rows[1]["pocket_id"] and rows[0]["pocket_poses"] == 2
    assert all(r["status"] == "ok" and r["truncated_units"] == 0 for r in rows.values())
    again = session.evaluate(lig, [0, 1], poses, {0: -50.0, 1: -50.0}, None)
    assert [k[0] for k in runner.keys].count("pocket") == 2 and again[0]["status"] == "ok"
