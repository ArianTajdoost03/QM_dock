import pandas as pd
from rdkit import Chem

from qmdock.chem.protonation import embed
from qmdock.tools.validate import recovery


def test_recovery_reports_rank_of_the_first_native_like_pose(tmp_path):
    ref = embed(Chem.MolFromSmiles("OCCN"), 3)
    ref_path = str(tmp_path / "ref.sdf")
    writer = Chem.SDWriter(ref_path)
    writer.write(ref)
    writer.close()
    far = Chem.Mol(ref)
    conf = far.GetConformer()
    for i in range(far.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (x + 6.0, y, z))
    poses_path = str(tmp_path / "poses.sdf")
    writer = Chem.SDWriter(poses_path)
    for name, mol in (("pose_0", far), ("pose_1", ref)):
        mol = Chem.Mol(mol)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    results = pd.DataFrame({"pose": [0, 1, 2], "status": ["ok", "ok", "scf_failed"], "final_rank": [1, 2, 3],
                            "e_final": [-10.0, -8.0, None]})
    path = str(tmp_path / "results.csv")
    results.to_csv(path, index=False)
    out = recovery(path, poses_path, ref_path, 2.0)
    assert out["top1_hit"] is False and out["first_hit_rank"] == 2 and out["best_rmsd_any"] < 0.01
    assert out["top1_rmsd"] > 5.0
