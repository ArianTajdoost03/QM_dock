import argparse
import numpy as np
from rdkit import Chem


def load_mols(path):
    with open(path, "r", errors="replace") as fh:
        text = fh.read().replace("\r\n", "\n")
    text = "\n".join(l for l in text.split("\n") if not l.startswith("M  RAD"))
    suppl = Chem.SDMolSupplier()
    suppl.SetData(text, removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules read from {path}")
    return mols


def skeleton(mol):
    conf = mol.GetConformer()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    new_idx = {old: i for i, old in enumerate(keep)}
    rw = Chem.RWMol()
    for old in keep:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in new_idx and j in new_idx:
            rw.AddBond(new_idx[i], new_idx[j], Chem.BondType.SINGLE)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in keep])
    return rw.GetMol(), coords


def kabsch_rmsd(P, Q):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return np.sqrt(((P @ R - Q) ** 2).sum() / len(P))


def best_rmsd(ref, probe, align=False):
    (rmol, rxyz), (pmol, pxyz) = skeleton(ref), skeleton(probe)
    if rmol.GetNumAtoms() != pmol.GetNumAtoms():
        raise ValueError(f"heavy-atom counts differ ({rmol.GetNumAtoms()} vs "
                         f"{pmol.GetNumAtoms()})")
    matches = pmol.GetSubstructMatches(rmol, uniquify=False,
                                       useChirality=False, maxMatches=100000)
    if not matches:
        raise ValueError("heavy-atom connectivity differs between the molecules")
    best = None
    for m in matches:                       # m[i] = probe atom matching ref atom i
        q = pxyz[list(m)]
        val = (kabsch_rmsd(q, rxyz) if align
               else np.sqrt(((q - rxyz) ** 2).sum() / len(rxyz)))
        if best is None or val < best:
            best = val
    return best, len(matches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ligand", help="reference ligand SDF")
    ap.add_argument("poses", help="SDF with pose(s)")
    ap.add_argument("--all", action="store_true", help="report every pose")
    ap.add_argument("--align", action="store_true",
                    help="also report RMSD after best-fit alignment")
    args = ap.parse_args()

    ref = load_mols(args.ligand)[0]
    poses = load_mols(args.poses)
    n = len(poses) if args.all else 1
    print(f"Reference heavy atoms: {skeleton(ref)[0].GetNumAtoms()} | poses in file: {len(poses)}")
    for i in range(n):
        name = poses[i].GetProp("_Name") if poses[i].HasProp("_Name") else f"pose {i+1}"
        try:
            r, nm = best_rmsd(ref, poses[i])
            line = f"Pose {i+1} ({name}): RMSD = {r:.3f} Å  [{nm} mappings tested]"
            if args.align:
                line += f" | aligned RMSD = {best_rmsd(ref, poses[i], True)[0]:.3f} Å"
            print(line)
        except Exception as e:
            print(f"Pose {i+1} ({name}): failed - {e}")


if __name__ == "__main__":
    main()