from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


ACID_SITES = (
    ("[CX3](=O)[OX2H1]", 2),
    ("[PX4](=O)[OX2H1]", 2),
    ("[SX4](=O)(=O)[OX2H1]", 3),
    ("[SX4](=O)(=O)[NX3;H1,H2]", 3),
    ("[OX2H1]c", 0),
    ("[OX2H1][#6X3]~[#6X3]=[OX1]", 0),
    ("[OX2H1][NX3][CX3]=O", 0),
    ("[SX2H1]", 0),
    ("[nH]", 0),
)


BASE_SITES = (
    ("[NX3;+0;!$(N-[!#6;!#1]);!$(N-C=[O,S,N]);!$(N-a);!$(N-C=C)]", 0),
    ("[NX3;H2,H1;!$(NC=O)][CX3]=[NX2;+0;!$(N-C=O)]", 2),
    ("[nX2]1:[cX3]:[nX3]:[cX3]:[cX3]:1", 0),
)


def find_sites(mol):
    sites = {}
    for acid, table in ((True, ACID_SITES), (False, BASE_SITES)):
        for smarts, pos in table:
            patt = Chem.MolFromSmarts(smarts)
            for match in mol.GetSubstructMatches(patt):
                idx = match[pos]
                atom = mol.GetAtomWithIdx(idx)
                if acid and atom.GetSymbol() == "N" and atom.GetIsAromatic():
                    ring = next((r for r in mol.GetRingInfo().AtomRings() if idx in r), ())
                    if sum(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring) < 4:
                        continue
                sites.setdefault(idx, acid)
    return sorted(sites.items())


def toggle(mol, idx, acid):
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    h = atom.GetTotalNumHs()
    if acid and h == 0:
        return None
    atom.SetNumExplicitHs(h - 1 if acid else h + 1)
    atom.SetFormalCharge(atom.GetFormalCharge() + (-1 if acid else 1))
    atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


def embed(mol, seed):
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=seed) < 0:
        if AllChem.EmbedMolecule(mh, randomSeed=seed, useRandomCoords=True) < 0:
            return None
    if AllChem.MMFFHasAllMoleculeParams(mh):
        AllChem.MMFFOptimizeMolecule(mh, maxIters=2000)
    return mh


def state_record(mol, label):
    return {"mol": mol, "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)), "charge": Chem.GetFormalCharge(mol),
            "label": label}


def ligand_states(base, mode, ph_min, ph_max, max_states, seed):
    given = state_record(base, "as given")
    if mode == "none":
        return [given]
    flat = Chem.RemoveHs(base)
    found = {given["smiles"]: given}
    if mode == "pka":
        from dimorphite_dl import protonate_smiles

        candidates = []
        for smi in protonate_smiles(Chem.MolToSmiles(flat), ph_min=ph_min, ph_max=ph_max,
                                    max_variants=max_states):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                candidates.append(m)
        label = f"Dimorphite-DL pH {ph_min}-{ph_max}"
        found = {}
    elif mode == "all":
        neutral = rdMolStandardize.Uncharger().uncharge(flat)
        sites = find_sites(neutral)[:5]
        candidates = []
        for mask in range(1 << len(sites)):
            m = neutral
            for bit, (idx, acid) in enumerate(sites):
                if mask >> bit & 1 and m is not None:
                    m = toggle(m, idx, acid)
            if m is not None:
                candidates.append(m)
        label = "site enumeration"
    else:
        raise ValueError(f"unknown protonation mode: {mode}")
    for m in candidates:
        smi = Chem.MolToSmiles(m)
        if smi in found:
            continue
        if smi == given["smiles"]:
            found[smi] = given
            continue
        mh = embed(m, seed)
        if mh is not None:
            found[smi] = state_record(mh, label)
    states = sorted(found.values(), key=lambda s: (abs(s["charge"]), s["charge"]))[:max_states]
    return states or [given]
