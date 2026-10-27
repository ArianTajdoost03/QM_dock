from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from qmdock.chem.pka import state_penalties

MAX_FREE_SITES = 10
DEFAULT_WINDOW = 5.0


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


def enumerate_all(neutral, ph, overrides):
    scored = []
    for idx, acid in find_sites(neutral):
        toggled = toggle(neutral, idx, acid)
        if toggled is None:
            continue
        free_energy = state_penalties([toggled], ph, overrides)[0]["free_energy"]
        scored.append((abs(free_energy), idx, acid, free_energy))
    scored.sort()
    free, fixed = scored[:MAX_FREE_SITES], scored[MAX_FREE_SITES:]
    base = neutral
    for _, idx, acid, free_energy in fixed:
        if free_energy < 0:
            toggled = toggle(base, idx, acid)
            if toggled is not None:
                base = toggled
    out = []
    for mask in range(1 << len(free)):
        m = base
        for bit, (_, idx, acid, _) in enumerate(free):
            if mask >> bit & 1 and m is not None:
                m = toggle(m, idx, acid)
        if m is not None:
            out.append(m)
    return out


def ligand_states(base, mode, ph_min, ph_max, max_states, seed, ph=None, window=DEFAULT_WINDOW, overrides=None,
                  log=None):
    given = state_record(base, "as given")
    if mode == "none":
        return [given]
    ph = (ph_min + ph_max) / 2 if ph is None else ph
    flat = Chem.RemoveHs(base)
    if mode == "pka":
        from dimorphite_dl import protonate_smiles

        mols = []
        for smi in protonate_smiles(Chem.MolToSmiles(flat), ph_min=ph_min, ph_max=ph_max,
                                    max_variants=max(16, 4 * max_states)):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                mols.append(m)
        label = f"Dimorphite-DL pH {ph_min}-{ph_max}"
    elif mode == "all":
        mols = [flat] + enumerate_all(rdMolStandardize.Uncharger().uncharge(flat), ph, overrides)
        label = "site enumeration"
    else:
        raise ValueError(f"unknown protonation mode: {mode}")
    unique = {}
    for m in mols:
        unique.setdefault(Chem.MolToSmiles(m), m)
    if not unique:
        return [given]
    items = list(unique.items())
    penalties = [r["penalty"] for r in state_penalties([m for _, m in items], ph, overrides)]
    order = sorted(range(len(items)), key=lambda i: (round(penalties[i], 6), abs(Chem.GetFormalCharge(items[i][1])),
                                                     items[i][0]))
    states, dropped = [], []
    for i in order:
        smi, m = items[i]
        if states and (len(states) >= max_states or (mode == "all" and window > 0 and penalties[i] > window)):
            dropped.append((smi, penalties[i]))
            continue
        if smi == given["smiles"]:
            record = dict(given)
        else:
            mh = embed(m, seed)
            if mh is None:
                dropped.append((smi, penalties[i]))
                continue
            record = state_record(mh, label)
        record["penalty"] = penalties[i]
        states.append(record)
    if log is not None:
        log(f"protonation states at pH {ph:g}: kept {len(states)} of {len(items)} candidates "
            f"(ranked by pKa penalty, window {window:g} kcal/mol, at most {max_states})")
        for smi, penalty in dropped[:6]:
            log(f"  dropped {smi} (penalty {penalty:.1f} kcal/mol)")
    return states or [given]
