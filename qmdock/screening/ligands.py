import os
import re
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from qmdock.chem.elements import ATOMIC_NUMBER
from qmdock.chem.protonation import embed

NAME_KEYS = ("_Name", "Name", "NAME", "ID", "id", "Title", "title")


@dataclass
class LigandRecord:
    index: int
    name: str
    mol: Chem.Mol = None
    status: str = "ready"
    error: str = ""
    notes: list = field(default_factory=list)
    input_smiles: str = ""
    heavy_atoms: int = 0


def safe_name(text, fallback):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")[:60]
    return cleaned or fallback


def raw_name(mol, fallback):
    for key in NAME_KEYS:
        if mol.HasProp(key) and mol.GetProp(key).strip():
            return safe_name(mol.GetProp(key), fallback)
    return fallback


def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return mol, None
    best = max(frags, key=lambda f: sum(1 for a in f.GetAtoms() if a.GetAtomicNum() > 1))
    return best, f"kept the largest of {len(frags)} fragments"


def prepare_molecule(mol, seed):
    mol, note = largest_fragment(mol)
    notes = [note] if note else []
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    unsupported = sorted(elements - set(ATOMIC_NUMBER))
    if unsupported:
        raise ValueError(f"unsupported element(s): {', '.join(unsupported)}")
    flat = Chem.RemoveHs(mol)
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        full = Chem.AddHs(mol, addCoords=True)
    else:
        full = embed(flat, seed)
        notes.append("no 3D coordinates in the file; embedded a conformer")
        if full is None:
            raise ValueError("3D embedding failed")
    electrons = sum(a.GetAtomicNum() for a in full.GetAtoms()) - Chem.GetFormalCharge(full)
    if electrons % 2:
        raise ValueError("odd electron count (radical or wrong protonation)")
    return full, notes


def read_ligands(path, max_heavy=60, seed=7):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ligand file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"ligand file is empty: {path}")
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    if len(supplier) == 0:
        raise ValueError(f"no molecules found in {path}; is it an SDF file?")
    records, used = [], set()
    for i in range(len(supplier)):
        fallback = f"lig{i + 1:05d}"
        try:
            mol = supplier[i]
        except Exception as exc:
            mol, parse_error = None, str(exc)[:120]
        else:
            parse_error = "RDKit could not parse this record"
        name = raw_name(mol, fallback) if mol is not None else fallback
        if name in used:
            name = f"{name}_{i + 1}"
        used.add(name)
        rec = LigandRecord(i, name)
        if mol is None:
            rec.status, rec.error = "invalid", parse_error
            records.append(rec)
            continue
        try:
            prepared, rec.notes = prepare_molecule(mol, seed)
        except (ValueError, RuntimeError) as exc:
            rec.status, rec.error = "invalid", str(exc)
            records.append(rec)
            continue
        rec.mol = prepared
        rec.heavy_atoms = sum(1 for a in prepared.GetAtoms() if a.GetAtomicNum() > 1)
        rec.input_smiles = Chem.MolToSmiles(Chem.RemoveHs(prepared))
        if rec.heavy_atoms > max_heavy:
            rec.status, rec.error = "skipped", f"{rec.heavy_atoms} heavy atoms exceeds --max-heavy-atoms {max_heavy}"
        records.append(rec)
    return records
