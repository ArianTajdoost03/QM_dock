import math

from rdkit import Chem

TEMPERATURE = 298.15
GAS_CONSTANT = 0.0019872041
KT_LN10 = GAS_CONSTANT * TEMPERATURE * math.log(10.0)

SITE_TABLE = (
    ("phosphate", "[O-][PX4]=O", True, (2.0, 7.2, 12.0)),
    ("sulfonate", "[O-][SX4](=O)=O", True, (-1.5,)),
    ("sulfonamide", "[N-][SX4](=O)=O", True, (10.1,)),
    ("carboxylate", "[O-][CX3]=O", True, (4.5,)),
    ("hydroxamate", "[O-][NX3;+0][CX3]=O", True, (9.0,)),
    ("tetrazolate", "[n-;$([n-]1nnnc1),$([n-]1nncn1)]", True, (4.9,)),
    ("aryl_thiolate", "[S-;X1]a", True, (6.6,)),
    ("thiolate", "[S-;X1]", True, (10.0,)),
    ("phenolate", "[O-]a", True, (10.0,)),
    ("vinylogous_acid", "[O-][#6X3]~[#6X3]=[OX1]", True, (7.0,)),
    ("guanidinium", "[NX3+;H1,H2]=[CX3]([NX3])[NX3]", False, (12.5,)),
    ("amidinium", "[NX3+;H1,H2]=[CX3]([NX3])[#6]", False, (11.5,)),
    ("imidazolium", "[nX3+;H1]1:[cX3]:[nX3;H1]:[cX3]:[cX3]:1", False, (6.9,)),
    ("pyridinium", "[nX3+;H1;r6]", False, (5.2,)),
    ("anilinium", "[NX4+;H1,H2,H3]a", False, (4.6,)),
    ("ammonium_primary", "[NX4+;H3]", False, (10.6,)),
    ("ammonium_secondary", "[NX4+;H2]", False, (10.8,)),
    ("ammonium_tertiary", "[NX4+;H1]", False, (9.8,)),
)

PERMANENT = (
    "[N+](=O)[O-]",
    "[O-][N+,n+]",
    "[NX4+;H0]",
    "[n+;H0]",
    "[S+,s+,P+,p+]",
    "[N+;H0]=[N-]",
    "[N-]=[N+]=N",
    "[Na+,K+,Li+,Cl-,Br-,I-,F-]",
)

COMPILED = tuple((name, Chem.MolFromSmarts(smarts), acid, pkas) for name, smarts, acid, pkas in SITE_TABLE)
PERMANENT_PATTERNS = tuple(Chem.MolFromSmarts(s) for s in PERMANENT)
SITE_NAMES = tuple(name for name, _, _, _ in SITE_TABLE)


def parse_overrides(items):
    out = {}
    for item in items or ():
        name, _, value = str(item).partition("=")
        name = name.strip()
        if name not in SITE_NAMES:
            raise ValueError(f"unknown pKa site '{name}'; choose from: {', '.join(SITE_NAMES)}")
        try:
            out[name] = float(value)
        except ValueError:
            raise ValueError(f"pKa override '{item}' must look like site=value") from None
    return out


def ionized_sites(mol, overrides=None):
    overrides = overrides or {}
    flat = Chem.RemoveHs(mol)
    claimed = {}
    found = []
    for name, patt, acid, pkas in COMPILED:
        for match in flat.GetSubstructMatches(patt):
            atom = match[0]
            if atom in claimed:
                continue
            claimed[atom] = name
            center = match[1] if name == "phosphate" else atom
            found.append({"name": name, "atom": atom, "center": center, "acid": acid,
                          "pkas": (overrides[name],) + tuple(pkas[1:]) if name in overrides else pkas})
    order = {}
    for site in sorted(found, key=lambda s: s["atom"]):
        k = order.get((site["name"], site["center"]), 0)
        order[(site["name"], site["center"])] = k + 1
        site["pka"] = site["pkas"][min(k, len(site["pkas"]) - 1)]
    permanent = set()
    for patt in PERMANENT_PATTERNS:
        for match in flat.GetSubstructMatches(patt):
            permanent.update(match)
    unclassified = [a.GetIdx() for a in flat.GetAtoms()
                    if a.GetFormalCharge() != 0 and a.GetIdx() not in claimed and a.GetIdx() not in permanent]
    return found, unclassified


def site_free_energy(site, ph):
    if site["acid"]:
        return KT_LN10 * (site["pka"] - ph)
    return KT_LN10 * (ph - site["pka"])


def state_penalties(mols, ph, overrides=None):
    rows = []
    for mol in mols:
        sites, unclassified = ionized_sites(mol, overrides)
        free = sum(site_free_energy(s, ph) for s in sites)
        rows.append({"free_energy": free, "sites": sites, "unclassified": unclassified})
    lowest = min(r["free_energy"] for r in rows)
    for r in rows:
        r["penalty"] = r["free_energy"] - lowest
        r["sites_text"] = ";".join(f"{s['name']}({s['pka']:g})" for s in r["sites"])
    return rows


def penalties_from_smiles(smiles, ph, overrides=None):
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"cannot parse protonation state SMILES: {smi}")
        mols.append(mol)
    return state_penalties(mols, ph, overrides)
