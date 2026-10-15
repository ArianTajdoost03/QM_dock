import argparse
import csv
import glob
import os
import re
import urllib.request
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

WATER = {"HOH", "WAT", "DOD", "H2O"}
METALS = {"ZN", "MG", "MN", "CA", "FE", "FE2", "CO", "NI", "CU", "CU1", "CD"}
EXCLUDE = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "1PE", "MPD", "DMS",
    "ACT", "ACY", "FMT", "TRS", "EPE", "MES", "BME", "NO3", "CIT", "TAR", "FLC",
    "IMD", "BOG", "NAG", "NDG", "BMA", "MAN", "FUC", "GAL", "BGC", "GLC", "SIA",
    "CL", "BR", "IOD", "NA", "K", "NH4", "SCN", "AZI", "PEO", "DTT", "MLI",
    "MLA", "OXL", "ACE", "NH2", "BU3", "BTB", "TLA", "SO3", "PE4", "PE3", "PG0",
}
METAL_CONTACT = 6.0
METAL_RADIUS = 8.0
TEMPLATE_URLS = (
    "https://files.rcsb.org/ligands/download/{c}_ideal.sdf",
    "https://files.rcsb.org/ligands/view/{c}_ideal.sdf",
)


def parse(path):
    atoms = []
    modres = set()
    with open(path) as fh:
        for raw in fh:
            if raw.startswith("MODRES"):
                modres.add(raw[12:15].strip().upper())
                continue
            if raw.startswith("ENDMDL"):
                break
            rec = raw[:6]
            if rec not in ("ATOM  ", "HETATM") or len(raw.rstrip("\r\n")) < 54:
                continue
            if raw[16] not in (" ", "A"):
                continue
            name = raw[12:16].strip()
            element = raw[76:78].strip().upper() if len(raw.rstrip("\r\n")) >= 78 else ""
            if not element:
                element = re.sub(r"[^A-Za-z]", "", name)[:1].upper()
            line = raw[:16] + " " + raw[17:]
            atoms.append({
                "record": rec.strip(),
                "name": name,
                "resname": raw[17:20].strip().upper(),
                "chain": raw[21],
                "resseq": raw[22:26].strip(),
                "icode": raw[26].strip(),
                "xyz": (float(raw[30:38]), float(raw[38:46]), float(raw[46:54])),
                "element": element,
                "line": line.rstrip("\r\n") + "\n",
            })
    return atoms, modres


def classify(atoms, modres):
    protein, metals, others = [], [], []
    for a in atoms:
        if a["element"] in ("H", "D"):
            continue
        rn = a["resname"]
        if rn in WATER:
            continue
        if a["record"] == "ATOM":
            protein.append(a)
        elif rn in METALS:
            metals.append(a)
        elif rn in modres:
            protein.append(a)
        else:
            others.append(a)
    return protein, metals, others


def min_dist(points, ref, chunk=4000):
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        block = points[i:i + chunk]
        out[i:i + chunk] = np.linalg.norm(block[:, None, :] - ref[None, :, :], axis=2).min(axis=1)
    return out


def pick_ligand(others, metals, min_heavy):
    groups = {}
    for a in others:
        if a["resname"] in EXCLUDE:
            continue
        key = (a["resname"], a["chain"], a["resseq"], a["icode"])
        groups.setdefault(key, []).append(a)
    metal_xyz = np.array([m["xyz"] for m in metals]) if metals else None
    best = None
    best_score = None
    for key, atoms in groups.items():
        if len(atoms) < min_heavy:
            continue
        if metal_xyz is not None:
            xyz = np.array([a["xyz"] for a in atoms])
            dmin = float(min_dist(xyz, metal_xyz).min())
        else:
            dmin = float("inf")
        score = (dmin <= METAL_CONTACT, len(atoms), -dmin)
        if best_score is None or score > best_score:
            best = (key, atoms)
            best_score = score
    return best


def select_chains(protein, lig_xyz, contact):
    pts = np.array([a["xyz"] for a in protein])
    d = min_dist(pts, lig_xyz)
    chains = {a["chain"] for a, dist in zip(protein, d) if dist <= contact}
    if not chains:
        chains = {protein[int(d.argmin())]["chain"]}
    return chains


def write_protein(lines, path, hydrogens, ph):
    tmp = path + ".tmp.pdb"
    with open(tmp, "w") as fh:
        fh.writelines(lines)
        fh.write("END\n")
    if not hydrogens:
        os.replace(tmp, path)
        return
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=tmp)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        with open(path, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_template(code, cache_dir):
    path = os.path.join(cache_dir, code + ".sdf")
    if not os.path.exists(path):
        data = None
        for url in TEMPLATE_URLS:
            try:
                with urllib.request.urlopen(url.format(c=code), timeout=30) as resp:
                    data = resp.read()
                if data and b"M  END" in data:
                    break
                data = None
            except Exception:
                data = None
        if data is None:
            return None
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    mol = suppl[0] if len(suppl) else None
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            return Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return None


def write_sdf(mol, path, kekulize=True):
    writer = Chem.SDWriter(path)
    writer.SetKekulize(kekulize)
    try:
        writer.write(mol)
        return True
    except Exception:
        return False
    finally:
        writer.close()


def write_ligand(atoms, resname, pdb_id, path, cache_dir):
    block = "".join(a["line"] for a in atoms) + "END\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None:
        raise ValueError("ligand could not be parsed")
    template = load_template(resname, cache_dir)
    status = "bond_orders_from_template"
    final = None
    if template is None:
        status = "single_bonds_only:template_unavailable"
    else:
        try:
            final = AllChem.AssignBondOrdersFromTemplate(template, mol)
        except Exception:
            status = "single_bonds_only:template_mismatch"
    if final is None:
        mol.UpdatePropertyCache(strict=False)
        final = mol
    name = "%s_%s" % (pdb_id, resname)
    final.SetProp("_Name", name)
    if not write_sdf(final, path) or Chem.MolFromMolFile(path, removeHs=False) is None:
        status = "single_bonds_only:unsanitizable"
        mol.UpdatePropertyCache(strict=False)
        mol.SetProp("_Name", name)
        write_sdf(mol, path, kekulize=False)
    return status


def process(path, out_dir, cache_dir, hydrogens, ph, contact, min_heavy):
    pdb_id = os.path.splitext(os.path.basename(path))[0].lower()
    row = {
        "id": pdb_id, "ligand": "", "chain": "", "resseq": "", "heavy_atoms": "",
        "chains_kept": "", "metals_kept": "", "ligand_bonds": "", "status": "",
    }
    try:
        atoms, modres = parse(path)
        protein, metals, others = classify(atoms, modres)
        if not protein:
            row["status"] = "no_protein"
            return row
        picked = pick_ligand(others, metals, min_heavy)
        if picked is None:
            row["status"] = "no_ligand"
            return row
        (resname, chain, resseq, icode), lig_atoms = picked
        lig_xyz = np.array([a["xyz"] for a in lig_atoms])
        chains = select_chains(protein, lig_xyz, contact)
        kept_metals = []
        for m in metals:
            near = min_dist(np.array([m["xyz"]]), lig_xyz)[0] <= METAL_RADIUS
            if m["chain"] in chains or near:
                kept_metals.append(m)
        lines = [a["line"] for a in protein if a["chain"] in chains]
        lines += [m["line"] for m in kept_metals]
        entry_dir = os.path.join(out_dir, pdb_id)
        os.makedirs(entry_dir, exist_ok=True)
        write_protein(lines, os.path.join(entry_dir, pdb_id + "_protein.pdb"), hydrogens, ph)
        row["ligand_bonds"] = write_ligand(
            lig_atoms, resname, pdb_id, os.path.join(entry_dir, pdb_id + "_ligand.sdf"), cache_dir
        )
        row.update({
            "ligand": resname,
            "chain": chain,
            "resseq": resseq + icode,
            "heavy_atoms": len(lig_atoms),
            "chains_kept": "".join(sorted(chains)),
            "metals_kept": ",".join("%s:%d" % kv for kv in sorted(Counter(m["resname"] for m in kept_metals).items())),
            "status": "ok",
        })
    except Exception as exc:
        row["status"] = "error: %s" % exc
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-hydrogens", action="store_true")
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--contact", type=float, default=6.0)
    parser.add_argument("--min-heavy", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cache_dir = os.path.join(args.output, "_ccd_templates")
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.pdb")))
    worker = partial(
        process,
        out_dir=args.output,
        cache_dir=cache_dir,
        hydrogens=not args.no_hydrogens,
        ph=args.ph,
        contact=args.contact,
        min_heavy=args.min_heavy,
    )
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            rows = list(pool.imap(worker, files))
    else:
        rows = [worker(f) for f in files]

    fields = ["id", "ligand", "chain", "resseq", "heavy_atoms", "chains_kept", "metals_kept", "ligand_bonds", "status"]
    with open(os.path.join(args.output, "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(r["status"] == "ok" for r in rows)
    print("processed %d files, ok: %d" % (len(rows), ok))
    for r in rows:
        if r["status"] != "ok" or r["ligand_bonds"] != "bond_orders_from_template":
            print(r["id"], r["status"], r["ligand_bonds"])


if __name__ == "__main__":
    main()
