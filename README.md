# qmdock

Dock ligands into a metalloprotein pocket or onto a molecular crystal surface, and rank the poses with quantum-chemistry energies (GFN2-xTB). A fast GPU search proposes poses; only a small, diverse shortlist is scored with xTB on a cropped pocket.

## Install

```
conda env create -f environment.yml -n qmdock
conda activate qmdock
pytest tests          # optional check
```

A CUDA GPU speeds up the pose search but is not required. `tblite` must come from the PyPI wheel (the environment file does this); mixing it into an environment with other QM packages can crash.

## Quick start

You need a receptor PDB **with explicit hydrogens** (waters are removed, metals are kept), a ligand SDF, and the pocket centre `X Y Z`.

Dock one ligand:

```
python dock.py --receptor receptor.pdb --ligand ligand.sdf --center X Y Z --search-radius 6 --out run1 --workers 12
```

Screen an SDF library against the same pocket:

```
python screen.py --receptor receptor.pdb --ligands ligands.sdf --center X Y Z --search-radius 6 --solvent water --out screen1 --workers 12
```


## Output

| Command | Files |
|---|---|
| `dock.py` | `results.csv` (ranked poses), `poses.sdf`, `best_complex.xyz`, `summary.json` |
| `screen.py` | `screening_results.csv` (ranked ligands), `best_poses.sdf`, one folder per ligand with its log |

Screening keeps the lowest-energy protonation state of each ligand and never stops on a bad ligand: the reason is written to the table.

## How it works

1. Ligand conformers are generated and relaxed with GFN2.
2. A GPU search places them in the pocket: random starts, then gradient polishing (optionally with flexible torsions and mutation rounds).
3. A diverse shortlist of poses is chosen (optionally grouped by RMSD clusters).
4. GFN2 scores each pose on a pocket cut from the receptor, then relaxes the ligand and re-scores the best poses in a larger pocket.

```
E_bind = E(complex) - E(pocket) - E(lowest-energy free ligand conformer)     (kcal/mol, lower is better)
       = interaction energy + ligand strain
```

## Main options

| Option | Purpose |
|---|---|
| `--center`, `--search-radius` | Pocket position and size |
| `--solvent water` | Implicit solvent in all QM calculations |
| `--protonation pka\|all`, `--joint-states` | Dock protonation states; search poses once and share them |
| `--selection cluster` | Group poses by RMSD before QM |
| `--torsions` | Flexible rotatable bonds during the search |
| `--n-qm`, `--max-cluster-atoms`, `--final-top` | Trade speed against accuracy |
| `--workers`, `--device` | CPU cores for QM, `cuda` or `cpu` for the search |

Run `python dock.py --help` for everything else.

## Layout and details

```
dock.py  screen.py     entry points
qmdock/                chem/ (receptor, ligand)  search/ (GPU poses)  qm/ (tblite)  pipeline/ (stages)  screening/
tests/
```
