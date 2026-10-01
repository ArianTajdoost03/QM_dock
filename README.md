# qmdock

QM-ranked docking for crystal surfaces and metalloprotein pockets. Poses are generated with a fast GPU steric score, then ranked with GFN2-xTB (tblite) on whole-molecule pocket clusters.

## Install

```
conda env create -f environment.yml
conda activate qmdock
pytest tests
```

## Run

```
python dock.py --receptor beta_haematin_27cell.pdb --ligand ligand.sdf --center X Y Z --out run1
```

`--center` can be repeated for several sites. `--auto-sites N` picks N surface patches. With neither, the centroid of the ligand in the SDF is used.

Useful options: `--workers` (CPU cores for GFN2), `--device cuda|cpu`, `--search-radius`, `--n-random`, `--n-qm`, `--max-cluster-atoms`, `--final-max-atoms`, `--relax-top`, `--relax-steps`, `--prerelax-steps`, `--ligand-charge`.

## Proteins and metalloproteins

Waters are removed by default (`--keep-waters` keeps them); metal ions are always kept. A covalently connected unit larger than 400 atoms is treated as a polymer. For a polymer the QM pocket is built from whole residues nearest the ligand, up to `--max-cluster-atoms`:

- Metal ions within the shell are always included together with every residue that has an N, O or S within 2.8 A of the metal.
- Broken peptide or disulfide bonds are capped with hydrogen link atoms placed along the cut bond.
- The cluster charge is computed from the hydrogens present in the PDB (ASP/GLU/CYS/TYR deprotonated, LYS/ARG/HIS protonated, termini, metal ions +2 by default). Use a protonated receptor (explicit hydrogens).
- A cluster with an odd electron count is rejected with status `odd_electron`.

The fast search seeds a share of the random placements in free space inside the search sphere, so buried or conical pockets are sampled. If no pose survives the clash filter the thresholds are relaxed in three steps and a warning is written to `summary.json`.

## Pipeline

1. Receptor is split into covalently connected units (Fe-O links keep the hematin dimers whole). Units with an odd electron count or fewer than 10 atoms are kept for clash checks but never enter QM.
2. Ligand gets explicit hydrogens, ETKDG conformers, MMFF pre-optimisation, then a GFN2 relaxation. The lowest GFN2 energy is the strain reference.
3. Random rigid-body poses are scored on GPU with a Vina-like steric term, refined by batched gradient descent, clash-filtered and clustered by RMSD.
4. Stage 1: GFN2 on the ligand plus the nearest whole units (default cap 300 atoms).
5. Stage 2: short GFN2 relaxation of the ligand in the frozen pocket for the top poses.
6. Stage 3: the top poses are rescored with a larger cluster (default cap 600 atoms). The final ranking uses the highest level reached.

## Energies

- `e_int` = E(complex) - E(pocket) - E(ligand at pose geometry)
- `strain` = E(ligand at pose geometry) - E(best free conformer)
- `e_bind` = `e_int` + `strain`
- `e_final` is the value used for ranking, `level` says which stage produced it. Compare `e_final` only between poses of the same level.


## Outputs

`results.csv`, `poses.sdf` (ranked, with energies as properties), `best_complex.xyz`, `summary.json` (timings, failures, energy gap to the next pose family).

## Speed

GFN2 cost grows roughly with the cube of the number of atoms. Measured on one core: a heme dimer plus this ligand (211 atoms) takes about 30 s; two dimers plus the ligand (about 359 atoms) takes several minutes. Cluster size is therefore the main cost.

Defaults are chosen for a first pass:

- Stage 1 (`--n-qm 30`, `--max-cluster-atoms 150`): the ligand plus the single nearest dimer.
- Relaxation (`--relax-top 3`, `--relax-steps 5`): each step costs about three single points; `--relax-top 0` skips it.
- Stage 3 (`--final-top 3`, `--final-max-atoms 300`): up to two dimers plus the ligand. This is the slowest stage; raise it only for the final shortlist.
- Use `--workers` equal to the number of physical cores. The fast stage uses the GPU.

The fast stage costs about `n_random + 3 * n_optimize * opt_steps` pose evaluations; on CPU use `--n-random 20000 --n-optimize 300 --opt-steps 60`.