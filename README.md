# ITG turbulence optimization of stellarator equilibria

This pipeline optimizes a VMEC equilibrium with [simsopt](https://github.com/hiddenSymmetries/simsopt). It targets an
aspect ratio, a mean rotational transform, an ITG turbulence proxy, and, optionally, quasisymmetry (QS).

One VMEC input file with one set of targets launches **10 optimizations** as a Slurm job array: turbulence
targets 1–5, each with and without QS. Each run then produces convergence plots, Boozer |B| plots (QS runs only),
a ParaView file of the turbulence targets, a 3D heatmap of the optimized target, and flux-surface cross sections.
A CSV of parameter sets can launch many of these sweeps at once.

## Repository layout

```
scripts/                 what you run / submit
  batch_run.bash         one input + targets → one 10-task job array
  batch_optim.slurm      the job array: optimization + post-processing for one task
  sweep_run.py           CSV of parameter sets → one batch_run.bash call per row
src/                     python run by the jobs
  run_optimization.py    builds and solves the least-squares problem
  grid_geom_features.py  turbulence proxy objective (targets 1–5)
  plot_total_obj.py      total objective vs. function evaluation
  plot_specific_qs.py    per-term convergence (4 panels) + Boozer |B| plot
  plot_specific_noqs.py  per-term convergence (3 panels)
  make_paraview.py       computes all targets on the boundary, writes a .vts for ParaView
  plot_surfaces.py       3D heatmap of the optimized target + flux-surface cross sections
  wout_info.py           small shared helpers (nfp, plot labels)
inputs/
  vmec/                  VMEC input files (input.*)
  sweeps/                parameter-sweep CSVs (param_sweep.csv)
legacy/                  older single-run scripts, kept for reference only
optimZips/               archived results of past runs
runs/                    (cluster only, gitignored) one optim-* folder per launched sweep
```

`src/` is flat on purpose. The scripts import each other as siblings (`plot_surfaces` → `make_paraview`,
`run_optimization` → `grid_geom_features`, plot scripts → `wout_info`). Python only searches the running script's own
folder for those imports, so they have to live in the same directory.

## Quick start

All commands below assume the repository is checked out at `/scratch/ms13725/codes` on the cluster. That path is set
as `CODE_DIR` in `scripts/batch_optim.slurm`. Change it there if the checkout lives somewhere else.

### One input file

```bash
cd /scratch/ms13725/codes
bash scripts/batch_run.bash inputs/vmec/input.nfp4_torus 6 1.24 1 -1
bash scripts/batch_run.bash -h          # full help
```

| Argument      | Meaning                                                              |
|---------------|----------------------------------------------------------------------|
| `vmec_input`  | VMEC input file, e.g. `inputs/vmec/input.nfp4_torus`                 |
| `aspect`      | target aspect ratio                                                  |
| `iota`        | target mean rotational transform                                     |
| `helicity_m`  | QS helicity m, used only by the 5 QS runs (usually 1)                |
| `helicity_n`  | QS helicity n, used only by the 5 QS runs (QA: `0`, QH: `-1`)        |

The results go to `runs/optim-<input name without "input.">-<aspect>-<iota>-<YYYYMMDD_HHMMSS>/`, e.g.
`runs/optim-nfp4_torus-6-1.24-20261017_143210/`. It works from any working directory.

### A parameter sweep

```bash
python3 scripts/sweep_run.py inputs/sweeps/param_sweep.csv --dry-run   # print the commands only
python3 scripts/sweep_run.py inputs/sweeps/param_sweep.csv             # submit
```

The CSV has one header row (skipped), then one row per sweep point:

```
vmec_input,aspect,iota,m,n
inputs/vmec/input.nfp2_torus,5,0.42,1,0
inputs/vmec/input.nfp4_torus,6,1.24,1,-1
```

- Relative input paths are resolved against the repository root, so the CSV itself can live anywhere.
- Blank rows and rows starting with `#` are skipped.
- Every row is validated before anything is submitted: the input file must exist, the numbers must parse, and there
  must be 5 columns. One bad row means nothing gets submitted.
- Rows are submitted 1 s apart so their timestamped folder names stay distinct.
- Each row is a full 10-task array, so a 20-row CSV submits 200 Slurm tasks.
- `sweep_run.py` uses only the Python standard library, so it runs with the login node's `python3` outside the
  container.

> **No directory in a run path may contain `input.`** VMEC names its output files from everything after the first
> `input.` in the input file's *full path*. Such a directory gives a `cdfcrt ... nf_create` error and no wout file.
> `batch_run.bash` strips the `input.` prefix from folder names and refuses to run if the path would still contain it.

## How a run works

### 1. Submission (`batch_run.bash`)

`batch_run.bash` checks its arguments and creates the timestamped folder under `runs/`. From inside that folder it
submits `batch_optim.slurm` as a 10-task array, passing the input's absolute path and the targets.

### 2. The job array (`batch_optim.slurm`)

Each array task picks its turbulence target and QS setting from its task ID and works in its own subfolder:

| Task | Target | QS  | Run folder          |
|------|--------|-----|---------------------|
| 0–4  | 1–5    | off | `t1_noqs/` … `t5_noqs/` |
| 5–9  | 1–5    | on  | `t1_qs/` … `t5_qs/`     |

Each task gets 16 MPI ranks, 20 GB and 2 h, and runs inside the singularity image with the simsopt overlay. The nfp
used in file names and plot titles is read from the input file's `NFP =` line. The task then runs the steps below in
order.

### 3. Optimization (`run_optimization.py`)

```
mpiexec python src/run_optimization.py <input> <aspect> <iota> --target N [--helicity-m M --helicity-n N]
```

This builds a simsopt `LeastSquaresProblem` with these terms, in this order and all with weight 1:

1. aspect ratio → `aspect`
2. mean iota → `iota`
3. *(QS runs only)* `QuasisymmetryRatioResidual` on s = 0.2, 0.4, 0.6, 0.8 with helicity (m, n) → 0
4. turbulence proxy → 0. This term is **always last**, because the plot scripts read the last 4 residual columns as
   the turbulence proxy.

The boundary is optimized in three continuation stages, freeing modes with |m|, |n| ≤ 2, then 3, then 4. VMEC
resolution grows with them (mpol = ntor = 2, 3, 4), and `rc(0,0)` is held fixed throughout. Each stage calls
`least_squares_mpi_solve` with finite-difference gradients across the MPI ranks. After the last stage VMEC runs once
more at the optimum, and its output is copied to `wout_final.nc`, which all the post-processing reads.

### 4. Turbulence proxy (`grid_geom_features.py`)

`MultiSurfaceTurbulenceObjective` evaluates the geometry with `vmec_compute_geometry` on 4 flux surfaces
(s = 0.2, 0.4, 0.6, 0.8), each on a 16 × 16 (θ, φ) grid over one field period. It returns **one residual per
surface**. Every target combines |B|, the normalized |∇s|² term f5, and a weight w = sigmoid(|B|²·f4), where f4 is
the normalized curvature-drift term. The weight emphasizes regions of bad curvature.

| Target | Per-surface quantity                 |
|--------|--------------------------------------|
| 1      | variance of w · f5 / \|B\|           |
| 2      | mean of w · f5⁴ / \|B\|²             |
| 3      | mean of w · f5² / \|B\|              |
| 4      | variance of w · f5² / \|B\|          |
| 5      | mean of w · f5² / \|B\|²             |

The exact feature definitions and normalizations are in `src/grid_geom_features.py`.

### 5. Post-processing

All of these run in the task's run folder after the optimization:

| Step | Script                                     | Writes |
|------|--------------------------------------------|--------|
| a    | `plot_total_obj.py N`                      | `total_obj.pdf`: total objective vs. function evaluation, stages concatenated |
| b    | `plot_specific_noqs.py aspect iota N`, or `plot_specific_qs.py aspect iota N m n` | `targets.pdf`: aspect, iota, turbulence (and QS) convergence. QS runs also write `boozer.pdf` |
| c    | `make_paraview.py`                         | `objectives_paraview.vts` |
| d    | `plot_surfaces.py N`                       | `heatmap.pdf`, `plot3.pdf` |

Finally, each file is moved into `plots/` and renamed `<stem>_nfp<NFP>_<N><qs|noqs>.<ext>`
(see [Outputs](#outputs)).

## Outputs

```
runs/optim-<name>-<aspect>-<iota>-<timestamp>/
  optim_<jobid>_<task>.out / .err     slurm logs, one pair per array task
  t1_noqs/ … t5_noqs/  t1_qs/ … t5_qs/
    objective_*.dat, residuals_*.dat  simsopt history, one file per continuation stage
    wout_*.nc                         VMEC outputs from the optimization
    wout_final.nc                     equilibrium at the optimum
    plots/
```

Files in `plots/`, shown with the example suffix `_nfp4_3qs` (nfp 4, target 3, QS on):

| File                         | Contents |
|------------------------------|----------|
| `total_obj_nfp4_3qs.pdf`     | total objective vs. function evaluation |
| `targets_nfp4_3qs.pdf`       | convergence of each term: aspect, iota, turbulence proxy, + QS on QS runs |
| `boozer_nfp4_3qs.pdf`        | *(QS runs only)* Boozer \|B\| on s ≈ 0.2/0.4/0.6/0.8. Needs `booz_xform` and is skipped without it |
| `paraview_nfp4_3qs.vts`      | boundary surface with all targets as arrays `t1_outer … t5_outer` (value on s = 1) and `t1_avg … t5_avg` (radial average) |
| `heatmap_nfp4_3qs.pdf`       | 3D boundary colored by the run's own target: on the boundary (left) and radially averaged (right). For targets 1 and 4 the color is the squared deviation from the surface mean |
| `flux_surfaces_nfp4_3qs.pdf` | cross sections at φ = 0, ¼, ½, ¾ field period: boundary overlay + 10 nested surfaces with the magnetic axis |

A line `skipping <file> (not found)` in a task's `.out` means that plot wasn't produced. The traceback is in the
matching `.err`.

## Known caveats

- **The plotted radial average isn't exactly the optimized quantity.** `make_paraview.py` and the heatmap average
  over 10 surfaces s = 0.1–1.0 on the full torus. The optimizer uses 4 surfaces s = 0.2–0.8 on one field period. For
  targets 1 and 4 the plots show the pointwise squared deviation, not the variance itself.
- **Unused features.** `grid_geom_features.py` computes f2, f3, f6 and f7, but no target uses them.
- **`legacy/`** scripts reference the old flat layout. Their paths are stale and they aren't maintained.

## Local development

The full pipeline only runs on the cluster, but most of it can be checked on a laptop without simsopt or numpy:

```bash
bash -n scripts/batch_run.bash scripts/batch_optim.slurm     # shell syntax
python3 -m py_compile src/*.py scripts/sweep_run.py           # python syntax
python3 scripts/sweep_run.py inputs/sweeps/param_sweep.csv --dry-run
```

To exercise `batch_run.bash` / `sweep_run.py` end to end without submitting anything, put a fake `sbatch` first on
`PATH`. Do this in a scratch copy of the repository, since the run folders are created under its `runs/`:

```bash
mkdir -p /tmp/fakebin && printf '#!/bin/bash\necho "sbatch $*"\n' > /tmp/fakebin/sbatch && chmod +x /tmp/fakebin/sbatch
PATH=/tmp/fakebin:$PATH bash scripts/batch_run.bash inputs/vmec/input.nfp4_torus 6 1.24 1 -1
```

For a quick real test on the cluster, run just one no-QS task and one QS task with a short time limit, from a new
run folder:

```bash
D=/scratch/ms13725/codes/runs/optim-test-$(date +%Y%m%d_%H%M%S) && mkdir -p "$D" && cd "$D"
sbatch --array=0,5 --time=00:45:00 /scratch/ms13725/codes/scripts/batch_optim.slurm \
    /scratch/ms13725/codes/inputs/vmec/input.nfp4_torus 6 1.24 1 -1
```

## Dependencies

These are provided on the cluster by the singularity image and simsopt overlay:

- simsopt (with VMEC) and mpi4py
- numpy, scipy, pandas
- matplotlib ≥ 3.6 (needed for the zoomed 3D heatmap)
- booz_xform (optional, only for `boozer.pdf`)
