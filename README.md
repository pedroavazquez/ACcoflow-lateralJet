# AC Co-flow Lateral Jet (Nondimensional, Multigrid)

This folder contains the Basilisk C simulation for an axisymmetric immersed co-flow jet with optional AC electric forcing, plus Python and C utilities for postprocessing generated data.

## Contents

### Core simulation (C/Basilisk)

- `lateralJet_params.c`: main simulation code with runtime parameters in `params.txt` and support for fresh/restore runs.
- `lateralJet_params/params.txt`: example runtime parameter file used by restart and sweep workflows.
- `drop_stat.h`, `poisson_complex.h`: helper headers used by the simulation.
- `Makefile`: includes Basilisk build defaults.

### Postprocessing (C)

- `lateralJet_postprocess.c`: loads params, restores a selected dump, and initializes postprocessing context.

### Postprocessing (Python)

- `postprocess.py`: orchestrates common postprocessing actions from one CLI.
- `interpolate_fields.py`: parses regular-grid field files (`fields.dat`) and builds interpolants.
- `vtk_export.py`: exports scalar/vector fields to legacy VTK for ParaView.
- `fields_to_vtk.py`: small CLI wrapper around `vtk_export.py`.
- `axis_fields.py`: samples all fields along axis `y = 0`.
- `fit_phiR_axis.py`: fits `1 - phiR` along the symmetry axis and estimates `Delta` using `Lj` from the first axis point where `f < 0.5`.
- `radial_lines.py`: exports field profiles along radial lines for selected `x` positions.
- `profiles.py`: computes/saves/plots average velocity and local BOE profiles.
- `radius_from_facets.py`: computes radius profile `R(x)` from Basilisk facets files.
- `sweep_boe.py`: automates BOE parameter sweeps using sequential restarts.

## Requirements

### Simulation

- Basilisk with `qcc`
- MPI compiler/runtime (`mpicc`, `mpirun`)
- Environment variable `BASILISK` configured

### Python tools

- Python 3.10+
- `numpy`
- `matplotlib` (for plots)
- `scipy` (optional; scripts include fallbacks for interpolation)

Install Python dependencies with:

```bash
pip install numpy matplotlib scipy
```

## Typical workflow

### 1) Run simulation

Compile and run the main solver according to your local Basilisk setup and parameters.

`lateralJet_params.c` supports:

- fresh run: `./lateralJet_params [params.txt]`
- forced restore: `./lateralJet_params --restore [params.txt]`
- forced fresh: `./lateralJet_params --fresh [params.txt]`
- explicit dump: `./lateralJet_params --restore --dump dump-... [params.txt]`

### 2) Build radius profile from facets (optional but common)

```bash
python radius_from_facets.py --facets-input facets --output radius_profile.dat
```

### 3) Export fields to VTK

```bash
python fields_to_vtk.py \
  --fields-input output_field/fields.dat \
  --output-dir post \
  --vtk-output all_fields.vtk
```

### 4) Extract axis fields

```bash
python axis_fields.py \
  --fields-input output_field/fields.dat \
  --output post/axis_fields.dat
```

### 5) Fit 1 - phiR on the axis

```bash
python fit_phiR_axis.py \
  --fields-input output_field/fields.dat \
  --data-output post/phiR_axis_fit.dat \
  --plot-output post/phiR_axis_fit.png
```

Useful options:

- `--f-threshold`: threshold to define `Lj` from the first axis location with `f < threshold` (default `0.5`).
- `--y-axis`: axis location to sample (default `0.0`).
- `--delta-min`, `--delta-max`, `--delta-guess`: bounds/initial guess for `Delta` fitting.

Outputs:

- data file with columns: `x`, `f_axis`, `phiR_axis`, `1_minus_phiR`, `fit`
- plot comparing sampled `1 - phiR` and fitted curve

### 6) Run combined Python postprocessing

```bash
python postprocess.py \
  --fields-input output_field/fields.dat \
  --radius-input radius_profile.dat \
  --output-dir post \
  --plot-radius \
  --plot-radial-lines \
  --export-all-fields-vtk
```

## BOE sweep workflow

Run a sequential BOE sweep (first case fresh, subsequent cases restored from previous dump):

```bash
python sweep_boe.py --np 4 --level 10
```

Options:

- `--no-build`: skip compilation and reuse existing executables
- `--np`: MPI ranks (also used as `NX` in this workflow)
- `--level`: initial refinement level

Outputs are stored under `sweep_boe/boe_*`.

## Input/Output conventions

- `fields.dat` is expected to be a regular `(x, y)` grid with a header containing column names.
- VTK output is legacy ASCII `STRUCTURED_POINTS`, readable by ParaView.
- Radius profile files use two columns: `x R(x)`.

## Notes

- Current Python orchestrator `postprocess.py` imports a module named `radius`.
  If your codebase only has `radius_from_facets.py`, keep compatibility by either:
  1) adding a small `radius.py` adapter, or
  2) updating imports in `postprocess.py` to use `radius_from_facets` APIs.