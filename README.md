# Lateral Jet (multigrid, viscous scales)

Axisymmetric, two-phase immersed coflow jet simulation using Basilisk multigrid,
with optional AC electric field coupling. This directory contains the main
simulation, a reader for parameterized restarts, and a small sweep helper.

## Files
- `lateralJet_params.c`: Main simulation with parameter file I/O and
  compile-time overrides. Supports fresh runs and restarts.
- `readlateralJet_params.c`: Variant focused on restore runs with parameters
  loaded from a text file.
- `drop_stat.h`, `poisson_complex.h`: Support headers for statistics and
  complex Poisson solves used by the electric model.
- `Makefile`: Minimal Basilisk build include.
- `params-00.txt`: Example parameter file template.
- `sweep_boe.py`: BOE sweep runner with sequential restarts.

## Build
The code is compiled with Basilisk `qcc` and MPI. A typical build line is:

```bash
CC='mpicc -D_MPI=_ -DNX=_ -DLEVEL=_ -DELECTRIC=_ -DDI=_' make lateralJet_params.tst
```

For `sweep_boe.py`, the script builds two binaries (fresh and restore) using:

```
qcc -DRESTORE=0 ... -o lateralJet_params_fresh
qcc -DRESTORE=1 ... -o lateralJet_params_restore
```

### Compile-time options
These are fixed at build time (via `-D` flags):
- `NX`: number of boxes in x (must equal MPI ranks).
- `LEVEL`: initial refinement level.
- `ELECTRIC`: `1` enable electric field; `0` disable.
- `DI`: `1` use distributed impedance (Robin) boundary; `0` Dirichlet.
- `RESTORE`: `1` start from dump; `0` fresh run. (Used in `lateralJet_params.c`.)

You can also override selected runtime parameters at compile time with
`-D_XXX=value` (see `apply_compile_time_overrides` in the `.c` files).

## Runtime parameters
Parameters are stored in a `params.txt`-style file and read at runtime:

```
./lateralJet_params [params.txt]
```

Key parameters (see `params-00.txt` for a full list):
- Geometry: `R1`, `R2`, `LIN`, `LBOX`
- Electrical: `EPSR1`, `EPSR2`, `SIGMA1`, `SIGMA2`, `CL`
- Viscosity: `MU1`, `MU2`, `OH1`, `OH2`
- Capillary: `CA1`, `CA2`
- Electric field: `FREQ`, `BOE`, `TELEC`, `V0`
- Surface tension: `GAMMA`
- Time control: `TEND`, `DTOUT`, `DTDUMP`, `DTMAXMINE`
- Restore: `DUMP_FILE` (path to latest dump for restart)

## Sweep helper
`sweep_boe.py` runs a sequential BOE sweep:

```bash
python sweep_boe.py --np 4 --level 10
```

It creates one folder per BOE value under `sweep_boe/`, uses a base `params.txt`
template, and restarts each case from the previous dump. Use `--no-build` to
skip compilation.
