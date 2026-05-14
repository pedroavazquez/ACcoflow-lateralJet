"""
Sweep BOE values for lateralJet_params with sequential restarts.

How to run:
  python sweep_boe.py                # builds and runs with defaults
  python sweep_boe.py --np 4 --level 10
  python sweep_boe.py --no-build     # skip compilation, use existing binaries

What it does:
  - Creates one folder per BOE value under sweep_boe/.
  - Runs the first case from scratch, then restarts each subsequent case
    from the previous run's latest dump.
  - Advances total simulation time by run_dt per case.
  - Forces dumps every dump_dt and snapshots every snapshot_dt.

Troubleshooting:
  - Missing executables: ensure lateralJet_params_fresh.tst and
    lateralJet_params_restore.tst exist in the working directory.
  - params.txt not found: this script expects a base params.txt in the
    working directory (used as a template for each run).
  - mpirun errors: verify OpenMPI is installed and adjust --np as needed.
"""

import argparse
import os
import subprocess
from pathlib import Path

# --- CONSTANT SETTINGS ---
boe_values = [0,10,20]
run_dt = 10.0

source = "lateralJet_params.c"
exe_fresh = "./lateralJet_params_fresh"
exe_restore = "./lateralJet_params_restore"
base_params = "params.txt"
run_root = Path("sweep_boe")

# Compile-time defaults
DEFAULT_NP = 4
DEFAULT_LEVEL = 10
ELECTRIC = 1
DI = 1
# -------------------------

def build(np, level):
    """Compile the fresh and restore binaries using qcc."""
    basilisk_gl = os.path.join(os.environ.get("BASILISK", ""), "gl")
    common_flags = (
        f"-O2 -D_MPI={np} -DNX={np} -DLEVEL={level}"
        f" -DELECTRIC={ELECTRIC} -DDI={DI}"
    )
    link_flags = f"-L{basilisk_gl} -lglutils -lfb_tiny -lm"

    for restore, exe in [(0, exe_fresh), (1, exe_restore)]:
        cmd = (
            f"CC99='mpicc -std=c99' qcc {common_flags}"
            f" -DRESTORE={restore} {source} -o {exe} {link_flags}"
        )
        print(f"Building: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    print("Build complete.\n")

def read_params(path):
    """Read params.txt into a list of lines."""
    return Path(path).read_text().splitlines()

def write_params(path, lines):
    """Write params.txt lines with a trailing newline."""
    Path(path).write_text("\n".join(lines) + "\n")

def set_param(lines, key, value):
    """Set or append a scalar key in params.txt format (KEY = value)."""
    out = []
    found = False
    for line in lines:
        if line.strip().startswith(key + " "):
            out.append(f"{key} = {value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key} = {value}")
    return out

def main():
    """Run the BOE sweep using mpirun --oversubscribe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=DEFAULT_NP,
                        help=f"MPI ranks = NX (default: {DEFAULT_NP})")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL,
                        help=f"Grid refinement level (default: {DEFAULT_LEVEL})")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip compilation (use existing binaries)")
    args = parser.parse_args()

    if not args.no_build:
        build(args.np, args.level)

    mpirun_cmd = ["mpirun", "--oversubscribe", "-np", str(args.np)]

    # Resolve executable paths to absolute so they work from any cwd
    abs_exe_fresh = str(Path(exe_fresh).resolve())
    abs_exe_restore = str(Path(exe_restore).resolve())

    run_root.mkdir(exist_ok=True)

    prev_run_dir = None
    current_time = 0.0

    for i, boe in enumerate(boe_values):
        run_dir = run_root / f"boe_{boe:06.2f}".replace(".", "p")
        run_dir.mkdir(exist_ok=True)

        params_lines = read_params(base_params)
        params_lines = set_param(params_lines, "BOE", boe)

        current_time += run_dt
        params_lines = set_param(params_lines, "TEND", current_time)

        if i == 0:
            exe = abs_exe_fresh
        else:
            exe = abs_exe_restore
            dump_path = Path("..") / prev_run_dir.name / "dump"
            params_lines = set_param(params_lines, "DUMP_FILE", dump_path.as_posix())

        params_path = run_dir / "params.txt"
        write_params(params_path, params_lines)

        cmd = mpirun_cmd + [exe, "params.txt"]
        shell_cmd = " ".join(cmd) + " | tee out"
        print(f"Running: {shell_cmd} in {run_dir}")
        subprocess.run(f"set -o pipefail && {shell_cmd}", cwd=run_dir,
                       shell=True, executable="/bin/bash", check=True)

        prev_run_dir = run_dir

if __name__ == "__main__":
    main()
