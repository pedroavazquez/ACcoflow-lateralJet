from __future__ import annotations

from pathlib import Path

import numpy as np


def export_fields_along_radial_lines(
    x_profile: np.ndarray,
    interpolants: dict[str, object],
    x_min: float,
    x_max: float,
    n_lines: int = 5,
    nr: int = 256,
    output_dir: str | Path | None = None,
) -> None:
    """
    Export all fields along radial lines at specified x positions to text files.

    For each x in np.linspace(x_min, x_max, n_lines), evaluates all
    interpolants along r = [0, 1] and saves to files.

    Each output file is named `fields_radial_x_{x_value:.6f}.dat` with columns:
      1:r  2:field1  3:field2  ...

    Parameters
    ----------
    x_profile : np.ndarray
        Array of x values from the fields grid (used to find valid x range).
    interpolants : dict[str, object]
        Precomputed interpolants from build_interpolants().
    x_min : float
        Minimum x value for radial lines.
    x_max : float
        Maximum x value for radial lines.
    n_lines : int
        Number of radial lines to export (default: 5).
    nr : int
        Number of radial samples per line (default: 256).
    output_dir : str | Path | None
        Directory to save output files; if None, current directory is used.
    """
    x_min = max(x_min, float(x_profile.min()))
    x_max = min(x_max, float(x_profile.max()))
    nr = max(2, int(nr))
    n_lines = max(1, int(n_lines))

    x_lines = np.linspace(x_min, x_max, n_lines)
    output_dir_path = Path(output_dir) if output_dir else Path(".")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Extract field names, skipping the 'interp_' prefix
    field_names = sorted(
        [k.replace("interp_", "") for k in interpolants.keys() if k.startswith("interp_")]
    )

    for x_val in x_lines:
        r_line = np.linspace(0.0, 1.0, nr)
        pts = np.column_stack(
            (
                np.full(r_line.shape, float(x_val), dtype=float),
                r_line,
            )
        )

        # Collect all field values for this radial line
        data_dict = {"r": r_line}
        for field_name in field_names:
            interp_key = f"interp_{field_name}"
            if interp_key in interpolants:
                interp_obj = interpolants[interp_key]
                values = np.asarray(interp_obj(pts), dtype=float)
                data_dict[field_name] = values

        # Save to file
        filename = output_dir_path / f"fields_radial_x_{x_val:.6f}.dat"
        with filename.open("w", encoding="utf-8") as fp:
            # Write header
            header_parts = ["r"] + field_names
            header_line = " ".join([f"{i+1}:{name}" for i, name in enumerate(header_parts)])
            fp.write(f"# {header_line}\n")

            # Write data
            for j in range(nr):
                line_parts = [str(data_dict["r"][j])]
                for field_name in field_names:
                    line_parts.append(str(data_dict[field_name][j]))
                fp.write(" ".join(line_parts) + "\n")

        print(f"Saved radial line data at x={x_val:.6f} to: {filename}")
