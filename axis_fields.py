#!/usr/bin/env python3
"""
Export all fields along the symmetry axis y = 0.

The input is a regular-grid Basilisk fields file with columns:

    x y field1 field2 ...

The output is a text file with columns:

    x field1 field2 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from interpolate_fields import build_interpolants, load_regular_grid, sanitize_name


def _format_header(names: list[str]) -> str:
    """Return a numbered header line compatible with the other postprocessing files."""
    return "# " + " ".join(f"{i + 1}:{name}" for i, name in enumerate(names))


def sample_fields_along_axis(
    fields_input: str | Path,
    y_axis: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    n_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Sample all fields along y = y_axis.

    If n_samples is None, values are taken at the original grid x coordinates.
    If n_samples is given, fields are linearly interpolated onto a uniform x grid.
    """
    x, y, fields = load_regular_grid(fields_input)

    if x_min is None:
        x_min = float(x.min())
    if x_max is None:
        x_max = float(x.max())

    if x_min < float(x.min()) or x_max > float(x.max()):
        raise ValueError(
            f"Requested x range [{x_min}, {x_max}] is outside "
            f"input range [{x.min()}, {x.max()}]."
        )
    if y_axis < float(y.min()) or y_axis > float(y.max()):
        raise ValueError(
            f"Requested y_axis={y_axis} is outside input range [{y.min()}, {y.max()}]."
        )

    if n_samples is None:
        mask = (x >= x_min) & (x <= x_max)
        x_axis = x[mask]

        y_matches = np.flatnonzero(np.isclose(y, y_axis))
        if y_matches.size:
            j = int(y_matches[0])
            axis_fields = {
                field_name: grid[mask, j].copy()
                for field_name, grid in fields.items()
            }
            return x_axis, axis_fields

        n_samples = x_axis.size

    n_samples = max(2, int(n_samples))
    x_axis = np.linspace(float(x_min), float(x_max), n_samples)
    points = np.column_stack(
        (
            x_axis,
            np.full(x_axis.shape, float(y_axis), dtype=float),
        )
    )

    interpolants = build_interpolants(x, y, fields, bounds_error=False, fill_value=np.nan)
    axis_fields = {}
    for field_name in fields:
        interp_name = f"interp_{sanitize_name(field_name)}"
        axis_fields[field_name] = np.asarray(interpolants[interp_name](points), dtype=float)

    return x_axis, axis_fields


def save_axis_fields(
    x_axis: np.ndarray,
    axis_fields: dict[str, np.ndarray],
    output: str | Path,
) -> None:
    """Save axis field values to a text file."""
    out = Path(output)
    field_names = list(axis_fields.keys())

    with out.open("w", encoding="utf-8") as fp:
        fp.write(_format_header(["x", *field_names]) + "\n")
        for i, x_value in enumerate(x_axis):
            values = [x_value] + [axis_fields[name][i] for name in field_names]
            fp.write(" ".join(f"{value:.15e}" for value in values) + "\n")


def export_fields_along_axis(
    fields_input: str | Path,
    output: str | Path,
    y_axis: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    n_samples: int | None = None,
) -> tuple[int, list[str]]:
    """Sample all fields along the symmetry axis and save them."""
    x_axis, axis_fields = sample_fields_along_axis(
        fields_input=fields_input,
        y_axis=y_axis,
        x_min=x_min,
        x_max=x_max,
        n_samples=n_samples,
    )
    save_axis_fields(x_axis, axis_fields, output)
    return x_axis.size, list(axis_fields.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all fields along the symmetry axis y=0."
    )
    parser.add_argument(
        "--fields-input",
        required=True,
        help="Input regular-grid fields file.",
    )
    parser.add_argument(
        "--output",
        default="axis_fields.dat",
        help="Output file with columns x and all field values (default: axis_fields.dat).",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum x value (default: input minimum).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum x value, normally LX (default: input maximum).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional number of uniform x samples. Default uses input x grid points.",
    )
    args = parser.parse_args()

    n_points, field_names = export_fields_along_axis(
        fields_input=args.fields_input,
        output=args.output,
        x_min=args.x_min,
        x_max=args.x_max,
        n_samples=args.n_samples,
    )

    print(f"Saved axis fields to: {args.output}")
    print(f"Axis points: {n_points}")
    print(f"Fields: {', '.join(field_names)}")


if __name__ == "__main__":
    main()
