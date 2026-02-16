#!/usr/bin/env python3
"""
Read a Basilisk-style fields.dat file and build interpolants for multiple fields.

Expected format:
  - First line contains column names, e.g.:
      # 1:x 2:y 3:f 4:ux 5:phiR
    or
      # x y f ux phiR
  - Remaining lines are numeric rows on a regular (x, y) grid.

Usage examples:
  python interpolate_fields.py --input readLateralJet_params/fields.dat
  python interpolate_fields.py --input readLateralJet_params/fields.dat --query 0.5 0.2
  python interpolate_fields.py --input readLateralJet_params/fields.dat --save-grid field_grid.npz
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_header_column_names(path: str | Path) -> list[str]:
    """Parse column names from the first non-empty line."""
    first = None
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                first = line.strip()
                break

    if first is None:
        raise ValueError(f"Empty file: {path}")

    if first.startswith("#"):
        first = first[1:].strip()

    tokens = first.split()
    names: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^\d+:(.+)$", tok)
        if m:
            names.append(m.group(1))
        else:
            names.append(tok)

    return names


def normalize_column_names(names: list[str], ncols: int) -> list[str]:
    """Make names unique and consistent with number of numeric columns."""
    if len(names) < ncols:
        for i in range(len(names), ncols):
            if i == 0:
                names.append("x")
            elif i == 1:
                names.append("y")
            else:
                names.append(f"field{i + 1}")
    elif len(names) > ncols:
        names = names[:ncols]

    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        base = name
        if base not in seen:
            seen[base] = 0
            unique_names.append(base)
        else:
            seen[base] += 1
            unique_names.append(f"{base}_{seen[base]}")
    return unique_names


def sanitize_name(name: str) -> str:
    """Map a field name to a safe identifier suffix."""
    out = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    out = out.strip("_")
    if not out:
        out = "field"
    if out[0].isdigit():
        out = f"f_{out}"
    return out


def load_regular_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Load regular (x, y) grid and all field columns.

    Returns
    -------
    x_unique : shape (nx,)
        Sorted x coordinates.
    y_unique : shape (ny,)
        Sorted y coordinates.
    fields : dict[name -> ndarray(nx, ny)]
        Field values on the regular grid for each field column.
    """
    raw_names = parse_header_column_names(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: x y <field...>")

    names = normalize_column_names(raw_names, data.shape[1])

    x = data[:, 0]
    y = data[:, 1]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = x_unique.size
    ny = y_unique.size

    if data.shape[0] != nx * ny:
        raise ValueError(
            f"Data are not a full regular grid: n_points={data.shape[0]}, nx*ny={nx*ny}"
        )

    ix = np.searchsorted(x_unique, x)
    iy = np.searchsorted(y_unique, y)

    fields: dict[str, np.ndarray] = {}
    for col in range(2, data.shape[1]):
        field_name = names[col]
        grid = np.full((nx, ny), np.nan, dtype=float)
        grid[ix, iy] = data[:, col]
        if np.isnan(grid).any():
            raise ValueError(
                f"Missing (x, y) points while building grid for field '{field_name}'."
            )
        fields[field_name] = grid

    return x_unique, y_unique, fields


@dataclass
class BilinearInterpolator:
    """Simple regular-grid bilinear interpolator (SciPy-free fallback)."""

    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    bounds_error: bool = False
    fill_value: float = np.nan

    def __call__(self, points: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray | float:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
            scalar_out = True
        else:
            scalar_out = False

        if pts.shape[1] != 2:
            raise ValueError("Points must have shape (n, 2) with columns [x, y].")

        px = pts[:, 0]
        py = pts[:, 1]

        xmin, xmax = self.x[0], self.x[-1]
        ymin, ymax = self.y[0], self.y[-1]
        inside = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)

        if self.bounds_error and not np.all(inside):
            raise ValueError("Some query points are outside the interpolation domain.")

        out = np.full(px.shape, self.fill_value, dtype=float)
        if not np.any(inside):
            return out[0] if scalar_out else out

        pxi = px[inside]
        pyi = py[inside]

        i = np.searchsorted(self.x, pxi, side="right") - 1
        j = np.searchsorted(self.y, pyi, side="right") - 1
        i = np.clip(i, 0, self.x.size - 2)
        j = np.clip(j, 0, self.y.size - 2)

        x0 = self.x[i]
        x1 = self.x[i + 1]
        y0 = self.y[j]
        y1 = self.y[j + 1]

        tx = (pxi - x0) / (x1 - x0)
        ty = (pyi - y0) / (y1 - y0)

        f00 = self.f[i, j]
        f10 = self.f[i + 1, j]
        f01 = self.f[i, j + 1]
        f11 = self.f[i + 1, j + 1]

        out_inside = (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )

        out[inside] = out_inside
        return out[0] if scalar_out else out


def build_interpolant(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    bounds_error: bool = False,
    fill_value: float = np.nan,
):
    """
    Build interpolant f(x, y).
    Uses SciPy RegularGridInterpolator when available; otherwise bilinear fallback.
    """
    try:
        from scipy.interpolate import RegularGridInterpolator  # type: ignore

        return RegularGridInterpolator(
            (x, y),
            f,
            method="linear",
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
    except Exception:
        return BilinearInterpolator(x=x, y=y, f=f, bounds_error=bounds_error, fill_value=fill_value)


def build_interpolants(
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    bounds_error: bool = False,
    fill_value: float = np.nan,
) -> dict[str, object]:
    """
    Build one interpolant per field.
    Returned keys are named as interp_<fieldname>.
    """
    out: dict[str, object] = {}
    for field_name, grid in fields.items():
        interp_name = f"interp_{sanitize_name(field_name)}"
        out[interp_name] = build_interpolant(
            x=x,
            y=y,
            f=grid,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
    return out


def compute_jet_radius_along_axis(
    x: np.ndarray,
    y: np.ndarray,
    f_grid: np.ndarray,
    threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Compute jet radius R(x) from a VOF field on a regular grid.

    For each axial position x[i], the radius is defined as the first y[j]
    such that f_grid[i, j] < threshold.

    Returns
    -------
    list of (x, radius) tuples.

    Notes
    -----
    - If no y satisfies f < threshold for a given x, radius is set to y[-1].
    - Assumes y is ordered from axis (small y) to outer boundary (large y).
    """
    if f_grid.shape != (x.size, y.size):
        raise ValueError(
            f"Shape mismatch: f_grid.shape={f_grid.shape} but expected {(x.size, y.size)}"
        )

    radius_profile: list[tuple[float, float]] = []
    for i, x_val in enumerate(x):
        below = np.flatnonzero(f_grid[i, :] < threshold)
        if below.size == 0:
            radius = float(y[-1])
        else:
            radius = float(y[below[0]])
        radius_profile.append((float(x_val), radius))

    return radius_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interpolants for all fields from a regular-grid fields.dat file."
    )
    parser.add_argument(
        "--input",
        default="fields.dat",
        help="Path to fields.dat file (default: fields.dat)",
    )
    parser.add_argument(
        "--query",
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        type=float,
        help="Evaluate all interpolants at one point; can be repeated.",
    )
    parser.add_argument(
        "--save-grid",
        default=None,
        help="Optional .npz output path to save x, y and all field grids.",
    )
    args = parser.parse_args()

    x, y, fields = load_regular_grid(args.input)
    interpolants = build_interpolants(x, y, fields, bounds_error=False, fill_value=np.nan)

    dx = np.diff(x)
    dy = np.diff(y)
    print(f"Loaded: {args.input}")
    print(f"Grid size: nx={x.size}, ny={y.size}, points={x.size*y.size}")
    print(f"x range: [{x[0]:.9g}, {x[-1]:.9g}]")
    print(f"y range: [{y[0]:.9g}, {y[-1]:.9g}]")
    print(f"dx ~ {np.mean(dx):.9g} (min={np.min(dx):.9g}, max={np.max(dx):.9g})")
    print(f"dy ~ {np.mean(dy):.9g} (min={np.min(dy):.9g}, max={np.max(dy):.9g})")
    print(f"Detected fields: {', '.join(fields.keys())}")
    print("Interpolants:")
    for field_name in fields:
        print(f"  interp_{sanitize_name(field_name)}")

    if args.save_grid:
        out_path = Path(args.save_grid)
        save_dict = {"x": x, "y": y}
        for field_name, grid in fields.items():
            save_dict[f"field_{sanitize_name(field_name)}"] = grid
        np.savez(out_path, **save_dict)
        print(f"Saved grid arrays to: {out_path}")

    if args.query:
        q = np.asarray(args.query, dtype=float)
        print("Interpolated values:")
        for idx, (qx, qy) in enumerate(q, start=1):
            print(f"  Point {idx}: (x={qx:.9g}, y={qy:.9g})")
            for field_name in fields:
                interp_name = f"interp_{sanitize_name(field_name)}"
                val = interpolants[interp_name](np.array([[qx, qy]], dtype=float))[0]
                print(f"    {interp_name} = {float(val):.15g}")



if __name__ == "__main__":
    main()
