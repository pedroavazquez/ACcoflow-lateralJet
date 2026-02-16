#!/usr/bin/env python3
"""
Read jet radius data (x, radius), build a 1D interpolant R(x), and plot radius vs x.

Expected file format:
  - Comment/header lines start with '#'
  - Data lines contain two columns:
      1:x 2:radius
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def load_radius_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load x and radius columns from a file with '#' comments."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns (x, radius) in '{path}', got shape={data.shape}"
        )

    x = data[:, 0].astype(float)
    r = data[:, 1].astype(float)

    # Ensure monotone x for interpolation.
    order = np.argsort(x)
    x = x[order]
    r = r[order]

    # If duplicate x values exist, keep the last value for each x.
    x_unique, idx = np.unique(x, return_index=True)
    if x_unique.size != x.size:
        rev_x = x[::-1]
        rev_r = r[::-1]
        x_last, rev_idx = np.unique(rev_x, return_index=True)
        x = x_last[::-1]
        r = rev_r[rev_idx][::-1]

    if x.size < 2:
        raise ValueError("Need at least two points to build an interpolant.")

    return x, r


def build_radius_interpolant(
    x: np.ndarray, r: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build linear interpolant R(x).
    Uses SciPy interp1d when available, otherwise numpy.interp fallback.
    """
    try:
        from scipy.interpolate import interp1d  # type: ignore

        interp_obj = interp1d(
            x,
            r,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        return lambda xq: np.asarray(interp_obj(xq), dtype=float)
    except Exception:
        return lambda xq: np.interp(
            np.asarray(xq, dtype=float),
            x,
            r,
            left=np.nan,
            right=np.nan,
        )


def plot_radius_profile(
    x: np.ndarray,
    r: np.ndarray,
    r_interp: Callable[[np.ndarray], np.ndarray],
    output: str | Path,
    n_plot: int = 1000,
    show: bool = False,
) -> None:
    """Plot raw data and interpolated profile R(x)."""
    x_dense = np.linspace(x.min(), x.max(), n_plot)
    r_dense = r_interp(x_dense)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_dense, r_dense, "-", lw=2, label="Interpolated radius R(x)")
    plt.plot(x, r, "o", ms=3, alpha=0.7, label="Input data")
    plt.xlabel("x")
    plt.ylabel("radius")
    plt.title("Jet radius profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interpolant for jet radius and plot radius vs x."
    )
    parser.add_argument(
        "--input",
        default="jet_radius_eq.dat",
        help="Input file with columns: x radius (default: jet_radius_eq.dat).",
    )
    parser.add_argument(
        "--output",
        default="jet_radius_vs_x.png",
        help="Output figure path (default: jet_radius_vs_x.png).",
    )
    parser.add_argument(
        "--n-plot",
        type=int,
        default=1000,
        help="Number of x points for plotting interpolated curve (default: 1000).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window in addition to saving.",
    )
    args = parser.parse_args()

    x, r = load_radius_data(args.input)
    r_interp = build_radius_interpolant(x, r)
    plot_radius_profile(
        x=x,
        r=r,
        r_interp=r_interp,
        output=args.output,
        n_plot=args.n_plot,
        show=args.show,
    )

    print(f"Loaded {x.size} points from: {args.input}")
    print(f"x range: [{x.min():.9g}, {x.max():.9g}]")
    print(f"radius range: [{r.min():.9g}, {r.max():.9g}]")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
