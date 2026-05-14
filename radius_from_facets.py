#!/usr/bin/env python3
"""
Build a jet-radius profile R(x) from a Basilisk facets file.

A facets file is expected to contain interface segments as pairs of points:

    x0 y0
    x1 y1

    x2 y2
    x3 y3

Blank lines separate segments. The radius profile is computed by intersecting
vertical lines x = constant with the interface segments and taking the first
nonnegative y value, i.e. the interface closest to the axis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RadiusFromFacets:
    """Container for a radius profile and its interpolant."""

    x: np.ndarray
    radius: np.ndarray
    interpolant: Callable[[np.ndarray], np.ndarray]


def read_facet_segments(path: str | Path) -> np.ndarray:
    """
    Read a Basilisk facets file.

    Returns
    -------
    segments : ndarray, shape (n_segments, 2, 2)
        Segment endpoints. For segment k:
        segments[k, 0] = [x0, y0], segments[k, 1] = [x1, y1].
    """
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    with open(path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if current:
                    if len(current) != 2:
                        raise ValueError(
                            f"Malformed segment ending near line {line_number}: "
                            f"expected 2 points, got {len(current)}."
                        )
                    segments.append(current)
                    current = []
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Expected x y at line {line_number}: {line!r}")
            current.append((float(parts[0]), float(parts[1])))
            if len(current) == 2:
                segments.append(current)
                current = []

    if current:
        if len(current) != 2:
            raise ValueError(f"Malformed final segment: expected 2 points, got {len(current)}.")
        segments.append(current)

    if not segments:
        raise ValueError(f"No facet segments found in: {path}")

    return np.asarray(segments, dtype=float)


def _segment_crossings_at_x(segments: np.ndarray, x_value: float) -> np.ndarray:
    """Return all y crossings of the facet segments with x = x_value."""
    x0 = segments[:, 0, 0]
    y0 = segments[:, 0, 1]
    x1 = segments[:, 1, 0]
    y1 = segments[:, 1, 1]

    xmin = np.minimum(x0, x1)
    xmax = np.maximum(x0, x1)
    in_range = (x_value >= xmin) & (x_value <= xmax)

    vertical = in_range & np.isclose(x0, x1)
    nonvertical = in_range & ~np.isclose(x0, x1)

    crossings: list[np.ndarray] = []

    if np.any(nonvertical):
        t = (x_value - x0[nonvertical]) / (x1[nonvertical] - x0[nonvertical])
        crossings.append(y0[nonvertical] + t * (y1[nonvertical] - y0[nonvertical]))

    if np.any(vertical):
        crossings.append(y0[vertical])
        crossings.append(y1[vertical])

    if not crossings:
        return np.asarray([], dtype=float)

    return np.concatenate(crossings)


def compute_radius_profile_from_segments(
    segments: np.ndarray,
    x_samples: np.ndarray | None = None,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute R(x) from facet segments.

    At each sampled x, all interface crossings are found and the radius is the
    smallest nonnegative y crossing. Samples without crossings are discarded.
    """
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError(
            f"Expected segments with shape (n_segments, 2, 2), got {segments.shape}."
        )

    if x_samples is None:
        if n_samples is None:
            x_samples = np.unique(segments[:, :, 0].ravel())
        else:
            xmin = float(np.min(segments[:, :, 0]))
            xmax = float(np.max(segments[:, :, 0]))
            x_samples = np.linspace(xmin, xmax, max(2, int(n_samples)))
    else:
        x_samples = np.asarray(x_samples, dtype=float)

    x_out: list[float] = []
    r_out: list[float] = []

    for x_value in x_samples:
        crossings = _segment_crossings_at_x(segments, float(x_value))
        crossings = crossings[np.isfinite(crossings) & (crossings >= 0.0)]
        if crossings.size == 0:
            continue
        x_out.append(float(x_value))
        r_out.append(float(np.min(crossings)))

    if len(x_out) < 2:
        raise ValueError("Need at least two valid radius points to build an interpolant.")

    x_arr = np.asarray(x_out, dtype=float)
    r_arr = np.asarray(r_out, dtype=float)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    r_arr = r_arr[order]

    x_unique, unique_idx = np.unique(x_arr, return_index=True)
    if x_unique.size != x_arr.size:
        r_unique = np.empty_like(x_unique)
        for i, x_value in enumerate(x_unique):
            r_unique[i] = np.min(r_arr[x_arr == x_value])
        x_arr = x_unique
        r_arr = r_unique
    else:
        _ = unique_idx

    return x_arr, r_arr


def build_radius_interpolant(
    x: np.ndarray,
    radius: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a linear interpolant R(x).

    Uses SciPy when available, otherwise falls back to numpy.interp.
    """
    try:
        from scipy.interpolate import interp1d  # type: ignore

        interp_obj = interp1d(
            x,
            radius,
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
            radius,
            left=np.nan,
            right=np.nan,
        )


def load_radius_from_facets(
    facets_input: str | Path,
    n_samples: int | None = None,
) -> RadiusFromFacets:
    """Read facets, compute R(x), and return the profile plus interpolant."""
    segments = read_facet_segments(facets_input)
    x, radius = compute_radius_profile_from_segments(segments, n_samples=n_samples)
    return RadiusFromFacets(
        x=x,
        radius=radius,
        interpolant=build_radius_interpolant(x, radius),
    )


def save_radius_profile(
    x: np.ndarray,
    radius: np.ndarray,
    output: str | Path,
) -> None:
    """Save radius profile as two columns: x R(x)."""
    out = Path(output)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# 1:x 2:R(x)\n")
        for x_value, radius_value in zip(x, radius):
            fp.write(f"{x_value:.15e} {radius_value:.15e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a jet-radius interpolant from a Basilisk facets file."
    )
    parser.add_argument(
        "--facets-input",
        default="facets",
        help="Input Basilisk facets file (default: facets).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file for the profile in columns: x R(x).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of uniform x samples. Default uses unique segment endpoint x values.",
    )
    args = parser.parse_args()

    profile = load_radius_from_facets(args.facets_input, n_samples=args.n_samples)

    print(f"Loaded facets from: {args.facets_input}")
    print(f"Radius points: {profile.x.size}")
    print(f"x range: [{profile.x.min():.9g}, {profile.x.max():.9g}]")
    print(f"R range: [{profile.radius.min():.9g}, {profile.radius.max():.9g}]")

    if args.output:
        save_radius_profile(profile.x, profile.radius, args.output)
        print(f"Saved radius profile to: {args.output}")


if __name__ == "__main__":
    main()
