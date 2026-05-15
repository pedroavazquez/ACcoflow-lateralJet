#!/usr/bin/env python3
"""
Fit 1 - phiR along the symmetry axis y = 0.

The input is a regular-grid fields file with columns including x, y, f, phiR,
for example the files written by readLateralJet_params.c or
lateralJet_postprocess.c.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interpolate_fields import build_interpolants, load_regular_grid, sanitize_name


def phi_fit_function(x: np.ndarray, delta: float, lj: float) -> np.ndarray:
    """Return the analytical fit function for 1 - phiR."""
    x = np.asarray(x, dtype=float)
    delta = float(delta)
    lj = float(lj)
    if delta <= 0.0:
        return np.full_like(x, np.nan, dtype=float)

    a = lj / delta
    b = (lj - x) / delta
    denom = np.cos(2.0 * a) + np.cosh(2.0 * a)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        term1 = 2.0 * np.cos(a) * np.cos(b) * np.cosh(a) * np.cosh(b)
        term2 = 2.0 * np.sin(a) * np.sin(b) * np.sinh(a) * np.sinh(b)
        return (term1 + term2) / denom


def extract_axis_fields(
    fields_input: str | Path,
    y_axis: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, f(x, y_axis), and phiR(x, y_axis)."""
    x, y, fields = load_regular_grid(fields_input)

    for required in ("f", "phiR"):
        if required not in fields:
            available = ", ".join(fields.keys())
            raise KeyError(f"Missing field '{required}'. Available fields: {available}")

    y_matches = np.flatnonzero(np.isclose(y, y_axis))
    if y_matches.size:
        j = int(y_matches[0])
        return x.copy(), fields["f"][:, j].copy(), fields["phiR"][:, j].copy()

    if y_axis < float(y.min()) or y_axis > float(y.max()):
        raise ValueError(
            f"Requested y_axis={y_axis} is outside input range [{y.min()}, {y.max()}]."
        )

    interpolants = build_interpolants(x, y, fields, bounds_error=False, fill_value=np.nan)
    points = np.column_stack((x, np.full(x.shape, float(y_axis), dtype=float)))
    f_axis = np.asarray(interpolants[f"interp_{sanitize_name('f')}"](points), dtype=float)
    phi_axis = np.asarray(
        interpolants[f"interp_{sanitize_name('phiR')}"](points), dtype=float
    )
    return x.copy(), f_axis, phi_axis


def find_jet_length(x: np.ndarray, f_axis: np.ndarray, threshold: float = 0.5) -> float:
    """Find Lj as the first x along the axis where f < threshold."""
    order = np.argsort(x)
    xs = np.asarray(x[order], dtype=float)
    fs = np.asarray(f_axis[order], dtype=float)

    valid = np.isfinite(xs) & np.isfinite(fs)
    below = np.flatnonzero(valid & (fs < threshold))
    if below.size == 0:
        raise ValueError(f"Could not find any axis point with f < {threshold}.")
    return float(xs[int(below[0])])


def _fit_with_scipy(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    lj: float,
    delta_min: float,
    delta_max: float,
    delta_guess: float,
) -> tuple[float, float]:
    """Fit delta using scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit  # type: ignore

    def model(x_values: np.ndarray, delta: float) -> np.ndarray:
        return phi_fit_function(x_values, delta, lj)

    popt, pcov = curve_fit(
        model,
        x_fit,
        y_fit,
        p0=[delta_guess],
        bounds=([delta_min], [delta_max]),
        maxfev=20000,
    )
    delta = float(popt[0])
    stderr = float(np.sqrt(pcov[0, 0])) if pcov.size and np.isfinite(pcov[0, 0]) else np.nan
    return delta, stderr


def _fit_with_grid_search(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    lj: float,
    delta_min: float,
    delta_max: float,
) -> tuple[float, float]:
    """SciPy-free fallback: minimize sum of squared residuals on a log grid."""
    deltas = np.geomspace(delta_min, delta_max, 2000)
    best_delta = deltas[0]
    best_sse = np.inf

    for delta in deltas:
        residual = y_fit - phi_fit_function(x_fit, float(delta), lj)
        sse = float(np.sum(residual * residual))
        if np.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best_delta = float(delta)

    return best_delta, np.nan


def fit_delta(
    x_axis: np.ndarray,
    phi_axis: np.ndarray,
    lj: float,
    delta_min: float | None = None,
    delta_max: float | None = None,
    delta_guess: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fit Delta to 1 - phiR for 0 <= x <= Lj."""
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = 1.0 - np.asarray(phi_axis, dtype=float)

    fit_mask = np.isfinite(x_axis) & np.isfinite(y_axis) & (x_axis >= 0.0) & (x_axis <= lj)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("Need at least three finite axis samples with 0 <= x <= Lj.")

    x_fit = x_axis[fit_mask]
    y_fit = y_axis[fit_mask]

    scale = max(abs(lj), np.ptp(x_fit), 1.0)
    delta_min = float(delta_min) if delta_min is not None else 1.0e-3 * scale
    delta_max = float(delta_max) if delta_max is not None else 100.0 * scale
    delta_guess = float(delta_guess) if delta_guess is not None else 0.25 * scale

    if not (0.0 < delta_min < delta_max):
        raise ValueError("Expected 0 < delta_min < delta_max.")
    delta_guess = float(np.clip(delta_guess, delta_min, delta_max))

    try:
        delta, stderr = _fit_with_scipy(
            x_fit, y_fit, lj, delta_min, delta_max, delta_guess
        )
    except Exception:
        delta, stderr = _fit_with_grid_search(x_fit, y_fit, lj, delta_min, delta_max)

    return delta, stderr, fit_mask


def save_fit_data(
    output: str | Path,
    x_axis: np.ndarray,
    f_axis: np.ndarray,
    phi_axis: np.ndarray,
    y_data: np.ndarray,
    y_model: np.ndarray,
) -> None:
    """Save axis data and fitted curve."""
    out = Path(output)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# 1:x 2:f_axis 3:phiR_axis 4:1_minus_phiR 5:fit\n")
        for row in zip(x_axis, f_axis, phi_axis, y_data, y_model):
            fp.write(" ".join(f"{float(value):.15e}" for value in row) + "\n")


def plot_fit(
    output: str | Path,
    x_axis: np.ndarray,
    y_data: np.ndarray,
    y_model: np.ndarray,
    lj: float,
    delta: float,
    fit_mask: np.ndarray,
    show: bool = False,
) -> None:
    """Plot 1 - phiR and the fitted function."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x_axis, y_data, "o", ms=3, label="1 - phiR")
    ax.plot(x_axis[fit_mask], y_model[fit_mask], "-", lw=2, label="fit")
    ax.axvline(lj, color="0.35", ls="--", lw=1, label=f"Lj = {lj:.6g}")
    ax.set_xlabel("x")
    ax.set_ylabel("1 - phiR")
    ax.set_title(f"Axis fit: Delta = {delta:.6g}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit 1 - phiR along y=0 using Lj from the first axis point with f < 0.5."
    )
    parser.add_argument(
        "--fields-input",
        "--input",
        dest="fields_input",
        default="fields",
        help="Input regular-grid fields file (default: fields).",
    )
    parser.add_argument(
        "--plot-output",
        default="phiR_axis_fit.png",
        help="Output plot filename (default: phiR_axis_fit.png).",
    )
    parser.add_argument(
        "--data-output",
        default="phiR_axis_fit.dat",
        help="Output data filename (default: phiR_axis_fit.dat).",
    )
    parser.add_argument("--y-axis", type=float, default=0.0, help="Axis location (default: 0).")
    parser.add_argument(
        "--f-threshold",
        type=float,
        default=0.5,
        help="Threshold used to define Lj from f_axis < threshold (default: 0.5).",
    )
    parser.add_argument("--delta-min", type=float, default=None, help="Lower Delta bound.")
    parser.add_argument("--delta-max", type=float, default=None, help="Upper Delta bound.")
    parser.add_argument("--delta-guess", type=float, default=None, help="Initial Delta guess.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    x_axis, f_axis, phi_axis = extract_axis_fields(args.fields_input, y_axis=args.y_axis)
    lj = find_jet_length(x_axis, f_axis, threshold=args.f_threshold)
    delta, stderr, fit_mask = fit_delta(
        x_axis=x_axis,
        phi_axis=phi_axis,
        lj=lj,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        delta_guess=args.delta_guess,
    )

    y_data = 1.0 - phi_axis
    y_model = phi_fit_function(x_axis, delta, lj)
    save_fit_data(args.data_output, x_axis, f_axis, phi_axis, y_data, y_model)
    plot_fit(args.plot_output, x_axis, y_data, y_model, lj, delta, fit_mask, show=args.show)

    print(f"Input fields: {args.fields_input}")
    print(f"Axis points: {x_axis.size}")
    print(f"Lj: {lj:.15g}")
    print(f"Delta: {delta:.15g}")
    if np.isfinite(stderr):
        print(f"Delta standard error: {stderr:.15g}")
    print(f"Saved data: {args.data_output}")
    print(f"Saved plot: {args.plot_output}")


if __name__ == "__main__":
    main()
