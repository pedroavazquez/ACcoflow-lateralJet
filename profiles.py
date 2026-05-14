from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def _extract_velocity_interpolants(interpolants: dict[str, object]) -> tuple[object, object]:
    """Return (interp_u_x, interp_u_y) from interpolant dictionary."""
    if "interp_u_x" in interpolants and "interp_u_y" in interpolants:
        return interpolants["interp_u_x"], interpolants["interp_u_y"]

    if "interp_ux" in interpolants and "interp_uy" in interpolants:
        return interpolants["interp_ux"], interpolants["interp_uy"]

    keys = ", ".join(sorted(interpolants.keys()))
    raise KeyError(
        "Could not find velocity interpolants. Expected 'interp_u_x'/'interp_u_y' "
        f"(or fallback 'interp_ux'/'interp_uy'). Available keys: {keys}"
    )


def _extract_ery_interpolant(interpolants: dict[str, object]) -> object:
    """Return ERy interpolant from interpolant dictionary."""
    if "interp_ER_y" in interpolants:
        return interpolants["interp_ER_y"]
    if "interp_ERy" in interpolants:
        return interpolants["interp_ERy"]

    keys = ", ".join(sorted(interpolants.keys()))
    raise KeyError(
        "Could not find ERy interpolant. Expected 'interp_ER_y' "
        f"(or fallback 'interp_ERy'). Available keys: {keys}"
    )


def compute_axisymmetric_average_velocity_profile(
    x_profile: np.ndarray,
    radius_interp: Callable[[np.ndarray], np.ndarray],
    interp_u_x: object,
    nr: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute axisymmetric cross-sectional average axial velocity vs x.

    For each x, with jet radius R(x), computes:
      U_avg(x) = [int_0^R u_x(x, r) * r dr] / [int_0^R r dr]
               = (2/R^2) * int_0^R u_x(x, r) * r dr
    """
    nr = max(2, int(nr))
    u_avg = np.full(x_profile.shape, np.nan, dtype=float)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    for i, x_val in enumerate(x_profile):
        r_max = float(radius_interp(np.array([x_val], dtype=float))[0])
        if not np.isfinite(r_max) or r_max <= 0.0:
            continue

        r_line = np.linspace(0.0, r_max, nr)
        pts = np.column_stack(
            (
                np.full(r_line.shape, float(x_val), dtype=float),
                r_line,
            )
        )
        u_line = np.asarray(interp_u_x(pts), dtype=float)
        valid = np.isfinite(u_line) & np.isfinite(r_line)
        if np.count_nonzero(valid) < 2:
            continue

        rv = r_line[valid]
        uv = u_line[valid]
        denom = integrate(rv, rv)
        if denom <= 0.0:
            continue
        num = integrate(uv * rv, rv)
        u_avg[i] = num / denom

    return x_profile, u_avg


def save_average_velocity_profile(
    x: np.ndarray,
    u_avg: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save x and axisymmetric average velocity to a text file."""
    out = Path(output_path)
    mask = np.isfinite(x) & np.isfinite(u_avg)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# Axisymmetric average axial velocity profile\n")
        fp.write("# U_avg(x) = [int_0^R u_x(x,r) r dr]/[int_0^R r dr]\n")
        fp.write("# 1:x 2:U_avg\n")
        for xv, uv in zip(x[mask], u_avg[mask]):
            fp.write(f"{xv:.15e} {uv:.15e}\n")


def plot_average_velocity_profile(
    x: np.ndarray,
    u_avg: np.ndarray,
    output_path: str | Path,
    show: bool = False,
) -> None:
    """Plot axisymmetric average velocity versus x."""
    mask = np.isfinite(x) & np.isfinite(u_avg)
    if not np.any(mask):
        raise ValueError("No finite values available to plot average velocity profile.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], u_avg[mask], "-", lw=2, label="Axisymmetric U_avg(x)")
    plt.xlabel("x")
    plt.ylabel("U_avg")
    plt.title("Average axial velocity for r <= R(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def compute_local_boe_profile(
    x_profile: np.ndarray,
    radius_interp: Callable[[np.ndarray], np.ndarray],
    interp_er_y: object,
    boe_global: float,
    nr: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local electric Bond number along x.

    For each x:
      R = radius(x)
      Er_max = max_{0 <= r <= R} |ERy(x, r)|
      BOE_local = R * Er_max^2 * BOE_global
    """
    nr = max(2, int(nr))
    boe_local = np.full(x_profile.shape, np.nan, dtype=float)

    for i, x_val in enumerate(x_profile):
        r_max = float(radius_interp(np.array([x_val], dtype=float))[0])
        if not np.isfinite(r_max) or r_max <= 0.0:
            continue

        r_line = np.linspace(0.0, r_max, nr)
        pts = np.column_stack(
            (
                np.full(r_line.shape, float(x_val), dtype=float),
                r_line,
            )
        )
        er_line = np.asarray(interp_er_y(pts), dtype=float)
        valid = np.isfinite(er_line)
        if not np.any(valid):
            continue

        er_max = float(np.max(np.abs(er_line[valid])))
        boe_local[i] = r_max * (er_max**2) * boe_global

    return x_profile, boe_local


def save_local_boe_profile(
    x: np.ndarray,
    boe_local: np.ndarray,
    boe_global: float,
    output_path: str | Path,
) -> None:
    """Save x and local BOE profile to a text file."""
    out = Path(output_path)
    mask = np.isfinite(x) & np.isfinite(boe_local)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# Local electric Bond number profile\n")
        fp.write("# BOE_local(x) = R(x) * max_r(|ERy(x,r)|)^2 * BOE_global\n")
        fp.write(f"# BOE_global = {boe_global:.15e}\n")
        fp.write("# 1:x 2:BOE_local\n")
        for xv, bv in zip(x[mask], boe_local[mask]):
            fp.write(f"{xv:.15e} {bv:.15e}\n")


def plot_local_boe_profile(
    x: np.ndarray,
    boe_local: np.ndarray,
    output_path: str | Path,
    show: bool = False,
) -> None:
    """Plot local BOE versus x."""
    mask = np.isfinite(x) & np.isfinite(boe_local)
    if not np.any(mask):
        raise ValueError("No finite values available to plot local BOE profile.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], boe_local[mask], "-", lw=2, label="Local BOE(x)")
    plt.xlabel("x")
    plt.ylabel("BOE_local")
    plt.title("Local electric Bond number along jet surface")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
