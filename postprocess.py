#!/usr/bin/env python3
"""
Postprocess Basilisk outputs.

Current features:
1) Build a 1D interpolant for jet radius R(x) from jet_radius_eq.dat and plot radius vs x.
2) Export interpolated velocity field (u_x, u_y) to a ParaView-readable VTK file.
3) Export all available fields to a ParaView-readable VTK file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from interpolate_fields import build_interpolants, load_regular_grid, sanitize_name


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
    x_unique, _ = np.unique(x, return_index=True)
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


def _extract_velocity_interpolants(interpolants: dict[str, object]) -> tuple[object, object]:
    """Return (interp_u_x, interp_u_y) from interpolant dictionary."""
    if "interp_u_x" in interpolants and "interp_u_y" in interpolants:
        return interpolants["interp_u_x"], interpolants["interp_u_y"]

    # Fallback if columns are named ux/uy in fields.dat
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


def _vtk_safe_name(name: str) -> str:
    """Convert a field name to a VTK-friendly array name."""
    out = sanitize_name(name)
    if not out:
        out = "field"
    return out


def _sample_interpolant_on_grid(
    interp_obj: object,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Evaluate interpolant on structured (xq, yq) grid and return array (nx, ny)."""
    nx = xq.size
    ny = yq.size
    xx, yy = np.meshgrid(xq, yq, indexing="ij")
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    return np.asarray(interp_obj(pts), dtype=float).reshape(nx, ny)


def export_velocity_field_vtk(
    fields_path: str | Path,
    output_vtk: str | Path,
    nx_out: int | None = None,
    ny_out: int | None = None,
) -> tuple[int, int]:
    """
    Export velocity field to VTK legacy format for ParaView.

    Velocity components are evaluated from interpolate_fields.py interpolants:
    - interp_u_x
    - interp_u_y
    """
    x, y, fields = load_regular_grid(fields_path)
    interpolants = build_interpolants(x, y, fields, bounds_error=False, fill_value=np.nan)
    interp_u_x, interp_u_y = _extract_velocity_interpolants(interpolants)

    nx = x.size if nx_out is None else max(2, int(nx_out))
    ny = y.size if ny_out is None else max(2, int(ny_out))

    xq = np.linspace(float(x.min()), float(x.max()), nx)
    yq = np.linspace(float(y.min()), float(y.max()), ny)
    xx, yy = np.meshgrid(xq, yq, indexing="ij")
    pts = np.column_stack((xx.ravel(), yy.ravel()))

    ux = np.asarray(interp_u_x(pts), dtype=float).reshape(nx, ny)
    uy = np.asarray(interp_u_y(pts), dtype=float).reshape(nx, ny)

    if np.isnan(ux).any() or np.isnan(uy).any():
        raise ValueError(
            "NaNs found in interpolated velocity field. Check sampling domain and input data."
        )

    dx = (xq[-1] - xq[0]) / (nx - 1)
    dy = (yq[-1] - yq[0]) / (ny - 1)

    out = Path(output_vtk)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# vtk DataFile Version 3.0\n")
        fp.write("Basilisk velocity field\n")
        fp.write("ASCII\n")
        fp.write("DATASET STRUCTURED_POINTS\n")
        fp.write(f"DIMENSIONS {nx} {ny} 1\n")
        fp.write(f"ORIGIN {xq[0]:.15e} {yq[0]:.15e} 0.0\n")
        fp.write(f"SPACING {dx:.15e} {dy:.15e} 1.0\n")
        fp.write(f"POINT_DATA {nx * ny}\n")
        fp.write("VECTORS velocity float\n")
        for j in range(ny):
            for i in range(nx):
                fp.write(f"{ux[i, j]:.8e} {uy[i, j]:.8e} 0.00000000e+00\n")

    return nx, ny


def export_all_fields_vtk(
    fields_path: str | Path,
    output_vtk: str | Path,
    nx_out: int | None = None,
    ny_out: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Export all fields in fields.dat to one VTK legacy file for ParaView.

    Component pairs (*.x, *.y) are exported as VECTORS.
    Remaining fields are exported as SCALARS.
    """
    x, y, fields = load_regular_grid(fields_path)
    interpolants = build_interpolants(x, y, fields, bounds_error=False, fill_value=np.nan)

    nx = x.size if nx_out is None else max(2, int(nx_out))
    ny = y.size if ny_out is None else max(2, int(ny_out))
    xq = np.linspace(float(x.min()), float(x.max()), nx)
    yq = np.linspace(float(y.min()), float(y.max()), ny)
    dx = (xq[-1] - xq[0]) / (nx - 1)
    dy = (yq[-1] - yq[0]) / (ny - 1)

    sampled: dict[str, np.ndarray] = {}
    for field_name in fields:
        interp_name = f"interp_{sanitize_name(field_name)}"
        if interp_name not in interpolants:
            raise KeyError(
                f"Interpolant '{interp_name}' not found for field '{field_name}'."
            )
        arr = _sample_interpolant_on_grid(interpolants[interp_name], xq, yq)
        if np.isnan(arr).any():
            raise ValueError(
                f"NaNs found while sampling field '{field_name}'. "
                "Check sampling domain and input data."
            )
        sampled[field_name] = arr

    field_names = list(fields.keys())
    used_in_vectors: set[str] = set()
    vectors: list[tuple[str, str, str]] = []
    scalars: list[str] = []

    for name in field_names:
        if name in used_in_vectors:
            continue

        paired = False
        for sx, sy in ((".x", ".y"), ("_x", "_y")):
            if name.endswith(sx):
                base = name[: -len(sx)]
                name_y = base + sy
                if name_y in sampled:
                    vectors.append((base, name, name_y))
                    used_in_vectors.add(name)
                    used_in_vectors.add(name_y)
                    paired = True
                break

        if not paired and name not in used_in_vectors:
            scalars.append(name)

    out = Path(output_vtk)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# vtk DataFile Version 3.0\n")
        fp.write("Basilisk all fields\n")
        fp.write("ASCII\n")
        fp.write("DATASET STRUCTURED_POINTS\n")
        fp.write(f"DIMENSIONS {nx} {ny} 1\n")
        fp.write(f"ORIGIN {xq[0]:.15e} {yq[0]:.15e} 0.0\n")
        fp.write(f"SPACING {dx:.15e} {dy:.15e} 1.0\n")
        fp.write(f"POINT_DATA {nx * ny}\n")

        for scalar_name in scalars:
            scalar_vtk_name = _vtk_safe_name(scalar_name)
            arr = sampled[scalar_name]
            fp.write(f"SCALARS {scalar_vtk_name} float 1\n")
            fp.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    fp.write(f"{arr[i, j]:.8e}\n")

        for vec_base, vec_x_name, vec_y_name in vectors:
            vec_vtk_name = _vtk_safe_name(vec_base)
            arr_x = sampled[vec_x_name]
            arr_y = sampled[vec_y_name]
            fp.write(f"VECTORS {vec_vtk_name} float\n")
            for j in range(ny):
                for i in range(nx):
                    fp.write(
                        f"{arr_x[i, j]:.8e} {arr_y[i, j]:.8e} 0.00000000e+00\n"
                    )

    return nx, ny, len(scalars), len(vectors)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess Basilisk data products.")

    parser.add_argument(
        "--input",
        default="jet_radius_eq.dat",
        help="Input file with columns x radius (default: jet_radius_eq.dat).",
    )
    parser.add_argument(
        "--output",
        default="jet_radius_vs_x.png",
        help="Output figure path for radius plot (default: jet_radius_vs_x.png).",
    )
    parser.add_argument(
        "--n-plot",
        type=int,
        default=1000,
        help="Number of x points for plotting interpolated radius (default: 1000).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive radius plot window in addition to saving.",
    )

    parser.add_argument(
        "--export-velocity-vtk",
        action="store_true",
        help="Export velocity field to VTK for ParaView.",
    )
    parser.add_argument(
        "--fields-input",
        default="readLateralJet_params/fields.dat",
        help="Input fields.dat used to build velocity interpolants.",
    )
    parser.add_argument(
        "--vtk-output",
        default="velocity_field.vtk",
        help="Output VTK filename (default: velocity_field.vtk).",
    )
    parser.add_argument(
        "--vtk-nx",
        type=int,
        default=None,
        help="Number of x samples for VTK export (default: use input grid nx).",
    )
    parser.add_argument(
        "--vtk-ny",
        type=int,
        default=None,
        help="Number of y samples for VTK export (default: use input grid ny).",
    )
    parser.add_argument(
        "--compute-avg-velocity",
        action="store_true",
        help="Compute axisymmetric average axial velocity U_avg(x) for r <= R(x).",
    )
    parser.add_argument(
        "--avg-nr",
        type=int,
        default=256,
        help="Number of radial samples used in average-velocity integration (default: 256).",
    )
    parser.add_argument(
        "--avg-vel-output",
        default="avg_velocity_vs_x.dat",
        help="Output data file for average velocity profile (default: avg_velocity_vs_x.dat).",
    )
    parser.add_argument(
        "--avg-vel-plot",
        default="avg_velocity_vs_x.png",
        help="Output figure for average velocity profile (default: avg_velocity_vs_x.png).",
    )
    parser.add_argument(
        "--compute-local-boe",
        action="store_true",
        help="Compute local BOE(x) = R(x) * max_r(|ERy|)^2 * BOE_global.",
    )
    parser.add_argument(
        "--boe-global",
        type=float,
        default=None,
        help="Global BOE parameter used in local BOE computation.",
    )
    parser.add_argument(
        "--boe-nr",
        type=int,
        default=256,
        help="Number of radial samples used to find max |ERy| (default: 256).",
    )
    parser.add_argument(
        "--boe-output",
        default="boe_local_vs_x.dat",
        help="Output data file for local BOE profile (default: boe_local_vs_x.dat).",
    )
    parser.add_argument(
        "--boe-plot",
        default="boe_local_vs_x.png",
        help="Output figure for local BOE profile (default: boe_local_vs_x.png).",
    )
    parser.add_argument(
        "--export-all-fields-vtk",
        action="store_true",
        help="Export all fields to one VTK file readable by ParaView.",
    )
    parser.add_argument(
        "--all-vtk-output",
        default="all_fields.vtk",
        help="Output VTK filename for all fields (default: all_fields.vtk).",
    )
    parser.add_argument(
        "--all-vtk-nx",
        type=int,
        default=None,
        help="Number of x samples for all-fields VTK export (default: input nx).",
    )
    parser.add_argument(
        "--all-vtk-ny",
        type=int,
        default=None,
        help="Number of y samples for all-fields VTK export (default: input ny).",
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

    print(f"Loaded {x.size} radius points from: {args.input}")
    print(f"x range: [{x.min():.9g}, {x.max():.9g}]")
    print(f"radius range: [{r.min():.9g}, {r.max():.9g}]")
    print(f"Saved radius plot to: {args.output}")

    if args.export_velocity_vtk:
        nx, ny = export_velocity_field_vtk(
            fields_path=args.fields_input,
            output_vtk=args.vtk_output,
            nx_out=args.vtk_nx,
            ny_out=args.vtk_ny,
        )
        print(f"Saved velocity VTK to: {args.vtk_output} (nx={nx}, ny={ny})")

    if args.export_all_fields_vtk:
        nx_all, ny_all, n_scalars, n_vectors = export_all_fields_vtk(
            fields_path=args.fields_input,
            output_vtk=args.all_vtk_output,
            nx_out=args.all_vtk_nx,
            ny_out=args.all_vtk_ny,
        )
        print(
            "Saved all-fields VTK to: "
            f"{args.all_vtk_output} (nx={nx_all}, ny={ny_all}, "
            f"scalars={n_scalars}, vectors={n_vectors})"
        )

    if args.compute_avg_velocity:
        x_field, y_field, fields = load_regular_grid(args.fields_input)
        interpolants = build_interpolants(
            x_field, y_field, fields, bounds_error=False, fill_value=np.nan
        )
        interp_u_x, _ = _extract_velocity_interpolants(interpolants)

        x_avg, u_avg = compute_axisymmetric_average_velocity_profile(
            x_profile=x_field,
            radius_interp=r_interp,
            interp_u_x=interp_u_x,
            nr=args.avg_nr,
        )
        save_average_velocity_profile(x_avg, u_avg, args.avg_vel_output)
        plot_average_velocity_profile(x_avg, u_avg, args.avg_vel_plot, show=args.show)
        nvalid = int(np.count_nonzero(np.isfinite(u_avg)))
        print(
            "Saved average velocity profile to: "
            f"{args.avg_vel_output} and {args.avg_vel_plot} (points={nvalid})"
        )

    if args.compute_local_boe:
        if args.boe_global is None:
            raise ValueError(
                "--boe-global is required when --compute-local-boe is enabled."
            )

        x_field, y_field, fields = load_regular_grid(args.fields_input)
        interpolants = build_interpolants(
            x_field, y_field, fields, bounds_error=False, fill_value=np.nan
        )
        interp_er_y = _extract_ery_interpolant(interpolants)

        x_boe, boe_local = compute_local_boe_profile(
            x_profile=x_field,
            radius_interp=r_interp,
            interp_er_y=interp_er_y,
            boe_global=float(args.boe_global),
            nr=args.boe_nr,
        )
        save_local_boe_profile(x_boe, boe_local, float(args.boe_global), args.boe_output)
        plot_local_boe_profile(x_boe, boe_local, args.boe_plot, show=args.show)
        nvalid_boe = int(np.count_nonzero(np.isfinite(boe_local)))
        print(
            "Saved local BOE profile to: "
            f"{args.boe_output} and {args.boe_plot} (points={nvalid_boe})"
        )


if __name__ == "__main__":
    main()
