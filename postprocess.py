#!/usr/bin/env python3
"""
Postprocess Basilisk outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from interpolate_fields import build_interpolants, load_regular_grid
from radius import build_radius_interpolant, load_radius_data, plot_radius_profile
from radial_lines import export_fields_along_radial_lines
from vtk_export import export_all_fields_vtk_from_interpolants


def _resolve_output_path(output_dir: Path, filename: str) -> Path:
    """Return full output path under output_dir, creating directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _load_radius_interpolant(radius_input: str | Path):
    x, r = load_radius_data(radius_input)
    return x, r, build_radius_interpolant(x, r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess Basilisk data products.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where all outputs will be saved (default: current directory).",
    )

    parser.add_argument(
        "--radius-input",
        required=True,
        help="Input file with columns x radius.",
    )
    parser.add_argument(
        "--plot-radius",
        action="store_true",
        help="Plot jet radius profile from the radius input file.",
    )
    parser.add_argument(
        "--radius-output",
        default="jet_radius_vs_x.png",
        help="Output figure filename for radius plot (default: jet_radius_vs_x.png).",
    )
    parser.add_argument(
        "--radius-n-plot",
        type=int,
        default=1000,
        help="Number of x points for plotting interpolated radius (default: 1000).",
    )

    parser.add_argument(
        "--fields-input",
        "--input",
        dest="fields_input",
        required=True,
        help="Input fields.dat used to build interpolants.",
    )
    parser.add_argument(
        "--plot-radial-lines",
        action="store_true",
        help="Plot all fields along radial lines.",
    )
    parser.add_argument(
        "--radial-lines-x-min",
        type=float,
        default=None,
        help="Minimum x value for radial lines (default: minimum from fields).",
    )
    parser.add_argument(
        "--radial-lines-x-max",
        type=float,
        default=None,
        help="Maximum x value for radial lines (default: maximum from fields).",
    )
    parser.add_argument(
        "--radial-lines-n",
        type=int,
        default=5,
        help="Number of radial lines to plot (default: 5).",
    )
    parser.add_argument(
        "--radial-lines-nr",
        type=int,
        default=256,
        help="Number of radial samples per line (default: 256).",
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

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot windows in addition to saving.",
    )

    args = parser.parse_args()

    if not any(
        (
            args.plot_radius,
            args.export_all_fields_vtk,
            args.plot_radial_lines,
        )
    ):
        parser.error(
            "No actions requested. Use --plot-radius, --export-all-fields-vtk, or --plot-radial-lines."
        )

    output_dir = Path(args.output_dir)

    x_radius = None
    r_radius = None
    r_interp = None
    if args.plot_radius:
        x_radius, r_radius, r_interp = _load_radius_interpolant(args.radius_input)

    fields_x = None
    fields_y = None
    fields = None
    interpolants = None
    if args.export_all_fields_vtk or args.plot_radial_lines:
        fields_x, fields_y, fields = load_regular_grid(args.fields_input)
        interpolants = build_interpolants(
            fields_x, fields_y, fields, bounds_error=False, fill_value=np.nan
        )

    if args.plot_radius and x_radius is not None and r_radius is not None and r_interp:
        radius_plot_path = _resolve_output_path(output_dir, args.radius_output)
        plot_radius_profile(
            x=x_radius,
            r=r_radius,
            r_interp=r_interp,
            output=radius_plot_path,
            n_plot=args.radius_n_plot,
            show=args.show,
        )
        print(f"Loaded {x_radius.size} radius points from: {args.radius_input}")
        print(f"x range: [{x_radius.min():.9g}, {x_radius.max():.9g}]")
        print(f"radius range: [{r_radius.min():.9g}, {r_radius.max():.9g}]")
        print(f"Saved radius plot to: {radius_plot_path}")

    if args.export_all_fields_vtk and fields is not None and interpolants is not None:
        all_vtk_path = _resolve_output_path(output_dir, args.all_vtk_output)
        nx_all, ny_all, n_scalars, n_vectors = export_all_fields_vtk_from_interpolants(
            x=fields_x,
            y=fields_y,
            fields=fields,
            interpolants=interpolants,
            output_vtk=all_vtk_path,
            nx_out=args.all_vtk_nx,
            ny_out=args.all_vtk_ny,
        )
        print(
            "Saved all-fields VTK to: "
            f"{all_vtk_path} (nx={nx_all}, ny={ny_all}, "
            f"scalars={n_scalars}, vectors={n_vectors})"
        )

    if args.plot_radial_lines and fields_x is not None and interpolants is not None:
        x_min = args.radial_lines_x_min
        x_max = args.radial_lines_x_max
        if x_min is None:
            x_min = float(fields_x.min())
        if x_max is None:
            x_max = float(fields_x.max())

        export_fields_along_radial_lines(
            x_profile=fields_x,
            interpolants=interpolants,
            x_min=x_min,
            x_max=x_max,
            n_lines=args.radial_lines_n,
            nr=args.radial_lines_nr,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
