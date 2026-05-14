#!/usr/bin/env python3
"""
Convert a regular-grid Basilisk fields file to legacy VTK format for ParaView.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vtk_export import export_all_fields_vtk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data from output_field/fields.dat to ParaView VTK format."
    )
    parser.add_argument(
        "--fields-input",
        required=True,
        help="File with the data from output_field.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Folder for the output (default: current directory).",
    )
    parser.add_argument(
        "--vtk-output",
        required=True,
        help="Name of the output VTK file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_vtk = output_dir / args.vtk_output

    nx, ny, n_scalars, n_vectors = export_all_fields_vtk(
        fields_path=args.fields_input,
        output_vtk=output_vtk,
    )

    print(
        "Saved VTK to: "
        f"{output_vtk} (nx={nx}, ny={ny}, scalars={n_scalars}, vectors={n_vectors})"
    )


if __name__ == "__main__":
    main()
