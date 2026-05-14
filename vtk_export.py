from __future__ import annotations

from pathlib import Path

import numpy as np

from interpolate_fields import build_interpolants, load_regular_grid, sanitize_name


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

    return export_all_fields_vtk_from_interpolants(
        x=x,
        y=y,
        fields=fields,
        interpolants=interpolants,
        output_vtk=output_vtk,
        nx_out=nx_out,
        ny_out=ny_out,
    )


def export_all_fields_vtk_from_interpolants(
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    interpolants: dict[str, object],
    output_vtk: str | Path,
    nx_out: int | None = None,
    ny_out: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Export all fields to VTK using precomputed interpolants.

    Component pairs (*.x, *.y) are exported as VECTORS.
    Remaining fields are exported as SCALARS.
    """

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
