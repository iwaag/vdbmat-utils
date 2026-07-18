"""Pure output-grid derivation and nearest-neighbour sampling.

Printer pitch is non-integer relative to typical source voxel sizes
(``25.4/600`` mm etc.), so the output grid is derived in physical space —
never by an integer-ratio resample — and each output cell is filled by the
nearest source voxel to its physical centre. No interpolation, averaging, or
majority voting is performed anywhere in this module. This module never
imports PNG or file-IO code so it can be unit-tested without the ``image``
extra and reused as the shared reference point for a future round-trip
contract with ``convert-image-stack``.
"""

import dataclasses
import math

import numpy as np
import numpy.typing as npt

from vdbmat_utils.printer import PrintSlicesError
from vdbmat_utils.printer.types import PrintSlicesConfig

_EPSILON = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class OutputGrid:
    """Derived printer-pitch output grid, guards already applied."""

    n_slices: int
    height: int  #: printer Y axis (``dpi_y``) pixel count — PNG rows
    width: int  #: printer X axis (``dpi_x``) pixel count — PNG columns
    pitch_x_m: float
    pitch_y_m: float
    pitch_z_m: float


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingPlan:
    """Precomputed nearest-neighbour source indices for one export."""

    grid: OutputGrid
    #: length ``n_slices``; source z index for each output slice
    z_source_index: npt.NDArray[np.int64]
    #: length ``width``; source index along the axis assigned to printer X
    printer_x_source_index: npt.NDArray[np.int64]
    #: length ``height``; source index along the axis assigned to printer Y
    printer_y_source_index: npt.NDArray[np.int64]
    #: True when ``printer_x_axis == "y"`` (source x/y are swapped)
    transposed: bool


def derive_output_grid(
    shape_zyx: tuple[int, int, int],
    voxel_size_xyz_m: tuple[float, float, float],
    config: PrintSlicesConfig,
) -> OutputGrid:
    """Derive the printer-pitch output grid and apply its size guards."""
    nz, ny, nx = shape_zyx
    sx, sy, sz = voxel_size_xyz_m
    pitch_x = 0.0254 / config.dpi_x
    pitch_y = 0.0254 / config.dpi_y
    pitch_z = config.layer_thickness_m

    extent_for_axis = {"x": nx * sx, "y": ny * sy}
    width = _derive_cell_count(extent_for_axis[config.printer_x_axis], pitch_x)
    height = _derive_cell_count(extent_for_axis[config.printer_y_axis], pitch_y)
    n_slices = _derive_cell_count(nz * sz, pitch_z)

    if n_slices < config.min_slices:
        raise PrintSlicesError(
            "grid",
            f"derived slice count {n_slices} is below min_slices "
            f"({config.min_slices}); use a thinner layer_thickness_m or "
            "a taller input to reach the minimum",
        )

    total_pixels = n_slices * height * width
    if total_pixels > config.max_total_pixels:
        raise PrintSlicesError(
            "grid",
            f"derived output has {total_pixels} total pixels, exceeds "
            f"max_total_pixels ({config.max_total_pixels}); use a coarser "
            "input voxel size or crop the input to reduce it",
        )

    return OutputGrid(
        n_slices=n_slices,
        height=height,
        width=width,
        pitch_x_m=pitch_x,
        pitch_y_m=pitch_y,
        pitch_z_m=pitch_z,
    )


def build_sampling_plan(
    shape_zyx: tuple[int, int, int],
    voxel_size_xyz_m: tuple[float, float, float],
    config: PrintSlicesConfig,
) -> SamplingPlan:
    """Derive the output grid and precompute its nearest-neighbour indices."""
    grid = derive_output_grid(shape_zyx, voxel_size_xyz_m, config)
    nz, ny, nx = shape_zyx
    sx, sy, sz = voxel_size_xyz_m
    src_cells_for_axis = {"x": nx, "y": ny}
    src_size_for_axis = {"x": sx, "y": sy}

    printer_x_index = _nearest_source_indices(
        grid.width,
        grid.pitch_x_m,
        src_size_for_axis[config.printer_x_axis],
        src_cells_for_axis[config.printer_x_axis],
    )
    printer_y_index = _nearest_source_indices(
        grid.height,
        grid.pitch_y_m,
        src_size_for_axis[config.printer_y_axis],
        src_cells_for_axis[config.printer_y_axis],
    )
    z_index = _nearest_source_indices(grid.n_slices, grid.pitch_z_m, sz, nz)

    if config.flip_x:
        printer_x_index = printer_x_index[::-1]
    if config.flip_y:
        printer_y_index = printer_y_index[::-1]
    if config.flip_z:
        z_index = z_index[::-1]

    return SamplingPlan(
        grid=grid,
        z_source_index=z_index,
        printer_x_source_index=printer_x_index,
        printer_y_source_index=printer_y_index,
        transposed=config.printer_x_axis == "y",
    )


def sample_slice(
    material_id: npt.NDArray[np.uint16], plan: SamplingPlan, output_slice_index: int
) -> npt.NDArray[np.uint16]:
    """Return the ``(height, width)`` material-id array for one output slice.

    Purely index-based nearest-neighbour extraction: no interpolation.
    """
    z_source = int(plan.z_source_index[output_slice_index])
    plane = material_id[z_source]  # shape (ny, nx)
    if not plan.transposed:
        rows = np.take(plane, plan.printer_y_source_index, axis=0)
        return np.take(rows, plan.printer_x_source_index, axis=1)
    cols_as_rows = np.take(plane, plan.printer_x_source_index, axis=0)
    swapped = np.take(cols_as_rows, plan.printer_y_source_index, axis=1)
    return np.ascontiguousarray(swapped.T)


def _derive_cell_count(extent_m: float, pitch_m: float) -> int:
    return math.ceil(extent_m / pitch_m - _EPSILON)


def _nearest_source_indices(
    out_cells: int, pitch_m: float, src_voxel_size_m: float, src_cells: int
) -> npt.NDArray[np.int64]:
    centers = (np.arange(out_cells, dtype=np.float64) + 0.5) * pitch_m
    index = np.floor(centers / src_voxel_size_m).astype(np.int64)
    return np.clip(index, 0, src_cells - 1)
