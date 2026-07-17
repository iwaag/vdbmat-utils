"""Generation body: grid derivation and cell-centre classification for a
transparent base block containing an A x B x C grid of opaque cube or
sphere inclusions.

Sampling is cell-centre, matching the mesh voxelizer: a cell is classified
by its centre point against each primitive's closed-boundary region
(``<=`` is inside). Because the config guarantees non-negative gaps, the
per-axis primitive windows never overlap (they may only touch when
``gap_m == 0``), so the classification below is a pure per-cell predicate
with no ordering or double-paint concerns — see ``docs/primitive-arrays.md``
and the plan's risk notes.
"""

from math import ceil

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from vdbmat_utils.core import build_material_label_volume, build_provenance

from . import PrimitiveArrayError
from .types import BUILTIN_MATERIAL_IDS, PrimitiveArrayConfig

GENERATOR = "vdbmat-utils.primitives.array"
GENERATOR_VERSION = "0.1.0"

_AIR_MATERIAL_ID = 0
# A span landing within this fraction of a cell of an integer is treated as
# exactly that many cells (matches voxelize-mesh's domain-snap epsilon), so
# float round-off does not add a spurious padded cell.
_DOMAIN_SNAP_EPS = 1e-6


def generate_primitive_array(config: PrimitiveArrayConfig) -> MaterialLabelVolume:
    """Build the material-label volume described by ``config``."""
    voxel_size = _as_float_triplet(config.voxel_size_xyz_m)
    counts = _as_int_triplet(config.counts_xyz)
    size = float(config.primitive_size_m)
    gap = float(config.gap_m)
    margin = float(config.margin_m)

    shape_zyx = _grid_shape(voxel_size, counts, size, gap, margin, config)
    inside = _inside_mask(
        voxel_size, counts, size, gap, margin, shape_zyx, config.primitive
    )

    base_id = BUILTIN_MATERIAL_IDS[config.base_material_name]
    inclusion_id = BUILTIN_MATERIAL_IDS[config.inclusion_material_name]
    label = np.where(inside, inclusion_id, base_id).astype(np.uint16)

    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=(),
        notes=(
            f"{config.primitive} primitive array counts_xyz={counts}; "
            f"base={config.base_material_name}; "
            f"inclusion={config.inclusion_material_name}"
        ),
    )
    return build_material_label_volume(
        material_id=label,
        voxel_size_xyz_m=voxel_size,
        palette=_palette(config),
        provenance=provenance,
    )


def _palette(config: PrimitiveArrayConfig) -> tuple[MaterialDefinition, ...]:
    return (
        MaterialDefinition(_AIR_MATERIAL_ID, "air", MaterialRole.BACKGROUND),
        MaterialDefinition(
            BUILTIN_MATERIAL_IDS[config.base_material_name],
            config.base_material_name,
            MaterialRole.MATERIAL,
        ),
        MaterialDefinition(
            BUILTIN_MATERIAL_IDS[config.inclusion_material_name],
            config.inclusion_material_name,
            MaterialRole.MATERIAL,
        ),
    )


def _grid_shape(
    voxel_size: tuple[float, float, float],
    counts: tuple[int, int, int],
    size: float,
    gap: float,
    margin: float,
    config: PrimitiveArrayConfig,
) -> tuple[int, int, int]:
    cells_xyz: list[int] = []
    for count, voxel in zip(counts, voxel_size, strict=True):
        extent = 2.0 * margin + count * size + (count - 1) * gap
        cells_xyz.append(max(1, ceil(extent / voxel - _DOMAIN_SNAP_EPS)))

    if any(cells > config.max_axis_cells for cells in cells_xyz):
        raise PrimitiveArrayError(
            "voxel_size_xyz_m",
            f"grid extent {tuple(cells_xyz)} exceeds the axis bound of "
            f"{config.max_axis_cells} cells; use a coarser voxel size",
        )
    total = cells_xyz[0] * cells_xyz[1] * cells_xyz[2]
    if total > config.max_total_cells:
        raise PrimitiveArrayError(
            "voxel_size_xyz_m",
            f"grid has {total} cells, exceeding the bound of "
            f"{config.max_total_cells}; use a coarser voxel size",
        )
    return (cells_xyz[2], cells_xyz[1], cells_xyz[0])


def _primitive_centers(
    count: int, size: float, gap: float, margin: float
) -> npt.NDArray[np.float64]:
    step = size + gap
    return margin + size / 2.0 + np.arange(count, dtype=np.float64) * step


def _inside_axis_cube(
    coords: npt.NDArray[np.float64], centers: npt.NDArray[np.float64], half: float
) -> npt.NDArray[np.bool_]:
    diff = np.abs(coords[:, None] - centers[None, :])
    return np.any(diff <= half, axis=1)


def _nearest_axis_sq_dist(
    coords: npt.NDArray[np.float64], centers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    diff = coords[:, None] - centers[None, :]
    return np.min(diff * diff, axis=1)


def _inside_mask(
    voxel_size: tuple[float, float, float],
    counts: tuple[int, int, int],
    size: float,
    gap: float,
    margin: float,
    shape_zyx: tuple[int, int, int],
    primitive: str,
) -> npt.NDArray[np.bool_]:
    nz, ny, nx = shape_zyx
    sx, sy, sz = voxel_size
    cx, cy, cz = counts

    x = (np.arange(nx, dtype=np.float64) + 0.5) * sx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * sy
    z = (np.arange(nz, dtype=np.float64) + 0.5) * sz

    centers_x = _primitive_centers(cx, size, gap, margin)
    centers_y = _primitive_centers(cy, size, gap, margin)
    centers_z = _primitive_centers(cz, size, gap, margin)

    half = size / 2.0

    if primitive == "cube":
        inside_x = _inside_axis_cube(x, centers_x, half)
        inside_y = _inside_axis_cube(y, centers_y, half)
        inside_z = _inside_axis_cube(z, centers_z, half)
        return (
            inside_z[:, None, None]
            & inside_y[None, :, None]
            & inside_x[None, None, :]
        )

    if primitive == "sphere":
        # The sum of squared per-axis distances is separable, so the
        # global-minimum-distance primitive combination is exactly the one
        # built from each axis's own nearest centre; no cross-axis
        # candidate search is needed.
        dist2_x = _nearest_axis_sq_dist(x, centers_x)
        dist2_y = _nearest_axis_sq_dist(y, centers_y)
        dist2_z = _nearest_axis_sq_dist(z, centers_z)
        total = dist2_z[:, None, None] + dist2_y[None, :, None] + dist2_x[None, None, :]
        return total <= half * half

    raise PrimitiveArrayError("primitive", f"unsupported primitive {primitive!r}")


def _as_float_triplet(value: object) -> tuple[float, float, float]:
    x, y, z = value  # type: ignore[misc]
    return (float(x), float(y), float(z))


def _as_int_triplet(value: object) -> tuple[int, int, int]:
    x, y, z = value  # type: ignore[misc]
    return (int(x), int(y), int(z))
