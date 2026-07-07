"""Shared helpers for label-safe volume operations."""

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialLabelVolume, MaterialPalette

from vdbmat_utils.core import build_material_label_volume


def rebuild(
    volume: MaterialLabelVolume,
    material_id: npt.NDArray[np.uint16],
    *,
    voxel_size_xyz_m: tuple[float, float, float] | None = None,
    local_to_world: tuple[tuple[float, ...], ...] | None = None,
    palette: MaterialPalette | None = None,
) -> MaterialLabelVolume:
    """Return a validated copy of ``volume`` with the given parts replaced."""
    return build_material_label_volume(
        material_id=material_id,
        voxel_size_xyz_m=(
            volume.geometry.voxel_size_xyz_m
            if voxel_size_xyz_m is None
            else voxel_size_xyz_m
        ),
        palette=volume.palette if palette is None else palette,
        provenance=volume.provenance,
        local_to_world=(
            volume.geometry.local_to_world
            if local_to_world is None
            else local_to_world
        ),
    )


def translated_local_to_world(
    volume: MaterialLabelVolume, offset_xyz_m: tuple[float, float, float]
) -> tuple[tuple[float, ...], ...]:
    """Compose ``local_to_world @ translation(offset)`` (offset in local metres).

    Used by crop/pad so that the world positions of surviving voxels are
    unchanged: a point at ``p`` in the new local frame sits at ``p + offset``
    in the old local frame.
    """
    matrix = np.array(volume.geometry.local_to_world, dtype=np.float64)
    translation = np.eye(4, dtype=np.float64)
    translation[0:3, 3] = offset_xyz_m
    composed = matrix @ translation
    return tuple(tuple(float(value) for value in row) for row in composed)


def require_matching_geometry(
    base: MaterialLabelVolume, other: MaterialLabelVolume, *, other_name: str
) -> None:
    """Raise ``OpsError`` unless ``other`` shares ``base``'s exact geometry.

    Binary operations require identical shape, voxel size, and
    ``local_to_world`` (plan D5); the error names the mismatching field and
    the operations that can reconcile it.
    """
    from vdbmat_utils.ops import OpsError

    pairs = (
        ("shape_zyx", base.geometry.shape_zyx, other.geometry.shape_zyx),
        (
            "voxel_size_xyz_m",
            base.geometry.voxel_size_xyz_m,
            other.geometry.voxel_size_xyz_m,
        ),
        (
            "local_to_world",
            base.geometry.local_to_world,
            other.geometry.local_to_world,
        ),
    )
    for field, expected, actual in pairs:
        if expected != actual:
            raise OpsError(
                f"{other_name}.{field} {actual} does not match base {expected}; "
                "align the volumes first (crop/pad/resample/place)"
            )


# The canonical palette contract (vdbmat.core.MaterialPalette) guarantees that
# material_id 0 exists, has role "background", and is the only background.
BACKGROUND_ID = 0
