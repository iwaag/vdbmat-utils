"""Crop and pad operations (plan D5)."""

import numpy as np
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError

from ._common import BACKGROUND_ID, rebuild, translated_local_to_world


def _require_triplet(value: tuple[int, int, int], *, name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise OpsError(f"{name} must have 3 components (z, y, x), got {len(value)}")
    return (int(value[0]), int(value[1]), int(value[2]))


def crop(
    volume: MaterialLabelVolume,
    *,
    min_zyx: tuple[int, int, int],
    max_zyx: tuple[int, int, int],
) -> MaterialLabelVolume:
    """Return the half-open index box ``[min_zyx, max_zyx)`` of ``volume``.

    The box must lie entirely within the array — there is no implicit
    clamping. ``local_to_world`` is recomposed with the translation of the
    cropped origin so surviving voxels keep their world positions.
    """
    lower = _require_triplet(min_zyx, name="min_zyx")
    upper = _require_triplet(max_zyx, name="max_zyx")
    shape = volume.geometry.shape_zyx
    for axis, (low, high, extent) in enumerate(
        zip(lower, upper, shape, strict=True)
    ):
        if not 0 <= low < high <= extent:
            raise OpsError(
                f"crop box axis {'zyx'[axis]}: need 0 <= min < max <= {extent}, "
                f"got [{low}, {high})"
            )
    array = np.ascontiguousarray(
        volume.material_id[
            lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]
        ]
    )
    size_x, size_y, size_z = volume.geometry.voxel_size_xyz_m
    offset_xyz_m = (lower[2] * size_x, lower[1] * size_y, lower[0] * size_z)
    return rebuild(
        volume, array, local_to_world=translated_local_to_world(volume, offset_xyz_m)
    )


def pad(
    volume: MaterialLabelVolume,
    *,
    before_zyx: tuple[int, int, int],
    after_zyx: tuple[int, int, int],
    fill_material_id: int | None = None,
) -> MaterialLabelVolume:
    """Grow ``volume`` by whole cells filled with ``fill_material_id``.

    The fill defaults to the palette's unique background material and must
    exist in the palette. ``local_to_world`` is recomposed so pre-existing
    voxels keep their world positions.
    """
    before = _require_triplet(before_zyx, name="before_zyx")
    after = _require_triplet(after_zyx, name="after_zyx")
    if any(value < 0 for value in before + after):
        raise OpsError("pad amounts must be non-negative")
    if fill_material_id is None:
        fill = BACKGROUND_ID
    else:
        fill = int(fill_material_id)
        if fill not in volume.palette.material_ids:
            raise OpsError(f"pad fill_material_id {fill} is not in the palette")
    array = np.pad(
        volume.material_id,
        tuple(zip(before, after, strict=True)),
        mode="constant",
        constant_values=np.uint16(fill),
    )
    size_x, size_y, size_z = volume.geometry.voxel_size_xyz_m
    offset_xyz_m = (
        -before[2] * size_x,
        -before[1] * size_y,
        -before[0] * size_z,
    )
    return rebuild(
        volume, array, local_to_world=translated_local_to_world(volume, offset_xyz_m)
    )
