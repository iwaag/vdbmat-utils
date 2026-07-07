"""Orientation (exact 90° data movement) and placement (metadata) ops.

Plan D5 keeps these deliberately separate: ``orient`` moves array data in
exact flip/rot90 steps, by default composing the inverse motion into
``local_to_world`` so world geometry is preserved; ``place`` touches only the
transform metadata and never resamples.
"""

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError

from ._common import rebuild

_AXES = ("x", "y", "z")
# Canonical array order is z, y, x; coordinate components are ordered x, y, z.
_ARRAY_AXIS = {"z": 0, "y": 1, "x": 2}
_COORD_AXIS = {"x": 0, "y": 1, "z": 2}

OrientStep = tuple[str, ...]


def _require_axis(value: str, *, step_index: int) -> str:
    if value not in _AXES:
        raise OpsError(
            f"orient step {step_index}: unknown axis {value!r}, expected one of "
            f"{', '.join(_AXES)}"
        )
    return value


def orient(
    volume: MaterialLabelVolume,
    *,
    steps: tuple[OrientStep, ...],
    update_transform: bool = True,
) -> MaterialLabelVolume:
    """Apply an ordered sequence of exact flip / 90°-rotation steps.

    Each step is ``("flip", axis)`` or ``("rot90", from_axis, to_axis)``
    (rotation by +90° taking ``from_axis`` toward ``to_axis``). With
    ``update_transform=True`` (default) the inverse motion is composed into
    ``local_to_world`` so every voxel keeps its world position; a net mirror
    (odd number of flips) cannot be expressed as a rigid transform and is
    rejected. With ``update_transform=False`` the transform is kept as-is and
    the volume is genuinely re-oriented in world space.
    """
    array: npt.NDArray[np.uint16] = volume.material_id
    sizes = list(volume.geometry.voxel_size_xyz_m)  # indexed x, y, z
    # matrix maps homogeneous local coordinates of the *new* array to local
    # coordinates of the original array.
    matrix = np.eye(4, dtype=np.float64)

    for index, step in enumerate(steps):
        if not step:
            raise OpsError(f"orient step {index}: empty step")
        kind = step[0]
        if kind == "flip":
            if len(step) != 2:
                raise OpsError(f"orient step {index}: flip takes exactly one axis")
            axis = _require_axis(step[1], step_index=index)
            array_axis = _ARRAY_AXIS[axis]
            coord = _COORD_AXIS[axis]
            extent = array.shape[array_axis] * sizes[coord]
            array = np.flip(array, axis=array_axis)
            step_matrix = np.eye(4, dtype=np.float64)
            step_matrix[coord, coord] = -1.0
            step_matrix[coord, 3] = extent
            matrix = matrix @ step_matrix
        elif kind == "rot90":
            if len(step) != 3 or step[1] == step[2]:
                raise OpsError(
                    f"orient step {index}: rot90 takes two distinct axes"
                )
            source = _require_axis(step[1], step_index=index)
            target = _require_axis(step[2], step_index=index)
            axis_u, axis_v = _ARRAY_AXIS[source], _ARRAY_AXIS[target]
            coord_u, coord_v = _COORD_AXIS[source], _COORD_AXIS[target]
            extent_v = array.shape[axis_v] * sizes[coord_v]
            # new[i_u, i_v] = old[i_v, n_v - 1 - i_u]
            array = np.flip(np.swapaxes(array, axis_u, axis_v), axis=axis_u)
            step_matrix = np.eye(4, dtype=np.float64)
            step_matrix[coord_u, coord_u] = 0.0
            step_matrix[coord_v, coord_v] = 0.0
            step_matrix[coord_u, coord_v] = 1.0
            step_matrix[coord_v, coord_u] = -1.0
            step_matrix[coord_v, 3] = extent_v
            matrix = matrix @ step_matrix
            sizes[coord_u], sizes[coord_v] = sizes[coord_v], sizes[coord_u]
        else:
            raise OpsError(
                f"orient step {index}: unknown step kind {kind!r}, "
                "expected 'flip' or 'rot90'"
            )

    voxel_size = (sizes[0], sizes[1], sizes[2])
    if not update_transform:
        return rebuild(
            volume, np.ascontiguousarray(array), voxel_size_xyz_m=voxel_size
        )
    if not np.isclose(float(np.linalg.det(matrix[0:3, 0:3])), 1.0):
        raise OpsError(
            "orient: a net mirror (odd number of flips) is not a rigid "
            "transform; use update_transform=False for a real re-orientation"
        )
    composed = np.array(volume.geometry.local_to_world, dtype=np.float64) @ matrix
    return rebuild(
        volume,
        np.ascontiguousarray(array),
        voxel_size_xyz_m=voxel_size,
        local_to_world=tuple(tuple(float(v) for v in row) for row in composed),
    )


def place(
    volume: MaterialLabelVolume,
    *,
    local_to_world: tuple[tuple[float, ...], ...],
    compose_with_existing: bool = False,
) -> MaterialLabelVolume:
    """Replace (or left-compose onto) ``local_to_world``; metadata only.

    Never moves or resamples array data. Rigidity of the resulting transform
    is validated by the canonical geometry type.
    """
    if compose_with_existing:
        composed = np.array(local_to_world, dtype=np.float64) @ np.array(
            volume.geometry.local_to_world, dtype=np.float64
        )
        target = tuple(tuple(float(v) for v in row) for row in composed)
    else:
        target = tuple(tuple(float(v) for v in row) for row in local_to_world)
    return rebuild(volume, volume.material_id, local_to_world=target)
