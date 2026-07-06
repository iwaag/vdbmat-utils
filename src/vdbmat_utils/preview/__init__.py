"""Text and PGM previews of material label volumes.

These diagnostics deliberately depend on nothing beyond NumPy so they stay
available in the base install (no OpenVDB, no matplotlib): counting voxels per
material, rendering a single slice as ASCII art with an orientation legend,
and writing a slice as a grayscale PGM image.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialLabelVolume, MaterialRole

from vdbmat_utils.core.errors import VdbmatUtilsError

__all__ = [
    "PreviewError",
    "material_counts",
    "slice_ascii",
    "slice_pgm",
]

Axis = Literal["z", "y", "x"]

_AXES: tuple[Axis, ...] = ("z", "y", "x")

# Row/column axes of a slice perpendicular to each axis, in array order.
_SLICE_ROWS_COLS: dict[str, tuple[str, str]] = {
    "z": ("y", "x"),
    "y": ("z", "x"),
    "x": ("z", "y"),
}

_SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyz"


class PreviewError(VdbmatUtilsError):
    """A preview request is invalid (unknown axis, index out of range)."""


def material_counts(volume: MaterialLabelVolume) -> dict[int, int]:
    """Voxel count per palette material id, including zero-count entries."""
    ids, counts = np.unique(volume.material_id, return_counts=True)
    found = dict(zip(ids.tolist(), counts.tolist(), strict=True))
    return {
        material.material_id: found.get(material.material_id, 0)
        for material in volume.palette
    }


def _extract_slice(
    volume: MaterialLabelVolume, axis: str, index: int
) -> npt.NDArray[np.uint16]:
    if axis not in _AXES:
        raise PreviewError(f"unknown axis {axis!r}; expected one of z, y, x")
    axis_position = _AXES.index(axis)
    extent = volume.geometry.shape_zyx[axis_position]
    if not 0 <= index < extent:
        raise PreviewError(
            f"slice index {index} out of range for axis {axis} "
            f"(shape_zyx={tuple(volume.geometry.shape_zyx)}; "
            f"valid: 0..{extent - 1})"
        )
    plane: npt.NDArray[np.uint16] = np.take(
        volume.material_id, index, axis=axis_position
    )
    return plane


def slice_ascii(volume: MaterialLabelVolume, axis: Axis, index: int) -> str:
    """Render one slice as text: one character per voxel plus a legend line.

    Background-role materials render as ``.``; every other material id maps to
    ``0-9a-z`` cycling by id. The legend states which world axes the rows and
    columns follow, so accidental transposes are visible to the eye.
    """
    plane = _extract_slice(volume, axis, index)
    symbol_by_id = {
        material.material_id: (
            "." if material.role is MaterialRole.BACKGROUND
            else _SYMBOLS[material.material_id % len(_SYMBOLS)]
        )
        for material in volume.palette
    }
    row_axis, col_axis = _SLICE_ROWS_COLS[axis]
    legend = f"slice {axis}={index}  +{col_axis} →  +{row_axis} ↓"
    lines = [legend]
    lines.extend(
        "".join(symbol_by_id[int(value)] for value in row) for row in plane
    )
    return "\n".join(lines)


def slice_pgm(
    volume: MaterialLabelVolume, axis: Axis, index: int, path: str | Path
) -> Path:
    """Write one slice as a binary (P5) grayscale PGM and return its path.

    Each palette material id maps to a distinct gray level, spread evenly over
    0..255 by rank in ascending-id order, so the mapping is deterministic for
    a given palette.
    """
    plane = _extract_slice(volume, axis, index)
    ordered_ids = sorted(material.material_id for material in volume.palette)
    step = 255 // max(len(ordered_ids) - 1, 1)
    gray_by_id = {
        material_id: rank * step for rank, material_id in enumerate(ordered_ids)
    }
    lookup = np.zeros(max(ordered_ids) + 1, dtype=np.uint8)
    for material_id, gray in gray_by_id.items():
        lookup[material_id] = gray
    pixels = lookup[plane]
    rows, cols = pixels.shape
    output = Path(path)
    output.write_bytes(f"P5\n{cols} {rows}\n255\n".encode("ascii") + pixels.tobytes())
    return output
