"""Binary morphology on boolean masks (plan D4).

Exact iterated one-step erosion/dilation with a 6- or 26-neighbourhood
structuring element, implemented with padded boolean shifts — no scipy, no
floats, and boolean arrays only (never label arrays; plan D6).
"""

import numpy as np
import numpy.typing as npt

_OFFSETS_6 = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
]
_OFFSETS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
]


def _validate(
    mask: npt.NDArray[np.bool_], radius_cells: int, connectivity: int
) -> list[tuple[int, int, int]]:
    from . import ProcgenError

    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype != np.bool_:
        raise ProcgenError(
            f"mask must be a 3-D bool array, got {array.ndim}-D {array.dtype}"
        )
    if isinstance(radius_cells, bool) or not isinstance(radius_cells, int):
        raise ProcgenError(f"radius_cells must be an integer, got {radius_cells!r}")
    if radius_cells < 0:
        raise ProcgenError(f"radius_cells must be non-negative, got {radius_cells}")
    if connectivity == 6:
        return _OFFSETS_6
    if connectivity == 26:
        return _OFFSETS_26
    raise ProcgenError(f"connectivity must be 6 or 26, got {connectivity!r}")


def _shift(
    mask: npt.NDArray[np.bool_], offset: tuple[int, int, int], *, fill: bool
) -> npt.NDArray[np.bool_]:
    """Shift a mask by ``offset``, filling vacated cells with ``fill``."""
    result = mask
    for axis, step in enumerate(offset):
        if step == 0:
            continue
        result = np.roll(result, step, axis=axis)
        edge = [slice(None)] * 3
        edge[axis] = slice(0, step) if step > 0 else slice(step, None)
        result = result.copy()
        result[tuple(edge)] = fill
    return result


def dilate(
    mask: npt.NDArray[np.bool_], *, radius_cells: int, connectivity: int = 6
) -> npt.NDArray[np.bool_]:
    """Grow the mask by ``radius_cells`` one-step dilations.

    Cells outside the domain are treated as ``False`` (features never grow in
    from the boundary).
    """
    offsets = _validate(mask, radius_cells, connectivity)
    result = np.asarray(mask).copy()
    for _ in range(radius_cells):
        step = result.copy()
        for offset in offsets:
            step |= _shift(result, offset, fill=False)
        result = step
    return result


def erode(
    mask: npt.NDArray[np.bool_], *, radius_cells: int, connectivity: int = 6
) -> npt.NDArray[np.bool_]:
    """Shrink the mask by ``radius_cells`` one-step erosions.

    Cells outside the domain are treated as ``False``, so foreground touching
    the boundary erodes from the boundary side too (the conservative rule for
    printability: a slab against the wall is not infinitely thick).
    """
    offsets = _validate(mask, radius_cells, connectivity)
    result = np.asarray(mask).copy()
    for _ in range(radius_cells):
        step = result.copy()
        for offset in offsets:
            step &= _shift(result, offset, fill=False)
        result = step
    return result


def open_mask(
    mask: npt.NDArray[np.bool_], *, radius_cells: int, connectivity: int = 6
) -> npt.NDArray[np.bool_]:
    """Erosion then dilation: removes features thinner than the radius."""
    eroded = erode(mask, radius_cells=radius_cells, connectivity=connectivity)
    return dilate(eroded, radius_cells=radius_cells, connectivity=connectivity)


def close_mask(
    mask: npt.NDArray[np.bool_], *, radius_cells: int, connectivity: int = 6
) -> npt.NDArray[np.bool_]:
    """Dilation then erosion: fills gaps narrower than the radius."""
    dilated = dilate(mask, radius_cells=radius_cells, connectivity=connectivity)
    return erode(dilated, radius_cells=radius_cells, connectivity=connectivity)
