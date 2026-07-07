"""Exact Euclidean distance transform (Felzenszwalb-Huttenlocher).

In-repo, NumPy-only implementation of the separable lower-envelope-of-
parabolas algorithm (Felzenszwalb & Huttenlocher, *Distance Transforms of
Sampled Functions*, ToC 2012), so the base install needs no scipy (plan D2).
Exact for cell-center distances, deterministic (fixed loop order), O(n) per
row. Row loops are plain Python — acceptable at Phase 2 sizes; acceleration
is a Phase 5 concern.
"""

import numpy as np
import numpy.typing as npt

from . import FieldError


def _dt_1d(row: npt.NDArray[np.float64], spacing: float) -> npt.NDArray[np.float64]:
    """1-D squared-distance transform: ``d(i) = min_j row[j] + (s*(i-j))^2``."""
    n = row.shape[0]
    finite = np.isfinite(row)
    if not finite.any():
        return row.copy()
    result = np.empty(n, dtype=np.float64)
    square = spacing * spacing
    vertices = np.empty(n, dtype=np.int64)  # parabola apex indices
    bounds = np.empty(n + 1, dtype=np.float64)  # envelope segment boundaries
    indices = np.flatnonzero(finite)
    first = int(indices[0])
    vertices[0] = first
    bounds[0] = -np.inf
    bounds[1] = np.inf
    count = 0
    for j in map(int, indices[1:]):
        while True:
            k = int(vertices[count])
            # Intersection of parabolas rooted at k and j (j > k).
            crossing = (
                (row[j] - row[k]) / square + float(j * j - k * k)
            ) / float(2 * (j - k))
            if crossing <= bounds[count]:
                count -= 1
                continue
            count += 1
            vertices[count] = j
            bounds[count] = crossing
            bounds[count + 1] = np.inf
            break
    segment = 0
    for i in range(n):
        while bounds[segment + 1] < i:
            segment += 1
        k = int(vertices[segment])
        result[i] = square * float((i - k) * (i - k)) + row[k]
    return result


def squared_edt(
    mask: npt.NDArray[np.bool_], spacing: tuple[float, ...]
) -> npt.NDArray[np.float64]:
    """Squared Euclidean distance from each cell center to the nearest
    ``True`` cell center, in the metric given by per-axis ``spacing``.

    Cells of an all-``False`` mask are at distance ``+inf``. ``spacing`` is
    ordered like the array axes (canonical volumes: z, y, x).
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise FieldError(f"mask dtype must be bool, got {array.dtype}")
    if len(spacing) != array.ndim:
        raise FieldError(
            f"spacing must have one component per axis: got {len(spacing)} "
            f"for a {array.ndim}-D mask"
        )
    if any(not (float(s) > 0) for s in spacing):
        raise FieldError("spacing components must be positive")
    distance = np.where(array, 0.0, np.inf)
    for axis, axis_spacing in enumerate(spacing):
        moved = np.moveaxis(distance, axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        for row_index in range(flat.shape[0]):
            flat[row_index] = _dt_1d(flat[row_index], float(axis_spacing))
        distance = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return distance


def signed_distance(
    mask: npt.NDArray[np.bool_], spacing: tuple[float, ...]
) -> npt.NDArray[np.float64]:
    """Signed cell-center distance: negative inside ``mask``, positive outside.

    Inside cells carry minus the distance to the nearest outside cell center;
    outside cells carry the distance to the nearest inside cell center. An
    all-``False`` mask is ``+inf`` everywhere (the morphing "absent label"
    rule); an all-``True`` mask is ``-inf`` everywhere.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise FieldError(f"mask dtype must be bool, got {array.dtype}")
    outside = np.sqrt(squared_edt(array, spacing))
    inside = np.sqrt(squared_edt(~array, spacing))
    return np.where(array, -inside, outside)
