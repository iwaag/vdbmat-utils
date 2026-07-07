"""Nearest-neighbor label resampling (plan D5).

Interpolation is forbidden by construction: every output cell copies exactly
one source label, chosen by sampling the source grid at the output cell's
center (fixed formula, integer index arithmetic). Integer up/downsampling
factors are exact; non-integer factors are allowed but alias-prone (see
``docs/volume-ops.md``). Material conservation is *not* claimed.
"""

import numpy as np
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError

from ._common import rebuild


def resample(
    volume: MaterialLabelVolume,
    *,
    factor_zyx: tuple[float, float, float] | None = None,
    voxel_size_xyz_m: tuple[float, float, float] | None = None,
) -> MaterialLabelVolume:
    """Resample to per-axis cell-count factors or a new voxel size.

    Exactly one of ``factor_zyx`` (output cells per source cell, ordered
    z, y, x) and ``voxel_size_xyz_m`` (new voxel size, ordered x, y, z) must
    be given. Each output cell takes the label of the source cell containing
    its center; the local origin is unchanged, so ``local_to_world`` is
    untouched and the manifest voxel size updates to match.
    """
    if (factor_zyx is None) == (voxel_size_xyz_m is None):
        raise OpsError(
            "resample: exactly one of factor_zyx and voxel_size_xyz_m required"
        )
    old_size = volume.geometry.voxel_size_xyz_m
    if factor_zyx is not None:
        if len(factor_zyx) != 3:
            raise OpsError(
                f"resample: factor_zyx must have 3 components, "
                f"got {len(factor_zyx)}"
            )
        factors = tuple(float(f) for f in factor_zyx)
        if any(not (f > 0) for f in factors):
            raise OpsError("resample: factors and voxel sizes must be positive")
        new_size = (
            old_size[0] / factors[2],
            old_size[1] / factors[1],
            old_size[2] / factors[0],
        )
    else:
        assert voxel_size_xyz_m is not None
        if len(voxel_size_xyz_m) != 3:
            raise OpsError(
                f"resample: voxel_size_xyz_m must have 3 components, "
                f"got {len(voxel_size_xyz_m)}"
            )
        new_size = (
            float(voxel_size_xyz_m[0]),
            float(voxel_size_xyz_m[1]),
            float(voxel_size_xyz_m[2]),
        )
        if any(not (s > 0) for s in new_size):
            raise OpsError("resample: factors and voxel sizes must be positive")
        factors = (
            old_size[2] / new_size[2],
            old_size[1] / new_size[1],
            old_size[0] / new_size[0],
        )
    shape = volume.geometry.shape_zyx
    new_shape = tuple(max(1, round(extent * f)) for extent, f in zip(
        shape, factors, strict=True
    ))
    indices = []
    for axis in range(3):
        # Output cell center i + 0.5 lands in source cell floor((i+0.5)/f).
        centers = (np.arange(new_shape[axis], dtype=np.float64) + 0.5) / factors[
            axis
        ]
        indices.append(
            np.clip(np.floor(centers).astype(np.int64), 0, shape[axis] - 1)
        )
    array = volume.material_id[np.ix_(indices[0], indices[1], indices[2])]
    return rebuild(
        volume, np.ascontiguousarray(array), voxel_size_xyz_m=new_size
    )
