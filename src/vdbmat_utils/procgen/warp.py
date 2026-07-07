"""Domain warping (plan D4).

Warping displaces *coordinates before evaluation* — "evaluate the primitive
at ``coords + amplitude * offsets``" — never resampling an already evaluated
field. There is therefore no interpolation anywhere and no label risk; the
warped result is exact and deterministic. Offset fields are ordinary
``ScalarField`` data (typically fBm instances with their own stream ids).
"""

import numpy as np
import numpy.typing as npt

from vdbmat_utils.fields import ScalarField

from .domain import FormationDomain


def warped_coordinates(
    domain: FormationDomain,
    *,
    offsets_xyz: tuple[ScalarField, ScalarField, ScalarField],
    amplitude_m: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Voxel-centre coordinates displaced by ``amplitude_m * offsets``.

    Returns full-shape ``(x, y, z)`` metre-coordinate arrays ready to feed
    into any ``*_at`` primitive. Each offset field must lie on the domain's
    grid (shape and voxel size must match). ``amplitude_m`` may be zero (no
    warp) but not negative — flip the offset fields instead.
    """
    if not (float(amplitude_m) >= 0):
        from . import ProcgenError

        raise ProcgenError(f"amplitude_m must be non-negative, got {amplitude_m!r}")
    for axis_name, offset in zip("xyz", offsets_xyz, strict=True):
        if offset.values.shape != domain.shape_zyx:
            from . import ProcgenError

            raise ProcgenError(
                f"offset field {axis_name} shape {offset.values.shape} does not "
                f"match domain shape {domain.shape_zyx}"
            )
        if offset.voxel_size_xyz_m != domain.voxel_size_xyz_m:
            from . import ProcgenError

            raise ProcgenError(
                f"offset field {axis_name} voxel size {offset.voxel_size_xyz_m} "
                f"does not match domain voxel size {domain.voxel_size_xyz_m}"
            )
    x, y, z = domain.coordinates_xyz_m()
    amplitude = float(amplitude_m)
    offset_x, offset_y, offset_z = offsets_xyz
    warped_x = x + amplitude * offset_x.values
    warped_y = y + amplitude * offset_y.values
    warped_z = z + amplitude * offset_z.values
    return warped_x, warped_y, warped_z
