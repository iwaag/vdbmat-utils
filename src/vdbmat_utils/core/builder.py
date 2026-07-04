"""Construction helper for canonical material-label volumes."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from vdbmat.core import (
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialPalette,
    Provenance,
)
from vdbmat.core.transforms import Matrix4

from .compat import require_compatible_volume_schema
from .errors import GeometryError, PaletteError


def _matrix4(rows: Sequence[Sequence[float]]) -> Matrix4:
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise GeometryError("local_to_world must be a 4x4 matrix")
    r0, r1, r2, r3 = (
        tuple(float(value) for value in row) for row in rows
    )
    return (r0, r1, r2, r3)  # type: ignore[return-value]


def build_material_label_volume(
    *,
    material_id: npt.NDArray[np.uint16],
    voxel_size_xyz_m: Sequence[float],
    palette: MaterialPalette | Sequence[MaterialDefinition],
    provenance: Provenance,
    local_to_world: Sequence[Sequence[float]] | None = None,
) -> MaterialLabelVolume:
    """Assemble a validated ``vdbmat.core.MaterialLabelVolume``.

    ``material_id`` must be a 3-D ``uint16`` array in canonical ``z, y, x``
    axis order; its shape defines the grid shape. Detailed value validation
    (palette references, dtype, transform rigidity) is delegated to the
    canonical types, whose ``VolumeValidationError`` passes through unwrapped.
    """
    require_compatible_volume_schema()

    array = np.asarray(material_id)
    if array.ndim != 3:
        raise GeometryError(
            f"material_id must be a 3-D array in z, y, x order, got {array.ndim}-D"
        )
    if array.dtype != np.uint16:
        raise GeometryError(f"material_id dtype must be uint16, got {array.dtype}")

    if not isinstance(palette, MaterialPalette):
        definitions = tuple(palette)
        if not definitions:
            raise PaletteError("palette must declare at least one material")
        palette = MaterialPalette.from_sequence(definitions)

    shape_zyx = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if len(voxel_size_xyz_m) != 3:
        raise GeometryError(
            f"voxel_size_xyz_m must have 3 components, got {len(voxel_size_xyz_m)}"
        )
    voxel_size = (
        float(voxel_size_xyz_m[0]),
        float(voxel_size_xyz_m[1]),
        float(voxel_size_xyz_m[2]),
    )
    if local_to_world is None:
        geometry = GridGeometry(shape_zyx=shape_zyx, voxel_size_xyz_m=voxel_size)
    else:
        geometry = GridGeometry(
            shape_zyx=shape_zyx,
            voxel_size_xyz_m=voxel_size,
            local_to_world=_matrix4(local_to_world),
        )

    return MaterialLabelVolume(
        geometry=geometry,
        palette=palette,
        provenance=provenance,
        material_id=array,
    )
