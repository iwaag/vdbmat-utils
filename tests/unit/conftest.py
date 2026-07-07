"""Shared fixtures for the Phase 2 ops/fields unit tests."""

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from vdbmat_utils.core import build_material_label_volume, build_provenance

VolumeFactory = Callable[..., MaterialLabelVolume]

_DEFAULT_PALETTE = (
    (0, "air", "background"),
    (1, "resin_a", "material"),
    (2, "resin_b", "material"),
    (3, "resin_c", "material"),
)


def _make_volume(
    array: npt.NDArray[np.uint16] | Sequence[object],
    *,
    voxel_size_xyz_m: tuple[float, float, float] = (0.001, 0.002, 0.003),
    palette: tuple[tuple[int, str, str], ...] = _DEFAULT_PALETTE,
    local_to_world: tuple[tuple[float, ...], ...] | None = None,
) -> MaterialLabelVolume:
    return build_material_label_volume(
        material_id=np.asarray(array, dtype=np.uint16),
        voxel_size_xyz_m=voxel_size_xyz_m,
        palette=tuple(
            MaterialDefinition(
                material_id=material_id, name=name, role=MaterialRole(role)
            )
            for material_id, name, role in palette
        ),
        provenance=build_provenance(
            generator="vdbmat-utils.tests", generator_version="0.0.0"
        ),
        local_to_world=local_to_world,
    )


@pytest.fixture()
def make_volume() -> VolumeFactory:
    """Factory for small canonical volumes with a permissive test palette."""
    return _make_volume


def voxel_centers_world(volume: MaterialLabelVolume) -> dict[tuple[float, ...], int]:
    """Map each nonzero voxel's world-space center to its material id.

    Coordinates are rounded to 9 decimals so exact-arithmetic transforms
    compare equal as dict keys.
    """
    matrix = np.array(volume.geometry.local_to_world, dtype=np.float64)
    size_x, size_y, size_z = volume.geometry.voxel_size_xyz_m
    result: dict[tuple[float, ...], int] = {}
    for z, y, x in zip(*np.nonzero(volume.material_id), strict=True):
        local = np.array(
            [(x + 0.5) * size_x, (y + 0.5) * size_y, (z + 0.5) * size_z, 1.0]
        )
        world = matrix @ local
        key = tuple(round(float(value), 9) for value in world[:3])
        result[key] = int(volume.material_id[z, y, x])
    return result


@pytest.fixture()
def world_centers() -> Callable[[MaterialLabelVolume], dict[tuple[float, ...], int]]:
    return voxel_centers_world
