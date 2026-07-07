"""Unit tests for orient and place (plan Step 0.2).

World-preservation checks compare the set of world-space foreground voxel
centers (with material ids) before and after — this catches any sign or
axis-order error in the composed transform, on an array that is asymmetric
along every axis.
"""

from collections.abc import Callable

import numpy as np
import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError, orient, place

VolumeFactory = Callable[..., MaterialLabelVolume]
WorldCenters = Callable[[MaterialLabelVolume], dict[tuple[float, ...], int]]

_SOURCE = [
    [[0, 1, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [3, 3, 0, 0], [0, 0, 0, 1]],
]  # shape (z=2, y=3, x=4)


@pytest.mark.parametrize(
    "steps",
    [
        (("rot90", "x", "y"),),
        (("rot90", "y", "z"),),
        (("rot90", "z", "x"),),
        (("rot90", "x", "y"), ("rot90", "x", "y")),
        (("flip", "x"), ("flip", "y")),
        (("flip", "z"), ("rot90", "x", "y"), ("flip", "x")),
    ],
)
def test_orient_preserves_world_positions(
    steps: tuple[tuple[str, ...], ...],
    make_volume: VolumeFactory,
    world_centers: WorldCenters,
) -> None:
    volume = make_volume(_SOURCE)
    result = orient(volume, steps=steps)
    assert world_centers(result) == world_centers(volume)


def test_rot90_moves_data_and_swaps_voxel_sizes(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)  # voxel size (x=0.001, y=0.002, z=0.003)
    result = orient(volume, steps=(("rot90", "x", "y"),))
    assert result.material_id.shape == (2, 4, 3)  # y and x extents swapped
    assert result.geometry.voxel_size_xyz_m == (0.002, 0.001, 0.003)
    expected = np.flip(np.swapaxes(np.asarray(_SOURCE), 2, 1), axis=2)
    assert np.array_equal(result.material_id, expected)


def test_orient_double_flip_round_trip(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    result = orient(volume, steps=(("flip", "x"), ("flip", "x")))
    assert np.array_equal(result.material_id, volume.material_id)
    assert result.geometry.local_to_world == volume.geometry.local_to_world


def test_orient_net_mirror_is_rejected(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"mirror"):
        orient(make_volume(_SOURCE), steps=(("flip", "z"),))


def test_orient_without_transform_update_reorients(
    make_volume: VolumeFactory, world_centers: WorldCenters
) -> None:
    volume = make_volume(_SOURCE)
    result = orient(volume, steps=(("flip", "x"),), update_transform=False)
    assert result.geometry.local_to_world == volume.geometry.local_to_world
    assert np.array_equal(result.material_id, np.flip(volume.material_id, axis=2))
    assert world_centers(result) != world_centers(volume)


def test_orient_rejects_bad_steps(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    with pytest.raises(OpsError, match=r"unknown axis"):
        orient(volume, steps=(("flip", "w"),))
    with pytest.raises(OpsError, match=r"two distinct axes"):
        orient(volume, steps=(("rot90", "x", "x"),))
    with pytest.raises(OpsError, match=r"unknown step kind"):
        orient(volume, steps=(("shear", "x", "y"),))


def test_place_replaces_transform_metadata_only(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    target = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, 0.02),
        (0.0, 0.0, 1.0, 0.03),
        (0.0, 0.0, 0.0, 1.0),
    )
    result = place(volume, local_to_world=target)
    assert result.geometry.local_to_world == target
    assert np.array_equal(result.material_id, volume.material_id)


def test_place_compose_left_multiplies(make_volume: VolumeFactory) -> None:
    volume = make_volume(
        _SOURCE,
        local_to_world=(
            (1.0, 0.0, 0.0, 0.005),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    shift = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.007),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    result = place(volume, local_to_world=shift, compose_with_existing=True)
    assert result.geometry.local_to_world[0][3] == 0.005
    assert result.geometry.local_to_world[1][3] == 0.007
