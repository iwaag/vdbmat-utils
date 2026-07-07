"""Unit tests for crop and pad (plan Step 0.2)."""

from collections.abc import Callable

import numpy as np
import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError, crop, pad

VolumeFactory = Callable[..., MaterialLabelVolume]
WorldCenters = Callable[[MaterialLabelVolume], dict[tuple[float, ...], int]]

_SOURCE = [
    [[0, 1, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [3, 3, 0, 0], [0, 0, 0, 1]],
]  # shape (z=2, y=3, x=4), asymmetric on every axis


def test_crop_extracts_expected_box(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    result = crop(volume, min_zyx=(0, 0, 1), max_zyx=(2, 2, 3))
    assert result.material_id.tolist() == [
        [[1, 0], [2, 2]],
        [[0, 0], [3, 0]],
    ]


def test_crop_preserves_world_positions(
    make_volume: VolumeFactory, world_centers: WorldCenters
) -> None:
    volume = make_volume(_SOURCE)
    result = crop(volume, min_zyx=(1, 1, 0), max_zyx=(2, 3, 4))
    before = world_centers(volume)
    after = world_centers(result)
    assert after  # box was chosen to keep some foreground
    assert set(after).issubset(set(before))
    assert all(before[key] == value for key, value in after.items())


def test_crop_out_of_range_is_an_error(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    with pytest.raises(OpsError, match=r"crop box axis x"):
        crop(volume, min_zyx=(0, 0, 0), max_zyx=(2, 3, 5))
    with pytest.raises(OpsError, match=r"crop box axis z"):
        crop(volume, min_zyx=(1, 0, 0), max_zyx=(1, 3, 4))


def test_pad_fills_with_background_and_keeps_world_positions(
    make_volume: VolumeFactory, world_centers: WorldCenters
) -> None:
    volume = make_volume(_SOURCE)
    result = pad(volume, before_zyx=(1, 0, 2), after_zyx=(0, 1, 0))
    assert result.material_id.shape == (3, 4, 6)
    assert result.material_id[0].max() == 0  # new leading z slab is background
    assert world_centers(result) == world_centers(volume)


def test_pad_explicit_fill_material(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    result = pad(
        volume, before_zyx=(0, 0, 1), after_zyx=(0, 0, 0), fill_material_id=2
    )
    assert result.material_id[:, :, 0].tolist() == [[2, 2, 2], [2, 2, 2]]


def test_pad_unknown_fill_material_is_an_error(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"fill_material_id 9"):
        pad(
            make_volume(_SOURCE),
            before_zyx=(0, 0, 1),
            after_zyx=(0, 0, 0),
            fill_material_id=9,
        )


def test_pad_negative_amount_is_an_error(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"non-negative"):
        pad(make_volume(_SOURCE), before_zyx=(0, 0, -1), after_zyx=(0, 0, 0))


def test_crop_pad_round_trip_restores_array(make_volume: VolumeFactory) -> None:
    volume = make_volume(_SOURCE)
    padded = pad(volume, before_zyx=(1, 2, 3), after_zyx=(3, 2, 1))
    restored = crop(padded, min_zyx=(1, 2, 3), max_zyx=(3, 5, 7))
    assert np.array_equal(restored.material_id, volume.material_id)
    assert restored.geometry.local_to_world == volume.geometry.local_to_world
