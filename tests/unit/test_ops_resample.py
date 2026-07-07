"""Unit tests for nearest-neighbor label resampling (plan D5, Step 3.2)."""

from collections.abc import Callable

import numpy as np
import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError, resample

VolumeFactory = Callable[..., MaterialLabelVolume]


def test_integer_upsample_is_exact_repetition(make_volume: VolumeFactory) -> None:
    volume = make_volume([[[1, 2], [3, 0]]])
    result = resample(volume, factor_zyx=(1, 2, 2))
    expected = np.array(
        [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(result.material_id, expected)


def test_integer_downsample_takes_containing_cell(
    make_volume: VolumeFactory,
) -> None:
    array = np.zeros((2, 4, 4), dtype=np.uint16)
    array[:, 0:2, 0:2] = 1
    array[:, 2:4, 2:4] = 2
    volume = make_volume(array)
    result = resample(volume, factor_zyx=(1, 0.5, 0.5))
    expected = np.array([[[1, 0], [0, 2]], [[1, 0], [0, 2]]], dtype=np.uint16)
    np.testing.assert_array_equal(result.material_id, expected)


def test_voxel_size_metadata_updates(make_volume: VolumeFactory) -> None:
    volume = make_volume(np.zeros((2, 2, 2), dtype=np.uint16))
    assert volume.geometry.voxel_size_xyz_m == (0.001, 0.002, 0.003)
    result = resample(volume, factor_zyx=(2, 2, 2))
    assert result.geometry.shape_zyx == (4, 4, 4)
    assert result.geometry.voxel_size_xyz_m == (0.0005, 0.001, 0.0015)
    assert result.geometry.local_to_world == volume.geometry.local_to_world


def test_new_voxel_size_form_matches_factor_form(
    make_volume: VolumeFactory,
) -> None:
    array = np.arange(8, dtype=np.uint16).reshape(2, 2, 2) % 4
    volume = make_volume(array)
    by_factor = resample(volume, factor_zyx=(2, 2, 2))
    by_size = resample(volume, voxel_size_xyz_m=(0.0005, 0.001, 0.0015))
    np.testing.assert_array_equal(by_factor.material_id, by_size.material_id)
    assert by_factor.geometry == by_size.geometry


def test_output_labels_are_a_subset_of_input(make_volume: VolumeFactory) -> None:
    # Nearest-neighbor only: no new (mixed) ids can ever appear.
    array = np.array([[[0, 1, 2], [3, 0, 1]]], dtype=np.uint16)
    volume = make_volume(array)
    result = resample(volume, factor_zyx=(3, 1.5, 0.7))
    assert set(np.unique(result.material_id)) <= set(np.unique(array))


def test_exactly_one_parameter_required(make_volume: VolumeFactory) -> None:
    volume = make_volume(np.zeros((1, 1, 1), dtype=np.uint16))
    with pytest.raises(OpsError, match="exactly one"):
        resample(volume)
    with pytest.raises(OpsError, match="exactly one"):
        resample(volume, factor_zyx=(1, 1, 1), voxel_size_xyz_m=(1, 1, 1))
    with pytest.raises(OpsError, match="positive"):
        resample(volume, factor_zyx=(0, 1, 1))
