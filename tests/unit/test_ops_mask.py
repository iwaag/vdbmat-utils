"""Unit tests for apply_mask (plan Step 0.2)."""

from collections.abc import Callable

import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError, apply_mask

VolumeFactory = Callable[..., MaterialLabelVolume]

_SOURCE = [[[1, 2, 0], [3, 0, 1]]]  # shape (1, 2, 3)
_MASK = [[[1, 0, 1], [1, 1, 0]]]


def test_keep_clears_outside_selection(make_volume: VolumeFactory) -> None:
    result = apply_mask(make_volume(_SOURCE), make_volume(_MASK), mode="keep")
    assert result.material_id.tolist() == [[[1, 0, 0], [3, 0, 0]]]


def test_clear_clears_the_selection(make_volume: VolumeFactory) -> None:
    result = apply_mask(make_volume(_SOURCE), make_volume(_MASK), mode="clear")
    assert result.material_id.tolist() == [[[0, 2, 0], [0, 0, 1]]]


def test_keep_with_explicit_fill(make_volume: VolumeFactory) -> None:
    result = apply_mask(
        make_volume(_SOURCE), make_volume(_MASK), mode="keep", fill_material_id=3
    )
    assert result.material_id.tolist() == [[[1, 3, 0], [3, 0, 3]]]


def test_geometry_mismatch_is_an_error(make_volume: VolumeFactory) -> None:
    small = make_volume([[[1]]])
    with pytest.raises(OpsError, match=r"mask\.shape_zyx .*crop/pad"):
        apply_mask(make_volume(_SOURCE), small, mode="keep")
    other_size = make_volume(_MASK, voxel_size_xyz_m=(0.001, 0.001, 0.001))
    with pytest.raises(OpsError, match=r"mask\.voxel_size_xyz_m"):
        apply_mask(make_volume(_SOURCE), other_size, mode="keep")


def test_unknown_mode_is_an_error(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"mode"):
        apply_mask(make_volume(_SOURCE), make_volume(_MASK), mode="invert")
