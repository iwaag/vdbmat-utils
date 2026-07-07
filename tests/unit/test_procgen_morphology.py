"""Plan D4: binary morphology; plan D7: connected components."""

import numpy as np
import pytest

from vdbmat_utils.procgen import (
    ProcgenError,
    close_mask,
    connected_components,
    dilate,
    erode,
    open_mask,
)


def _mask_with(coords: list[tuple[int, int, int]], shape: tuple[int, int, int]) -> (
    np.ndarray
):
    mask = np.zeros(shape, dtype=bool)
    for coordinate in coords:
        mask[coordinate] = True
    return mask


def test_dilate_single_voxel_6() -> None:
    mask = _mask_with([(2, 2, 2)], (5, 5, 5))
    grown = dilate(mask, radius_cells=1, connectivity=6)
    expected = _mask_with(
        [(2, 2, 2), (1, 2, 2), (3, 2, 2), (2, 1, 2), (2, 3, 2), (2, 2, 1), (2, 2, 3)],
        (5, 5, 5),
    )
    np.testing.assert_array_equal(grown, expected)


def test_dilate_single_voxel_26() -> None:
    mask = _mask_with([(2, 2, 2)], (5, 5, 5))
    grown = dilate(mask, radius_cells=1, connectivity=26)
    expected = np.zeros((5, 5, 5), dtype=bool)
    expected[1:4, 1:4, 1:4] = True
    np.testing.assert_array_equal(grown, expected)


def test_erode_is_dual_and_boundary_conservative() -> None:
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:4, 1:5, 1:6] = True
    eroded = erode(mask, radius_cells=1, connectivity=6)
    expected = np.zeros((5, 6, 7), dtype=bool)
    expected[2:3, 2:4, 2:5] = True
    np.testing.assert_array_equal(eroded, expected)
    # A slab touching the domain wall erodes from the wall side too.
    slab = np.zeros((4, 4, 4), dtype=bool)
    slab[:, :, 0:2] = True
    assert not erode(slab, radius_cells=1, connectivity=6)[:, :, 0].any()


def test_dilate_does_not_wrap_around() -> None:
    mask = _mask_with([(0, 0, 0)], (3, 3, 3))
    grown = dilate(mask, radius_cells=1, connectivity=6)
    assert not grown[2, 0, 0] and not grown[0, 2, 0] and not grown[0, 0, 2]


def test_open_removes_thin_feature_close_fills_gap() -> None:
    mask = np.zeros((5, 7, 7), dtype=bool)
    mask[2, 1:6, 1:6] = True  # a one-voxel-thick sheet
    assert not open_mask(mask, radius_cells=1, connectivity=6).any()
    gap = np.zeros((5, 5, 7), dtype=bool)
    gap[1:4, 1:4, 1:3] = True
    gap[1:4, 1:4, 4:6] = True  # one-voxel gap at x=3
    closed = close_mask(gap, radius_cells=1, connectivity=6)
    assert closed[2, 2, 3]
    assert not closed[0, 0, 0]


def test_radius_zero_is_identity() -> None:
    mask = _mask_with([(1, 2, 3), (0, 0, 0)], (4, 4, 4))
    np.testing.assert_array_equal(dilate(mask, radius_cells=0), mask)
    np.testing.assert_array_equal(erode(mask, radius_cells=0), mask)


def test_morphology_validation() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    with pytest.raises(ProcgenError):
        dilate(np.zeros((3, 3, 3), dtype=np.uint16), radius_cells=1)
    with pytest.raises(ProcgenError):
        dilate(np.zeros((3, 3), dtype=bool), radius_cells=1)
    with pytest.raises(ProcgenError):
        erode(mask, radius_cells=-1)
    with pytest.raises(ProcgenError):
        erode(mask, radius_cells=1, connectivity=18)


def test_components_two_blobs_with_sizes() -> None:
    mask = np.zeros((4, 5, 6), dtype=bool)
    mask[0:2, 0:2, 0:2] = True  # 8 voxels, first in C order
    mask[3, 4, 3:6] = True  # 3 voxels
    result = connected_components(mask)
    assert result.count == 2
    np.testing.assert_array_equal(result.sizes, [8, 3])
    assert result.component_ids[0, 0, 0] == 1
    assert result.component_ids[3, 4, 4] == 2
    assert (result.component_ids[~mask] == 0).all()


def test_components_diagonal_touch_is_separate_under_6() -> None:
    mask = _mask_with([(1, 1, 1), (2, 2, 2), (1, 1, 2)], (4, 4, 4))
    # (1,1,2) face-connects (1,1,1); (2,2,2) touches only diagonally.
    result = connected_components(mask)
    assert result.count == 2
    np.testing.assert_array_equal(result.sizes, [2, 1])


def test_components_empty_and_full() -> None:
    empty = connected_components(np.zeros((3, 3, 3), dtype=bool))
    assert empty.count == 0 and empty.sizes.size == 0
    assert (empty.component_ids == 0).all()
    full = connected_components(np.ones((3, 4, 5), dtype=bool))
    assert full.count == 1
    np.testing.assert_array_equal(full.sizes, [60])


def test_components_u_shape_merges_across_scan_order() -> None:
    # Two arms meet only at the bottom — exercises the union step.
    mask = np.zeros((4, 4, 5), dtype=bool)
    mask[0:4, 1, 0] = True
    mask[0:4, 1, 4] = True
    mask[3, 1, 0:5] = True
    result = connected_components(mask)
    assert result.count == 1


def test_components_validation() -> None:
    with pytest.raises(ProcgenError):
        connected_components(np.zeros((3, 3), dtype=bool))
    with pytest.raises(ProcgenError):
        connected_components(np.zeros((3, 3, 3), dtype=np.uint16))
