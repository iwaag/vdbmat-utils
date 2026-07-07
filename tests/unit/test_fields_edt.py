"""Unit tests for the in-repo exact EDT (plan Step 1.2).

Property-style checks compare against a brute-force O(n²) distance over all
pairs of cells, on small random masks with fixed seeds.
"""

import numpy as np
import numpy.typing as npt
import pytest

from vdbmat_utils.core import rng_from_seed
from vdbmat_utils.fields import FieldError, signed_distance, squared_edt


def _brute_force(
    mask: npt.NDArray[np.bool_], spacing: tuple[float, ...]
) -> npt.NDArray[np.float64]:
    targets = np.argwhere(mask).astype(np.float64) * np.asarray(spacing)
    result = np.full(mask.shape, np.inf, dtype=np.float64)
    if targets.size == 0:
        return result
    for index in np.ndindex(mask.shape):
        point = np.asarray(index, dtype=np.float64) * np.asarray(spacing)
        result[index] = np.min(np.sum((targets - point) ** 2, axis=1))
    return result


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize(
    ("shape", "spacing"),
    [
        ((9, 7), (1.0, 1.0)),
        ((6, 8), (0.5, 2.0)),  # anisotropic
        ((4, 5, 6), (1.0, 0.25, 3.0)),  # 3-D anisotropic
    ],
)
def test_squared_edt_matches_brute_force(
    seed: int, shape: tuple[int, ...], spacing: tuple[float, ...]
) -> None:
    mask = rng_from_seed(seed).random(shape) < 0.2
    expected = _brute_force(mask, spacing)
    np.testing.assert_allclose(squared_edt(mask, spacing), expected, rtol=1e-12)


def test_single_point_analytic() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 1] = True
    result = squared_edt(mask, (2.0, 3.0))
    assert result[2, 1] == 0.0
    assert result[0, 4] == pytest.approx((2 * 2.0) ** 2 + (3 * 3.0) ** 2)


def test_empty_mask_is_all_inf_and_full_mask_all_zero() -> None:
    empty = np.zeros((3, 4), dtype=bool)
    assert np.isinf(squared_edt(empty, (1.0, 1.0))).all()
    full = np.ones((3, 4), dtype=bool)
    assert (squared_edt(full, (1.0, 1.0)) == 0.0).all()


def test_signed_distance_signs_and_extremes() -> None:
    mask = np.zeros((1, 7), dtype=bool)
    mask[0, 2:5] = True
    signed = signed_distance(mask, (1.0, 1.0))
    assert signed[0, 3] == -2.0  # innermost cell: two cells from background
    assert signed[0, 2] == -1.0
    assert signed[0, 0] == 2.0
    assert np.isposinf(signed_distance(np.zeros((2, 2), dtype=bool), (1.0, 1.0))).all()
    assert np.isneginf(signed_distance(np.ones((2, 2), dtype=bool), (1.0, 1.0))).all()


def test_determinism_double_run() -> None:
    mask = rng_from_seed(7).random((5, 6, 4)) < 0.3
    first = squared_edt(mask, (1.0, 2.0, 0.5))
    second = squared_edt(mask, (1.0, 2.0, 0.5))
    assert np.array_equal(first, second)


def test_input_validation() -> None:
    with pytest.raises(FieldError, match=r"dtype must be bool"):
        squared_edt(np.zeros((2, 2), dtype=np.uint16), (1.0, 1.0))
    with pytest.raises(FieldError, match=r"one component per axis"):
        squared_edt(np.zeros((2, 2), dtype=bool), (1.0,))
    with pytest.raises(FieldError, match=r"positive"):
        squared_edt(np.zeros((2, 2), dtype=bool), (1.0, 0.0))
