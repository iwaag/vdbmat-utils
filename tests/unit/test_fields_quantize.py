"""Unit tests for scalar-field quantization (plan Step 1.2)."""

import numpy as np
import pytest

from vdbmat_utils.fields import FieldError, ScalarField, quantize_to_labels


def _field(values: list[list[list[float]]]) -> ScalarField:
    return ScalarField(
        values=np.asarray(values, dtype=np.float64),
        voxel_size_xyz_m=(0.001, 0.001, 0.001),
    )


def test_bins_map_to_material_ids() -> None:
    field = _field([[[-2.0, -0.5, 0.0, 0.5, 2.0]]])
    labels = quantize_to_labels(
        field, bin_edges=(-1.0, 0.0, 1.0), material_ids=(0, 1, 2, 3)
    )
    assert labels.dtype == np.uint16
    assert labels.tolist() == [[[0, 1, 2, 2, 3]]]


def test_edge_value_goes_to_the_higher_bin() -> None:
    field = _field([[[0.0]]])
    labels = quantize_to_labels(field, bin_edges=(0.0,), material_ids=(1, 2))
    assert labels.tolist() == [[[2]]]


def test_nan_is_an_error() -> None:
    with pytest.raises(FieldError, match=r"NaN"):
        quantize_to_labels(
            _field([[[float("nan")]]]), bin_edges=(0.0,), material_ids=(0, 1)
        )


def test_non_increasing_edges_are_an_error() -> None:
    with pytest.raises(FieldError, match=r"strictly increasing"):
        quantize_to_labels(
            _field([[[0.0]]]), bin_edges=(1.0, 1.0), material_ids=(0, 1, 2)
        )


def test_id_count_mismatch_is_an_error() -> None:
    with pytest.raises(FieldError, match=r"len\(bin_edges\) \+ 1"):
        quantize_to_labels(_field([[[0.0]]]), bin_edges=(0.0,), material_ids=(1,))


def test_scalar_field_validation() -> None:
    with pytest.raises(FieldError, match=r"3-D"):
        ScalarField(
            values=np.zeros((2, 2), dtype=np.float64),
            voxel_size_xyz_m=(0.001, 0.001, 0.001),
        )
    with pytest.raises(FieldError, match=r"float64"):
        ScalarField(
            values=np.zeros((2, 2, 2), dtype=np.float32),
            voxel_size_xyz_m=(0.001, 0.001, 0.001),
        )
