"""Unit tests for the primitive-array config contract and generation body."""

import numpy as np
import pytest

from vdbmat_utils.core import ConfigError
from vdbmat_utils.primitives import (
    PrimitiveArrayConfig,
    PrimitiveArrayError,
    generate_primitive_array,
)


def _base_kwargs() -> dict:
    return dict(
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        primitive="cube",
        counts_xyz=(3, 2, 1),
        primitive_size_m=4e-4,
        gap_m=2e-4,
        margin_m=1e-4,
    )


def _config(**overrides: object) -> PrimitiveArrayConfig:
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    return PrimitiveArrayConfig(**kwargs)


# --- config contract -------------------------------------------------------


def test_valid_config_round_trips() -> None:
    config = _config()
    text = config.to_json()
    restored = PrimitiveArrayConfig.from_json(text)
    assert restored == config
    assert restored.to_json() == text


def test_unknown_field_rejected() -> None:
    payload = _config().to_json()
    import json

    data = json.loads(payload)
    data["bogus"] = 1
    with pytest.raises(ConfigError, match="unknown configuration fields"):
        PrimitiveArrayConfig.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(voxel_size_xyz_m=(0.0, 1e-4, 1e-4)),
        dict(voxel_size_xyz_m=(-1e-4, 1e-4, 1e-4)),
        dict(voxel_size_xyz_m=(1e-4, 1e-4)),
        dict(primitive="cylinder"),
        dict(counts_xyz=(0, 2, 1)),
        dict(counts_xyz=(3, 2)),
        dict(counts_xyz=(1.5, 2, 1)),
        dict(primitive_size_m=0.0),
        dict(primitive_size_m=-1e-4),
        dict(gap_m=-1e-6),
        dict(margin_m=-1e-6),
        dict(base_material_name="air"),
        dict(inclusion_material_name="air"),
        dict(base_material_name="unobtanium"),
        dict(base_material_name="white-resin", inclusion_material_name="white-resin"),
        dict(max_axis_cells=0),
        dict(max_total_cells=-1),
    ],
)
def test_rejected_configs(overrides: dict) -> None:
    with pytest.raises(PrimitiveArrayError):
        _config(**overrides)


def test_gap_zero_and_margin_zero_are_accepted() -> None:
    _config(gap_m=0.0, margin_m=0.0)


def test_config_digest_stable_for_equal_configs() -> None:
    from vdbmat_utils.core import config_digest

    a = _config()
    b = _config()
    assert config_digest(a) == config_digest(b)


def test_config_digest_changes_with_seed() -> None:
    from vdbmat_utils.core import config_digest

    a = _config(seed=0)
    b = _config(seed=1)
    assert config_digest(a) != config_digest(b)


# --- grid derivation and classification ------------------------------------


def test_exact_multiple_cube_voxel_count() -> None:
    # voxel_size 1e-4; size=4 voxels, gap=2 voxels, margin=1 voxel.
    config = _config(counts_xyz=(3, 2, 1))
    volume = generate_primitive_array(config)
    inclusion_id = 3  # black-opaque-resin
    label = volume.material_id
    inclusion_count = int(np.count_nonzero(label == inclusion_id))
    voxels_per_cube = 4 * 4 * 4
    assert inclusion_count == voxels_per_cube * 3 * 2 * 1

    # extent_x = 2*1e-4 + 3*4e-4 + 2*2e-4 = 18e-4 -> 18 cells
    # extent_y = 2*1e-4 + 2*4e-4 + 1*2e-4 = 12e-4 -> 12 cells
    # extent_z = 2*1e-4 + 1*4e-4 + 0*2e-4 = 6e-4 -> 6 cells
    assert volume.geometry.shape_zyx == (6, 12, 18)


def test_sphere_flip_symmetry_leaves_payload_unchanged() -> None:
    config = _config(
        primitive="sphere", counts_xyz=(3, 2, 2), gap_m=2e-4, margin_m=1e-4
    )
    volume = generate_primitive_array(config)
    label = volume.material_id
    # The array is symmetric per-axis under a full grid flip since the
    # counts/size/gap/margin recipe places primitives symmetrically about
    # the block centre on each axis.
    flipped = label[::-1, ::-1, ::-1]
    np.testing.assert_array_equal(label, flipped)


def test_degenerate_single_primitive_gap_and_margin_zero() -> None:
    config = _config(counts_xyz=(1, 1, 1), gap_m=0.0, margin_m=0.0)
    volume = generate_primitive_array(config)
    assert volume.geometry.shape_zyx == (4, 4, 4)
    assert int(np.count_nonzero(volume.material_id == 3)) == 4 * 4 * 4


def test_touching_cubes_gap_zero_no_double_paint_or_gap() -> None:
    config = _config(counts_xyz=(2, 1, 1), gap_m=0.0, margin_m=0.0)
    volume = generate_primitive_array(config)
    # extent_x = 2*4e-4 = 8e-4 -> 8 cells; two touching 4-cell cubes fill
    # the whole span with no gap and no overlap double count.
    assert volume.geometry.shape_zyx == (4, 4, 8)
    assert int(np.count_nonzero(volume.material_id == 3)) == 4 * 4 * 8


def test_boundary_tie_is_closed_inclusive_for_cube() -> None:
    # size == voxel_size, gap == 0, margin == 0: every cell centre sits
    # exactly on a primitive boundary or interior; all cells must classify.
    config = _config(
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        counts_xyz=(2, 1, 1),
        primitive_size_m=1e-4,
        gap_m=0.0,
        margin_m=0.0,
    )
    volume = generate_primitive_array(config)
    assert volume.geometry.shape_zyx == (1, 1, 2)
    assert int(np.count_nonzero(volume.material_id == 3)) == 2


def test_guard_axis_exceeded_raises_actionable_error() -> None:
    config = _config(
        voxel_size_xyz_m=(1e-6, 1e-6, 1e-6),
        counts_xyz=(1, 1, 1),
        primitive_size_m=1e-3,
        gap_m=0.0,
        margin_m=0.0,
        max_axis_cells=100,
    )
    with pytest.raises(PrimitiveArrayError, match="coarser voxel size"):
        generate_primitive_array(config)


def test_guard_total_exceeded_raises() -> None:
    config = _config(
        voxel_size_xyz_m=(1e-6, 1e-6, 1e-6),
        counts_xyz=(1, 1, 1),
        primitive_size_m=1e-4,
        gap_m=0.0,
        margin_m=0.0,
        max_axis_cells=10_000,
        max_total_cells=10,
    )
    with pytest.raises(PrimitiveArrayError, match="exceeding the bound"):
        generate_primitive_array(config)


def test_dtype_and_axis_order() -> None:
    volume = generate_primitive_array(_config())
    assert volume.material_id.dtype == np.uint16
    assert volume.material_id.ndim == 3


def test_palette_matches_configured_materials() -> None:
    volume = generate_primitive_array(_config())
    names = {m.name for m in volume.palette}
    assert names == {"air", "transparent-resin", "black-opaque-resin"}


def test_seed_does_not_affect_payload() -> None:
    a = generate_primitive_array(_config(seed=0))
    b = generate_primitive_array(_config(seed=42))
    np.testing.assert_array_equal(a.material_id, b.material_id)
