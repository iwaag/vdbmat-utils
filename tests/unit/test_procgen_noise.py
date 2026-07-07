"""Plan D4: gradient noise, fBm, and ridged fBm on formation domains."""

import numpy as np
import pytest

from vdbmat_utils.core.errors import ConfigError
from vdbmat_utils.fields import ScalarField
from vdbmat_utils.procgen import (
    FormationDomain,
    ProcgenError,
    fbm,
    gradient_noise,
    ridged_fbm,
)

_DOMAIN = FormationDomain(
    shape_zyx=(24, 20, 16), voxel_size_xyz_m=(0.001, 0.001, 0.001)
)


def test_domain_validation() -> None:
    with pytest.raises(ConfigError):
        FormationDomain(shape_zyx=(0, 4, 4), voxel_size_xyz_m=(0.001,) * 3)
    with pytest.raises(ConfigError):
        FormationDomain(shape_zyx=(4, 4), voxel_size_xyz_m=(0.001,) * 3)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        FormationDomain(shape_zyx=(4, 4, 4), voxel_size_xyz_m=(0.001, 0.0, 0.001))
    with pytest.raises(ConfigError):
        FormationDomain(shape_zyx=(300, 4, 4), voxel_size_xyz_m=(0.001,) * 3)
    with pytest.raises(ConfigError):
        FormationDomain(
            shape_zyx=(200, 200, 200),
            voxel_size_xyz_m=(0.001,) * 3,
            max_total_cells=1_000_000,
        )


def test_domain_coordinates_are_voxel_centres() -> None:
    domain = FormationDomain(
        shape_zyx=(2, 3, 4), voxel_size_xyz_m=(0.001, 0.002, 0.004)
    )
    x, y, z = domain.coordinates_xyz_m()
    assert x.shape == (1, 1, 4) and y.shape == (1, 3, 1) and z.shape == (2, 1, 1)
    np.testing.assert_allclose(x.ravel(), [0.0005, 0.0015, 0.0025, 0.0035])
    np.testing.assert_allclose(y.ravel(), [0.001, 0.003, 0.005])
    np.testing.assert_allclose(z.ravel(), [0.002, 0.006])


def test_noise_returns_scalar_field_with_domain_geometry() -> None:
    field = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=0, seed=1)
    assert isinstance(field, ScalarField)
    assert field.values.shape == _DOMAIN.shape_zyx
    assert field.values.dtype == np.float64
    assert field.voxel_size_xyz_m == _DOMAIN.voxel_size_xyz_m


def test_noise_range_and_variation() -> None:
    field = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=0, seed=1)
    assert float(np.abs(field.values).max()) < 1.5
    assert float(field.values.std()) > 0.05


def test_noise_is_smooth() -> None:
    # With 10 voxels per lattice cell, adjacent samples must stay close.
    field = gradient_noise(_DOMAIN, frequency_per_m=100.0, stream_id=0, seed=1)
    for axis in range(3):
        steps = np.abs(np.diff(field.values, axis=axis))
        assert float(steps.max()) < 0.35


def test_noise_determinism_double_run() -> None:
    first = gradient_noise(_DOMAIN, frequency_per_m=500.0, stream_id=2, seed=9)
    second = gradient_noise(_DOMAIN, frequency_per_m=500.0, stream_id=2, seed=9)
    assert first.values.tobytes() == second.values.tobytes()


def test_noise_domain_extension_invariance() -> None:
    """Plan D3 keystone: a larger domain reproduces the interior byte-exactly."""
    small = FormationDomain(shape_zyx=(8, 8, 8), voxel_size_xyz_m=(0.001,) * 3)
    large = FormationDomain(shape_zyx=(12, 12, 12), voxel_size_xyz_m=(0.001,) * 3)
    for function in (gradient_noise, fbm, ridged_fbm):
        kwargs = {} if function is gradient_noise else {"octaves": 3}
        inner = function(
            small, frequency_per_m=700.0, stream_id=4, seed=13, **kwargs
        )
        outer = function(
            large, frequency_per_m=700.0, stream_id=4, seed=13, **kwargs
        )
        assert outer.values[:8, :8, :8].tobytes() == inner.values.tobytes()


def test_noise_stream_and_seed_sensitivity() -> None:
    base = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=0, seed=1)
    other_stream = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=1, seed=1)
    other_seed = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=0, seed=2)
    assert not np.array_equal(base.values, other_stream.values)
    assert not np.array_equal(base.values, other_seed.values)


def test_voxel_size_enters_evaluation() -> None:
    coarse = FormationDomain(shape_zyx=(8, 8, 8), voxel_size_xyz_m=(0.002,) * 3)
    fine = FormationDomain(shape_zyx=(8, 8, 8), voxel_size_xyz_m=(0.001,) * 3)
    a = gradient_noise(coarse, frequency_per_m=400.0, stream_id=0, seed=1)
    b = gradient_noise(fine, frequency_per_m=400.0, stream_id=0, seed=1)
    assert not np.array_equal(a.values, b.values)


def test_fbm_single_octave_equals_gradient_noise() -> None:
    single = fbm(_DOMAIN, frequency_per_m=400.0, octaves=1, stream_id=6, seed=3)
    plain = gradient_noise(_DOMAIN, frequency_per_m=400.0, stream_id=6, seed=3)
    assert single.values.tobytes() == plain.values.tobytes()


def test_fbm_octave_normalization_keeps_range() -> None:
    many = fbm(_DOMAIN, frequency_per_m=400.0, octaves=5, stream_id=6, seed=3)
    assert float(np.abs(many.values).max()) < 1.5
    assert float(many.values.std()) > 0.02


def test_ridged_fbm_range() -> None:
    field = ridged_fbm(_DOMAIN, frequency_per_m=400.0, octaves=3, stream_id=8, seed=3)
    assert (field.values >= 0.0).all() and (field.values <= 1.0).all()
    assert float(field.values.std()) > 0.01


def test_parameter_validation() -> None:
    with pytest.raises(ProcgenError):
        gradient_noise(_DOMAIN, frequency_per_m=0.0, stream_id=0, seed=1)
    with pytest.raises(ProcgenError):
        fbm(_DOMAIN, frequency_per_m=400.0, octaves=0, stream_id=0, seed=1)
    with pytest.raises(ProcgenError):
        fbm(_DOMAIN, frequency_per_m=400.0, octaves=2, gain=0.0, stream_id=0, seed=1)
    with pytest.raises(ProcgenError):
        ridged_fbm(
            _DOMAIN, frequency_per_m=400.0, octaves=2, lacunarity=-1.0,
            stream_id=0, seed=1,
        )
