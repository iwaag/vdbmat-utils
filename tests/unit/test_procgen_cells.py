"""Plan D4: Worley cellular fields and domain warping."""

import numpy as np
import pytest

from vdbmat_utils.fields import ScalarField
from vdbmat_utils.procgen import (
    FormationDomain,
    ProcgenError,
    fbm,
    fbm_at,
    hash_derive,
    hash_lattice,
    hash_to_unit,
    warped_coordinates,
    worley,
)

_DOMAIN = FormationDomain(
    shape_zyx=(10, 12, 14), voxel_size_xyz_m=(0.001, 0.001, 0.001)
)


def _offset_fields(
    domain: FormationDomain,
) -> tuple[ScalarField, ScalarField, ScalarField]:
    x, y, z = (
        fbm(domain, frequency_per_m=300.0, octaves=2, stream_id=10 + i, seed=7)
        for i in range(3)
    )
    return x, y, z


def _brute_force_worley(
    point_xyz: tuple[float, float, float],
    *,
    cell_size_m: float,
    stream_id: int,
    seed: int,
) -> tuple[float, float, int]:
    """Same 3x3x3-window definition, one point at a time, re-deriving the
    feature points from the public hash API (re-pins the jitter scheme)."""
    px, py, pz = point_xyz
    cell = tuple(int(np.floor(c / cell_size_m)) for c in point_xyz)
    candidates: list[tuple[float, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx, cy, cz = cell[0] + dx, cell[1] + dy, cell[2] + dz
                identity = hash_lattice(
                    np.array([cx]), np.array([cy]), np.array([cz]),
                    stream_id=stream_id, seed=seed,
                )
                jitter = [
                    float(hash_to_unit(hash_derive(identity, salt=salt))[0])
                    for salt in (0, 1, 2)
                ]
                fx = (cx + jitter[0]) * cell_size_m
                fy = (cy + jitter[1]) * cell_size_m
                fz = (cz + jitter[2]) * cell_size_m
                distance = float(
                    np.sqrt((px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2)
                )
                candidates.append((distance, int(identity[0])))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][0], candidates[1][0], candidates[0][1]


def test_worley_matches_brute_force() -> None:
    cells = worley(_DOMAIN, cell_size_m=0.004, stream_id=3, seed=17)
    x, y, z = _DOMAIN.coordinates_xyz_m()
    rng = np.random.default_rng(0)
    for _ in range(30):
        iz = int(rng.integers(_DOMAIN.shape_zyx[0]))
        iy = int(rng.integers(_DOMAIN.shape_zyx[1]))
        ix = int(rng.integers(_DOMAIN.shape_zyx[2]))
        point = (float(x[0, 0, ix]), float(y[0, iy, 0]), float(z[iz, 0, 0]))
        f1, f2, site = _brute_force_worley(
            point, cell_size_m=0.004, stream_id=3, seed=17
        )
        assert cells.f1.values[iz, iy, ix] == pytest.approx(f1, rel=1e-12)
        assert cells.f2.values[iz, iy, ix] == pytest.approx(f2, rel=1e-12)
        assert int(cells.site_id[iz, iy, ix]) == site


def test_worley_invariants() -> None:
    cells = worley(_DOMAIN, cell_size_m=0.003, stream_id=1, seed=5)
    assert (cells.f1.values >= 0).all()
    assert (cells.f2.values >= cells.f1.values).all()
    assert (cells.boundary().values >= 0).all()
    assert cells.site_id.shape == _DOMAIN.shape_zyx
    # More than one cell must appear at this pitch on this domain.
    assert len(np.unique(cells.site_id)) > 1


def test_worley_determinism_and_sensitivity() -> None:
    first = worley(_DOMAIN, cell_size_m=0.003, stream_id=1, seed=5)
    second = worley(_DOMAIN, cell_size_m=0.003, stream_id=1, seed=5)
    assert first.f1.values.tobytes() == second.f1.values.tobytes()
    assert first.site_id.tobytes() == second.site_id.tobytes()
    other = worley(_DOMAIN, cell_size_m=0.003, stream_id=2, seed=5)
    assert not np.array_equal(first.site_id, other.site_id)


def test_worley_domain_extension_invariance() -> None:
    small = FormationDomain(shape_zyx=(6, 6, 6), voxel_size_xyz_m=(0.001,) * 3)
    large = FormationDomain(shape_zyx=(9, 9, 9), voxel_size_xyz_m=(0.001,) * 3)
    inner = worley(small, cell_size_m=0.002, stream_id=4, seed=13)
    outer = worley(large, cell_size_m=0.002, stream_id=4, seed=13)
    assert (
        outer.f1.values[:6, :6, :6].tobytes() == inner.f1.values.tobytes()
    )
    assert (
        outer.site_id[:6, :6, :6].tobytes() == inner.site_id.tobytes()
    )


def test_worley_validation() -> None:
    with pytest.raises(ProcgenError):
        worley(_DOMAIN, cell_size_m=0.0, stream_id=0, seed=0)


def test_warped_coordinates_compose_manually() -> None:
    offsets = _offset_fields(_DOMAIN)
    x, y, z = _DOMAIN.coordinates_xyz_m()
    wx, wy, wz = warped_coordinates(
        _DOMAIN, offsets_xyz=offsets, amplitude_m=0.002
    )
    np.testing.assert_array_equal(wx, x + 0.002 * offsets[0].values)
    np.testing.assert_array_equal(wy, y + 0.002 * offsets[1].values)
    np.testing.assert_array_equal(wz, z + 0.002 * offsets[2].values)
    # Feeding warped coordinates into a primitive equals manual composition.
    warped_noise = fbm_at(
        wx, wy, wz, frequency_per_m=400.0, octaves=2, stream_id=20, seed=7
    )
    manual = fbm_at(
        x + 0.002 * offsets[0].values,
        y + 0.002 * offsets[1].values,
        z + 0.002 * offsets[2].values,
        frequency_per_m=400.0,
        octaves=2,
        stream_id=20,
        seed=7,
    )
    assert warped_noise.tobytes() == manual.tobytes()


def test_warp_zero_amplitude_is_identity() -> None:
    offsets = _offset_fields(_DOMAIN)
    x, y, z = _DOMAIN.coordinates_xyz_m()
    wx, wy, wz = warped_coordinates(_DOMAIN, offsets_xyz=offsets, amplitude_m=0.0)
    np.testing.assert_array_equal(wx, np.broadcast_to(x, _DOMAIN.shape_zyx))
    np.testing.assert_array_equal(wy, np.broadcast_to(y, _DOMAIN.shape_zyx))
    np.testing.assert_array_equal(wz, np.broadcast_to(z, _DOMAIN.shape_zyx))


def test_warp_validation() -> None:
    offsets = _offset_fields(_DOMAIN)
    with pytest.raises(ProcgenError):
        warped_coordinates(_DOMAIN, offsets_xyz=offsets, amplitude_m=-1.0)
    other_domain = FormationDomain(
        shape_zyx=(4, 4, 4), voxel_size_xyz_m=(0.001,) * 3
    )
    with pytest.raises(ProcgenError):
        warped_coordinates(other_domain, offsets_xyz=offsets, amplitude_m=0.001)
