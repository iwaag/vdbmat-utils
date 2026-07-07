"""Plan D3: the lattice hash is the determinism keystone (ADR-0010)."""

import numpy as np
import pytest

from vdbmat_utils.procgen import ProcgenError, hash_lattice, hash_to_unit

# The hash constants are fixed forever; these goldens pin them. A failure
# here means every previously generated formation would change.
_GOLDEN = [
    ((0, 0, 0), 0, 0, 0x98F0EF561B7B1390),
    ((1, 2, 3), 4, 5, 0x5D1CDDF79C95E611),
    ((-7, 11, -13), 2, 42, 0x18D9024EBED33670),
    ((1_000_000, -1_000_000, 0), 7, 123_456_789, 0xCB0E41A449F35685),
]


def test_golden_hash_values() -> None:
    for (ix, iy, iz), stream_id, seed, expected in _GOLDEN:
        result = hash_lattice(
            np.array([ix]), np.array([iy]), np.array([iz]),
            stream_id=stream_id, seed=seed,
        )
        assert int(result[0]) == expected


def test_broadcasting_matches_elementwise() -> None:
    ix = np.arange(-2, 3).reshape(1, 1, 5)
    iy = np.arange(0, 4).reshape(1, 4, 1)
    iz = np.arange(3, 6).reshape(3, 1, 1)
    broadcast = hash_lattice(ix, iy, iz, stream_id=1, seed=9)
    assert broadcast.shape == (3, 4, 5)
    for z in range(3):
        for y in range(4):
            for x in range(5):
                single = hash_lattice(
                    np.array([ix[0, 0, x]]),
                    np.array([iy[0, y, 0]]),
                    np.array([iz[z, 0, 0]]),
                    stream_id=1,
                    seed=9,
                )
                assert broadcast[z, y, x] == single[0]


def test_domain_extension_invariance() -> None:
    small = hash_lattice(
        np.arange(8).reshape(1, 1, 8),
        np.arange(8).reshape(1, 8, 1),
        np.arange(8).reshape(8, 1, 1),
        stream_id=3,
        seed=7,
    )
    large = hash_lattice(
        np.arange(12).reshape(1, 1, 12),
        np.arange(12).reshape(1, 12, 1),
        np.arange(12).reshape(12, 1, 1),
        stream_id=3,
        seed=7,
    )
    assert (large[:8, :8, :8] == small).all()


def test_stream_and_seed_sensitivity() -> None:
    coords = (np.arange(64), np.zeros(64, dtype=np.int64), np.zeros(64, dtype=np.int64))
    base = hash_lattice(*coords, stream_id=0, seed=0)
    assert (hash_lattice(*coords, stream_id=1, seed=0) != base).any()
    assert (hash_lattice(*coords, stream_id=0, seed=1) != base).any()


def test_unit_mapping_range_and_uniformity() -> None:
    hashes = hash_lattice(
        np.arange(10_000),
        np.zeros(10_000, dtype=np.int64),
        np.zeros(10_000, dtype=np.int64),
        stream_id=5,
        seed=11,
    )
    unit = hash_to_unit(hashes)
    assert unit.dtype == np.float64
    assert (unit >= 0.0).all() and (unit < 1.0).all()
    assert abs(float(unit.mean()) - 0.5) < 0.02


def test_input_validation() -> None:
    ones = np.ones(1, dtype=np.int64)
    with pytest.raises(ProcgenError):
        hash_lattice(np.ones(1, dtype=np.float64), ones, ones, stream_id=0, seed=0)
    with pytest.raises(ProcgenError):
        hash_lattice(ones, ones, ones, stream_id=-1, seed=0)
    with pytest.raises(ProcgenError):
        hash_lattice(ones, ones, ones, stream_id=0, seed=-1)
    with pytest.raises(ProcgenError):
        hash_lattice(ones, ones, ones, stream_id=True, seed=0)
