"""Counter-based lattice hashing — the determinism keystone (plan D3).

All lattice-dependent randomness (noise gradients, cell feature points,
per-cell material picks) derives from a fixed SplitMix64-style mix over the
lattice coordinates, a stream id, and the generator seed. Values therefore
depend only on ``(ix, iy, iz, stream_id, seed)`` — never on evaluation order,
chunking, or domain bounds — so enlarging a domain never reshuffles its
interior (ADR-0010).

The constants below are fixed forever; changing any of them changes every
generated formation.
"""

import numpy as np
import numpy.typing as npt

# SplitMix64 finalizer constants (Steele, Lea & Flood 2014).
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
# Per-input fold keys: distinct large odd constants, one per hash input.
_KEY_SEED = np.uint64(0x9E3779B97F4A7C15)
_KEY_X = np.uint64(0xC2B2AE3D27D4EB4F)
_KEY_Y = np.uint64(0x165667B19E3779F9)
_KEY_Z = np.uint64(0x27D4EB2F165667C5)
_KEY_STREAM = np.uint64(0x85EBCA77C2B2AE63)

_UNIT_SCALE = float(2.0**-53)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)
_SHIFT_11 = np.uint64(11)


def _mix(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """SplitMix64 finalizer, vectorized over uint64 arrays."""
    state = (state ^ (state >> _SHIFT_30)) * _MIX_1
    state = (state ^ (state >> _SHIFT_27)) * _MIX_2
    return state ^ (state >> _SHIFT_31)


def _as_uint64(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.uint64]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        from . import ProcgenError

        raise ProcgenError(f"{name} must be integer-typed, got {array.dtype}")
    # Two's-complement wrap for negative lattice coordinates: deterministic
    # and collision-free within the int64 range.
    return array.astype(np.int64).astype(np.uint64)


def hash_lattice(
    ix: npt.ArrayLike,
    iy: npt.ArrayLike,
    iz: npt.ArrayLike,
    *,
    stream_id: int,
    seed: int,
) -> npt.NDArray[np.uint64]:
    """Hash lattice coordinates to uniform uint64 values.

    ``ix``/``iy``/``iz`` are integer arrays (broadcastable against each
    other); ``stream_id`` names the consumer so two primitives sharing a seed
    still get independent lattices; ``seed`` is the generator seed. The fold
    order (seed, x, y, z, stream) and all constants are fixed by ADR-0010.
    """
    if not isinstance(stream_id, int) or isinstance(stream_id, bool) or stream_id < 0:
        from . import ProcgenError

        raise ProcgenError(
            f"stream_id must be a non-negative integer, got {stream_id!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        from . import ProcgenError

        raise ProcgenError(f"seed must be a non-negative integer, got {seed!r}")
    mask = (1 << 64) - 1
    seed_word = np.asarray((seed * 2 + 1) & mask, dtype=np.uint64)
    stream_word = np.asarray(stream_id & mask, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = _mix(seed_word ^ _KEY_SEED)
        state = _mix(state ^ (_as_uint64(ix, name="ix") * _KEY_X))
        state = _mix(state ^ (_as_uint64(iy, name="iy") * _KEY_Y))
        state = _mix(state ^ (_as_uint64(iz, name="iz") * _KEY_Z))
        state = _mix(state ^ (stream_word * _KEY_STREAM))
    return state


def hash_derive(hashes: npt.NDArray[np.uint64], *, salt: int) -> npt.NDArray[np.uint64]:
    """Derive an independent hash stream from existing hashes.

    Used where one lattice point needs several independent uniform draws
    (cell jitter components, per-cell material picks): each consumer folds a
    distinct non-negative ``salt`` into the parent hash. The derivation is
    ``mix(h ^ (2*salt + 1) * KEY)`` — fixed by ADR-0010.
    """
    if not isinstance(salt, int) or isinstance(salt, bool) or salt < 0:
        from . import ProcgenError

        raise ProcgenError(f"salt must be a non-negative integer, got {salt!r}")
    salt_word = np.asarray((salt * 2 + 1) & ((1 << 64) - 1), dtype=np.uint64)
    with np.errstate(over="ignore"):
        return _mix(hashes ^ (salt_word * _KEY_STREAM))


def hash_to_unit(hashes: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
    """Map uint64 hashes to uniform float64 values in ``[0, 1)``.

    Uses the top 53 bits so every representable output is exact.
    """
    result: npt.NDArray[np.float64] = (hashes >> _SHIFT_11) * np.float64(_UNIT_SCALE)
    return result
