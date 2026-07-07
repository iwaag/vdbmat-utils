"""Seeded gradient noise and fractal sums (plan D4).

Classic Perlin-style gradient noise on an integer lattice whose gradients come
from the counter-based lattice hash — so noise values depend only on physical
coordinates, the stream id, and the seed, never on domain bounds or evaluation
order. Frequencies are denominated in cycles per metre; anisotropy comes from
domain warping, not per-axis frequency.

Each primitive has two forms: a coordinate-level ``*_at`` function evaluating
at arbitrary metre coordinates (this is what domain warping feeds displaced
coordinates into), and a domain-level wrapper returning a ``ScalarField`` on a
:class:`~vdbmat_utils.procgen.domain.FormationDomain`.
"""

import numpy as np
import numpy.typing as npt

from vdbmat_utils.fields import ScalarField

from .domain import FormationDomain
from .hashing import hash_lattice

# Perlin's 12 cube-edge gradient vectors, in (x, y, z) component order. The
# table and the ``hash % 12`` selection rule are fixed by ADR-0010.
_GRADIENTS = np.array(
    [
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, -1.0, -1.0),
    ],
    dtype=np.float64,
)
_GRAD_X = np.ascontiguousarray(_GRADIENTS[:, 0])
_GRAD_Y = np.ascontiguousarray(_GRADIENTS[:, 1])
_GRAD_Z = np.ascontiguousarray(_GRADIENTS[:, 2])
_TWELVE = np.uint64(12)


def _validate_positive(value: float, *, name: str) -> float:
    if not (float(value) > 0):
        from . import ProcgenError

        raise ProcgenError(f"{name} must be positive, got {value!r}")
    return float(value)


def _fade(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Perlin's quintic fade ``6t^5 - 15t^4 + 10t^3``."""
    result: npt.NDArray[np.float64] = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return result


def _lerp(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = a + t * (b - a)
    return result


def gradient_noise_at(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    frequency_per_m: float,
    stream_id: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Single-octave gradient noise at arbitrary metre coordinates.

    ``x``/``y``/``z`` are broadcastable float64 arrays of metre coordinates.
    Output values lie in approximately ``[-1, 1]`` (not clamped, not
    rescaled). The noise lattice has pitch ``1 / frequency_per_m`` metres and
    is anchored at the coordinate origin.
    """
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    lattice_x = np.asarray(x, dtype=np.float64) * frequency
    lattice_y = np.asarray(y, dtype=np.float64) * frequency
    lattice_z = np.asarray(z, dtype=np.float64) * frequency
    cell_x = np.floor(lattice_x).astype(np.int64)
    cell_y = np.floor(lattice_y).astype(np.int64)
    cell_z = np.floor(lattice_z).astype(np.int64)
    frac_x = lattice_x - cell_x
    frac_y = lattice_y - cell_y
    frac_z = lattice_z - cell_z

    def dot(corner: tuple[int, int, int]) -> npt.NDArray[np.float64]:
        cx, cy, cz = corner
        hashes = hash_lattice(
            cell_x + cx, cell_y + cy, cell_z + cz, stream_id=stream_id, seed=seed
        )
        index = (hashes % _TWELVE).astype(np.intp)
        result: npt.NDArray[np.float64] = (
            _GRAD_X[index] * (frac_x - cx)
            + _GRAD_Y[index] * (frac_y - cy)
            + _GRAD_Z[index] * (frac_z - cz)
        )
        return result

    u = _fade(frac_x)
    v = _fade(frac_y)
    w = _fade(frac_z)
    # Fixed interpolation order: x, then y, then z (ADR-0010).
    x00 = _lerp(dot((0, 0, 0)), dot((1, 0, 0)), u)
    x10 = _lerp(dot((0, 1, 0)), dot((1, 1, 0)), u)
    x01 = _lerp(dot((0, 0, 1)), dot((1, 0, 1)), u)
    x11 = _lerp(dot((0, 1, 1)), dot((1, 1, 1)), u)
    y0 = _lerp(x00, x10, v)
    y1 = _lerp(x01, x11, v)
    return _lerp(y0, y1, w)


def _fractal_parameters(
    octaves: int, lacunarity: float, gain: float
) -> tuple[int, float, float]:
    if isinstance(octaves, bool) or not isinstance(octaves, int) or octaves < 1:
        from . import ProcgenError

        raise ProcgenError(f"octaves must be a positive integer, got {octaves!r}")
    return (
        octaves,
        _validate_positive(lacunarity, name="lacunarity"),
        _validate_positive(gain, name="gain"),
    )


def fbm_at(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    frequency_per_m: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    stream_id: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Fractal Brownian motion at arbitrary metre coordinates.

    Octave ``i`` uses frequency ``frequency_per_m * lacunarity**i``, amplitude
    ``gain**i``, and stream id ``stream_id + i`` (plan D3 stream scheme). The
    sum is divided by the total amplitude, so the output range is
    approximately ``[-1, 1]`` independent of the octave count.
    """
    octaves, lacunarity, gain = _fractal_parameters(octaves, lacunarity, gain)
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
    total = np.zeros(shape, dtype=np.float64)
    amplitude_sum = 0.0
    for octave in range(octaves):
        amplitude = gain**octave
        total += amplitude * gradient_noise_at(
            x,
            y,
            z,
            frequency_per_m=frequency * lacunarity**octave,
            stream_id=stream_id + octave,
            seed=seed,
        )
        amplitude_sum += amplitude
    result: npt.NDArray[np.float64] = total / amplitude_sum
    return result


def ridged_fbm_at(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    frequency_per_m: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    stream_id: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Ridged fractal noise at arbitrary metre coordinates.

    Each octave contributes ``(1 - |noise|) ** 2`` (the standard ridge
    formula), accumulated exactly like :func:`fbm_at` — amplitude ``gain**i``,
    frequency ``frequency_per_m * lacunarity**i``, stream id
    ``stream_id + i`` — and divided by the total amplitude. Output lies in
    ``[0, 1]``, with values near 1 along the ridge sheets.
    """
    octaves, lacunarity, gain = _fractal_parameters(octaves, lacunarity, gain)
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
    total = np.zeros(shape, dtype=np.float64)
    amplitude_sum = 0.0
    for octave in range(octaves):
        amplitude = gain**octave
        octave_values = gradient_noise_at(
            x,
            y,
            z,
            frequency_per_m=frequency * lacunarity**octave,
            stream_id=stream_id + octave,
            seed=seed,
        )
        total += amplitude * (1.0 - np.abs(octave_values)) ** 2
        amplitude_sum += amplitude
    result: npt.NDArray[np.float64] = total / amplitude_sum
    return result


def _as_field(
    domain: FormationDomain, values: npt.NDArray[np.float64]
) -> ScalarField:
    return ScalarField(
        values=np.ascontiguousarray(np.broadcast_to(values, domain.shape_zyx)),
        voxel_size_xyz_m=domain.voxel_size_xyz_m,
        local_to_world=domain.local_to_world,
    )


def gradient_noise(
    domain: FormationDomain,
    *,
    frequency_per_m: float,
    stream_id: int,
    seed: int,
) -> ScalarField:
    """Single-octave gradient noise sampled at the domain's voxel centres.

    A domain extended at its far end reproduces the original interior exactly
    (the lattice is anchored at the local origin).
    """
    x, y, z = domain.coordinates_xyz_m()
    return _as_field(
        domain,
        gradient_noise_at(
            x, y, z, frequency_per_m=frequency_per_m, stream_id=stream_id, seed=seed
        ),
    )


def fbm(
    domain: FormationDomain,
    *,
    frequency_per_m: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    stream_id: int,
    seed: int,
) -> ScalarField:
    """Fractal Brownian motion sampled at the domain's voxel centres."""
    x, y, z = domain.coordinates_xyz_m()
    return _as_field(
        domain,
        fbm_at(
            x,
            y,
            z,
            frequency_per_m=frequency_per_m,
            octaves=octaves,
            lacunarity=lacunarity,
            gain=gain,
            stream_id=stream_id,
            seed=seed,
        ),
    )


def ridged_fbm(
    domain: FormationDomain,
    *,
    frequency_per_m: float,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    stream_id: int,
    seed: int,
) -> ScalarField:
    """Ridged fractal noise sampled at the domain's voxel centres."""
    x, y, z = domain.coordinates_xyz_m()
    return _as_field(
        domain,
        ridged_fbm_at(
            x,
            y,
            z,
            frequency_per_m=frequency_per_m,
            octaves=octaves,
            lacunarity=lacunarity,
            gain=gain,
            stream_id=stream_id,
            seed=seed,
        ),
    )
