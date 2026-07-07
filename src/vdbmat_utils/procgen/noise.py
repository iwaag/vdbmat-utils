"""Seeded gradient noise and fractal sums (plan D4).

Classic Perlin-style gradient noise on an integer lattice whose gradients come
from the counter-based lattice hash — so noise values depend only on physical
coordinates, the stream id, and the seed, never on domain bounds or evaluation
order. Frequencies are denominated in cycles per metre; anisotropy comes from
domain warping, not per-axis frequency.
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


def _corner_dot(
    cell_x: npt.NDArray[np.int64],
    cell_y: npt.NDArray[np.int64],
    cell_z: npt.NDArray[np.int64],
    frac_x: npt.NDArray[np.float64],
    frac_y: npt.NDArray[np.float64],
    frac_z: npt.NDArray[np.float64],
    corner: tuple[int, int, int],
    *,
    stream_id: int,
    seed: int,
) -> npt.NDArray[np.float64]:
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


def gradient_noise(
    domain: FormationDomain,
    *,
    frequency_per_m: float,
    stream_id: int,
    seed: int,
) -> ScalarField:
    """Single-octave gradient noise sampled at the domain's voxel centres.

    Output values lie in approximately ``[-1, 1]`` (not clamped, not
    rescaled). The noise lattice has pitch ``1 / frequency_per_m`` metres and
    is anchored at the local origin, so a domain extended at its far end
    reproduces the original interior exactly.
    """
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    x, y, z = domain.coordinates_xyz_m()
    lattice_x = x * frequency
    lattice_y = y * frequency
    lattice_z = z * frequency
    cell_x = np.floor(lattice_x).astype(np.int64)
    cell_y = np.floor(lattice_y).astype(np.int64)
    cell_z = np.floor(lattice_z).astype(np.int64)
    frac_x = lattice_x - cell_x
    frac_y = lattice_y - cell_y
    frac_z = lattice_z - cell_z

    def dot(corner: tuple[int, int, int]) -> npt.NDArray[np.float64]:
        return _corner_dot(
            cell_x,
            cell_y,
            cell_z,
            frac_x,
            frac_y,
            frac_z,
            corner,
            stream_id=stream_id,
            seed=seed,
        )

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
    values = np.ascontiguousarray(_lerp(y0, y1, w))
    return ScalarField(
        values=values,
        voxel_size_xyz_m=domain.voxel_size_xyz_m,
        local_to_world=domain.local_to_world,
    )


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
    """Fractal Brownian motion: amplitude-weighted sum of noise octaves.

    Octave ``i`` uses frequency ``frequency_per_m * lacunarity**i``, amplitude
    ``gain**i``, and stream id ``stream_id + i`` (plan D3 stream scheme). The
    sum is divided by the total amplitude, so the output range is
    approximately ``[-1, 1]`` independent of the octave count.
    """
    octaves, lacunarity, gain = _fractal_parameters(octaves, lacunarity, gain)
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    total = np.zeros(domain.shape_zyx, dtype=np.float64)
    amplitude_sum = 0.0
    for octave in range(octaves):
        amplitude = gain**octave
        octave_field = gradient_noise(
            domain,
            frequency_per_m=frequency * lacunarity**octave,
            stream_id=stream_id + octave,
            seed=seed,
        )
        total += amplitude * octave_field.values
        amplitude_sum += amplitude
    return ScalarField(
        values=total / amplitude_sum,
        voxel_size_xyz_m=domain.voxel_size_xyz_m,
        local_to_world=domain.local_to_world,
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
    """Ridged fractal noise: sharp creases for veins and fracture guides.

    Each octave contributes ``(1 - |noise|) ** 2`` (the standard ridge
    formula), accumulated exactly like :func:`fbm` — amplitude ``gain**i``,
    frequency ``frequency_per_m * lacunarity**i``, stream id
    ``stream_id + i`` — and divided by the total amplitude. Output lies in
    ``[0, 1]``, with values near 1 along the ridge sheets.
    """
    octaves, lacunarity, gain = _fractal_parameters(octaves, lacunarity, gain)
    frequency = _validate_positive(frequency_per_m, name="frequency_per_m")
    total = np.zeros(domain.shape_zyx, dtype=np.float64)
    amplitude_sum = 0.0
    for octave in range(octaves):
        amplitude = gain**octave
        octave_field = gradient_noise(
            domain,
            frequency_per_m=frequency * lacunarity**octave,
            stream_id=stream_id + octave,
            seed=seed,
        )
        ridge = (1.0 - np.abs(octave_field.values)) ** 2
        total += amplitude * ridge
        amplitude_sum += amplitude
    return ScalarField(
        values=total / amplitude_sum,
        voxel_size_xyz_m=domain.voxel_size_xyz_m,
        local_to_world=domain.local_to_world,
    )
