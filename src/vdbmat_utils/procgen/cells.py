"""Worley/Voronoi cellular fields (plan D4).

One feature point per lattice cell, at a hash-derived jittered position, so
cell structure — like noise — depends only on physical coordinates, the
stream id, and the seed. Distances are Euclidean metres.

Semantics fixed by ADR-0011: F1/F2 and the nearest-site id are computed over
the 3x3x3 lattice-cell neighbourhood of each query point (the standard Worley
search window). This is the *definition*, not an approximation to something
else — the brute-force tests use the same window.
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from vdbmat_utils.fields import ScalarField

from .domain import FormationDomain
from .hashing import hash_derive, hash_lattice, hash_to_unit

# Salts for the three jitter components of a cell's feature point (ADR-0011).
_SALT_JITTER_X = 0
_SALT_JITTER_Y = 1
_SALT_JITTER_Z = 2


@dataclasses.dataclass(frozen=True)
class WorleyCells:
    """Cellular-field outputs on a formation domain.

    ``f1``/``f2`` are the distances (metres) to the nearest and second-nearest
    feature points; ``f2 - f1`` is the crystal-boundary field. ``site_id`` is
    the uint64 hash identity of the nearest feature point — stable under
    domain changes, so per-cell material picks key off it (plan D5 grains).
    """

    f1: ScalarField
    f2: ScalarField
    site_id: npt.NDArray[np.uint64]

    def boundary(self) -> ScalarField:
        """The ``f2 - f1`` field: zero on cell boundaries, positive inside."""
        return ScalarField(
            values=self.f2.values - self.f1.values,
            voxel_size_xyz_m=self.f1.voxel_size_xyz_m,
            local_to_world=self.f1.local_to_world,
        )


def _feature_point(
    cell_x: npt.NDArray[np.int64],
    cell_y: npt.NDArray[np.int64],
    cell_z: npt.NDArray[np.int64],
    pitch: float,
    *,
    stream_id: int,
    seed: int,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """The identity and metre position of each cell's feature point."""
    identity = hash_lattice(cell_x, cell_y, cell_z, stream_id=stream_id, seed=seed)
    px = (cell_x + hash_to_unit(hash_derive(identity, salt=_SALT_JITTER_X))) * pitch
    py = (cell_y + hash_to_unit(hash_derive(identity, salt=_SALT_JITTER_Y))) * pitch
    pz = (cell_z + hash_to_unit(hash_derive(identity, salt=_SALT_JITTER_Z))) * pitch
    return identity, px, py, pz


def worley_at(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    cell_size_m: float,
    stream_id: int,
    seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint64],
]:
    """Worley ``(f1, f2, site_id)`` at arbitrary metre coordinates.

    ``x``/``y``/``z`` are broadcastable float64 metre coordinates;
    ``cell_size_m`` is the isotropic lattice pitch. Ties in the distance
    ordering are resolved by search order over the fixed 3x3x3 offset scan
    (z-major, documented for completeness; exact ties have measure zero).
    """
    if not (float(cell_size_m) > 0):
        from . import ProcgenError

        raise ProcgenError(f"cell_size_m must be positive, got {cell_size_m!r}")
    pitch = float(cell_size_m)
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    pz = np.asarray(z, dtype=np.float64)
    cell_x = np.floor(px / pitch).astype(np.int64)
    cell_y = np.floor(py / pitch).astype(np.int64)
    cell_z = np.floor(pz / pitch).astype(np.int64)
    shape = np.broadcast_shapes(px.shape, py.shape, pz.shape)
    best = np.full(shape, np.inf, dtype=np.float64)
    second = np.full(shape, np.inf, dtype=np.float64)
    best_site = np.zeros(shape, dtype=np.uint64)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                identity, fx, fy, fz = _feature_point(
                    cell_x + dx,
                    cell_y + dy,
                    cell_z + dz,
                    pitch,
                    stream_id=stream_id,
                    seed=seed,
                )
                distance = np.sqrt(
                    (px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2
                )
                closer = distance < best
                second = np.where(
                    closer, best, np.minimum(second, distance)
                )
                best_site = np.where(closer, identity, best_site)
                best = np.where(closer, distance, best)
    return best, second, best_site


def worley(
    domain: FormationDomain,
    *,
    cell_size_m: float,
    stream_id: int,
    seed: int,
) -> WorleyCells:
    """Worley cellular fields sampled at the domain's voxel centres."""
    x, y, z = domain.coordinates_xyz_m()
    f1, f2, site_id = worley_at(
        x, y, z, cell_size_m=cell_size_m, stream_id=stream_id, seed=seed
    )

    def field(values: npt.NDArray[np.float64]) -> ScalarField:
        return ScalarField(
            values=np.ascontiguousarray(np.broadcast_to(values, domain.shape_zyx)),
            voxel_size_xyz_m=domain.voxel_size_xyz_m,
            local_to_world=domain.local_to_world,
        )

    return WorleyCells(
        f1=field(f1),
        f2=field(f2),
        site_id=np.ascontiguousarray(np.broadcast_to(site_id, domain.shape_zyx)),
    )
