"""The bounded grid a formation is generated on (plan D1/D4).

``FormationDomain`` carries the canonical ``z, y, x`` shape, the physical
voxel size, and the rigid placement transform. Primitives evaluate on the
metre coordinates of voxel centres in the *local* frame — the rigid
``local_to_world`` placement is output metadata, exactly as in every other
generator — so anisotropic voxels and metre-denominated feature sizes are
handled once, here.
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from vdbmat_utils.core.errors import ConfigError

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# Size-guard defaults (plan D2). Exceeding them is an explicit error, not a
# slow run; phase 5 owns scale.
MAX_AXIS_CELLS = 256
MAX_TOTAL_CELLS = 16_000_000


@dataclasses.dataclass(frozen=True)
class FormationDomain:
    """A dense generation grid in canonical ``z, y, x`` order."""

    shape_zyx: tuple[int, int, int]
    voxel_size_xyz_m: tuple[float, float, float]
    local_to_world: tuple[tuple[float, ...], ...] = _IDENTITY
    max_axis_cells: int = MAX_AXIS_CELLS
    max_total_cells: int = MAX_TOTAL_CELLS

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(
            isinstance(cells, bool) or not isinstance(cells, int) or cells < 1
            for cells in self.shape_zyx
        ):
            raise ConfigError(
                f"shape_zyx must be 3 positive integers, got {self.shape_zyx!r}"
            )
        if len(self.voxel_size_xyz_m) != 3 or any(
            not (float(size) > 0) for size in self.voxel_size_xyz_m
        ):
            raise ConfigError(
                "voxel_size_xyz_m must be 3 positive components, got "
                f"{self.voxel_size_xyz_m!r}"
            )
        for axis, cells in zip("zyx", self.shape_zyx, strict=True):
            if cells > self.max_axis_cells:
                raise ConfigError(
                    f"size guard: axis {axis} has {cells} cells, exceeding "
                    f"max_axis_cells {self.max_axis_cells}; use a smaller domain"
                )
        total = self.shape_zyx[0] * self.shape_zyx[1] * self.shape_zyx[2]
        if total > self.max_total_cells:
            raise ConfigError(
                f"size guard: domain has {total} cells, exceeding "
                f"max_total_cells {self.max_total_cells}; use a smaller domain"
            )

    def coordinates_xyz_m(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Voxel-centre metre coordinates as broadcastable ``z, y, x`` arrays.

        Returns ``(x, y, z)`` where ``x`` has shape ``(1, 1, nx)``, ``y``
        ``(1, ny, 1)``, and ``z`` ``(nz, 1, 1)``; broadcasting them together
        yields the full grid without materializing three dense volumes. The
        centre of cell ``i`` along an axis sits at ``(i + 0.5) * voxel_size``.
        """
        nz, ny, nx = self.shape_zyx
        sx, sy, sz = (float(size) for size in self.voxel_size_xyz_m)
        x = ((np.arange(nx, dtype=np.float64) + 0.5) * sx).reshape(1, 1, nx)
        y = ((np.arange(ny, dtype=np.float64) + 0.5) * sy).reshape(1, ny, 1)
        z = ((np.arange(nz, dtype=np.float64) + 0.5) * sz).reshape(nz, 1, 1)
        return x, y, z
