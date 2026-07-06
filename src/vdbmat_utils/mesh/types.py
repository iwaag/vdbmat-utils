"""Triangle-soup mesh type (ported from the recovered ``RawMesh``)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vdbmat_utils.mesh import MeshReadError


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """An unwelded triangle soup in the STL's own source units.

    ``source_sha256`` is the hex SHA-256 of the STL bytes the mesh was parsed
    from (set by the loader); it feeds provenance ``sources`` and the asset
    identity per plan D6.
    """

    triangles: npt.NDArray[np.float64]  # shape (m, 3, 3): [triangle, vertex, xyz]
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.triangles, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise MeshReadError("triangles", "must have shape (m, 3, 3)")
        if not np.isfinite(array).all():
            raise MeshReadError("triangles", "vertex coordinates must be finite")
        object.__setattr__(self, "triangles", array)

    @property
    def triangle_count(self) -> int:
        """Return the number of triangles."""
        return int(self.triangles.shape[0])
