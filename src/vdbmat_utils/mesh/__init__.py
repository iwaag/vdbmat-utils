"""Watertight STL meshes → canonical material-label volumes.

Port of the mesh path recovered from vdbmat git history (commit ``8f55562``,
modules ``vbdmat.io.mesh`` and ``vbdmat.voxelize.mesh``; design rationale in
the historical ADR-0006). The voxelization semantics — dense cell-centre
classification by signed winding number along a +X ray, closed-solid surface
rule, debugged jitter/snap constants — are adopted wholesale; only the
packaging (config dataclass, shared builders, error hierarchy) is new.
No third-party mesh dependency is used.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class MeshReadError(VdbmatUtilsError):
    """An STL payload cannot be parsed as a triangle mesh."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


class MeshTopologyError(VdbmatUtilsError):
    """A mesh violates the watertight single-solid contract."""

    def __init__(
        self, field_path: str, message: str, *, count: int | None = None
    ) -> None:
        self.field_path = field_path
        self.message = message
        self.count = count
        suffix = f" (count={count})" if count is not None else ""
        super().__init__(f"{field_path}: {message}{suffix}")


class VoxelizationError(VdbmatUtilsError):
    """A voxelization argument or domain violates the contract."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


from .loader import load_mesh, read_stl_bytes  # noqa: E402
from .types import TriangleMesh  # noqa: E402
from .voxelizer import (  # noqa: E402
    MeshVoxelizeConfig,
    VoxelizationDiagnostics,
    VoxelizationResult,
    inspect_topology,
    voxelize_mesh,
)

__all__ = [
    "MeshReadError",
    "MeshTopologyError",
    "MeshVoxelizeConfig",
    "TriangleMesh",
    "VoxelizationDiagnostics",
    "VoxelizationError",
    "VoxelizationResult",
    "inspect_topology",
    "load_mesh",
    "read_stl_bytes",
    "voxelize_mesh",
]
