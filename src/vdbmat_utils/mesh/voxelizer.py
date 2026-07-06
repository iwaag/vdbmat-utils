"""Dense, cell-centre reference voxelization for watertight single solids.

Algorithm ported without change from the recovered ``vbdmat.voxelize.mesh``
(commit ``8f55562``; historical ADR-0006): topology inspection, deterministic
domain construction, and a signed-winding +X ray classification of cell
centres. The debugged numerical constants (jitter split, snap epsilon, facing
mask) are preserved verbatim. Correctness and inspectability are favoured
over speed; the configurable cell bounds keep the dense method tractable.
"""

import dataclasses
from collections.abc import Mapping
from math import ceil, floor
from typing import cast

import numpy as np
import numpy.typing as npt
from vdbmat.core import (
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialRole,
)
from vdbmat.core.transforms import IDENTITY_MATRIX_4, Matrix4

from vdbmat_utils.core import (
    GeneratorConfig,
    build_material_label_volume,
    build_provenance,
)
from vdbmat_utils.mesh import MeshTopologyError, VoxelizationError
from vdbmat_utils.mesh.types import TriangleMesh

GENERATOR = "vdbmat-utils.mesh.voxelize"
GENERATOR_VERSION = "0.1.0"

_UNIT_TO_METRES = {"m": 1.0, "mm": 1.0e-3}
#: Source units the mesh voxelizer accepts, exposed for config-time validation.
SUPPORTED_MESH_UNITS: tuple[str, ...] = tuple(sorted(_UNIT_TO_METRES))
_BARYCENTRIC_TOLERANCE = 1e-9
_SURFACE_TOLERANCE_M = 1e-9
# Distinct sub-voxel Y/Z offsets (fractions of a voxel). They must differ so
# that cell centres with equal Y and Z do not stay on a 45-degree
# triangulation diagonal.
_SAMPLE_JITTER_Y = 7.3e-5
_SAMPLE_JITTER_Z = 3.1e-5
# A span landing within this fraction of a cell of an integer is treated as
# exactly that many cells, so float32 STL round-off does not add a spurious
# padded cell and equivalent unit expressions of one solid yield the same grid.
_DOMAIN_SNAP_EPS = 1e-6

_DEFAULT_BACKGROUND: dict[str, object] = {
    "material_id": 0,
    "name": "air",
    "role": "background",
}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MeshVoxelizeConfig(GeneratorConfig):
    """Configuration for the mesh workflow.

    ``source_unit`` is required with no default: STL is unitless and guessing
    units is how prints come out 1000x too small. ``material`` declares the
    single foreground material (``material_id``, ``name``, ``role``);
    ``background`` defaults to air at id 0. ``domain_min_m``/``domain_max_m``
    (both or neither) override the auto-fitted mesh AABB; no padding is added
    to explicit bounds. ``seed`` is inherited, unused, and reserved.
    """

    source_unit: str
    voxel_size_xyz_m: tuple[float, float, float]
    material: Mapping[str, object]
    background: Mapping[str, object] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_BACKGROUND)
    )
    domain_min_m: tuple[float, float, float] | None = None
    domain_max_m: tuple[float, float, float] | None = None
    padding_cells: int = 1
    placement: tuple[tuple[float, ...], ...] | None = None
    max_axis_cells: int = 128
    max_total_cells: int = 2_000_000


@dataclasses.dataclass(frozen=True, slots=True)
class VoxelizationDiagnostics:
    """Structured, machine-readable voxelization findings."""

    triangle_count: int
    bounds_min_xyz_m: tuple[float, float, float]
    bounds_max_xyz_m: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    occupied_cells: int
    material_id: int


@dataclasses.dataclass(frozen=True, slots=True)
class VoxelizationResult:
    """A canonical label volume plus its voxelization diagnostics."""

    volume: MaterialLabelVolume
    diagnostics: VoxelizationDiagnostics


def voxelize_mesh(
    mesh: TriangleMesh, config: MeshVoxelizeConfig
) -> VoxelizationResult:
    """Voxelize a watertight single-solid mesh into a label volume."""
    factor = _unit_factor(config.source_unit)
    voxel_size = _voxel_size(config.voxel_size_xyz_m)
    material_id = _interior_material_id(config.material)
    if not isinstance(config.padding_cells, int) or config.padding_cells < 0:
        raise VoxelizationError("padding_cells", "must be a non-negative integer")

    inspect_topology(mesh)
    triangles_m = mesh.triangles * factor

    origin, shape_zyx = _domain(triangles_m, voxel_size, config)
    placement = (
        IDENTITY_MATRIX_4 if config.placement is None else _matrix4(config.placement)
    )
    local_to_world = _compose_placement(placement, origin)

    label = _classify(triangles_m, origin, voxel_size, shape_zyx, material_id)
    occupied = int(np.count_nonzero(label))

    sources: tuple[str, ...] = (
        (f"sha256:{mesh.source_sha256}",)
        if mesh.source_sha256 is not None
        else ()
    )
    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=sources,
        notes=(
            f"dense cell-centre voxelization; source_unit={config.source_unit}; "
            f"padding_cells={config.padding_cells}; "
            f"triangles={mesh.triangle_count}"
        ),
    )
    volume = build_material_label_volume(
        material_id=label,
        voxel_size_xyz_m=voxel_size,
        palette=_palette(config),
        provenance=provenance,
        local_to_world=local_to_world,
    )

    vertices = triangles_m.reshape(-1, 3)
    bounds_min = cast(
        "tuple[float, float, float]", tuple(vertices.min(axis=0).tolist())
    )
    bounds_max = cast(
        "tuple[float, float, float]", tuple(vertices.max(axis=0).tolist())
    )
    diagnostics = VoxelizationDiagnostics(
        triangle_count=mesh.triangle_count,
        bounds_min_xyz_m=bounds_min,
        bounds_max_xyz_m=bounds_max,
        shape_zyx=shape_zyx,
        occupied_cells=occupied,
        material_id=material_id,
    )
    return VoxelizationResult(volume=volume, diagnostics=diagnostics)


def inspect_topology(mesh: TriangleMesh) -> None:
    """Reject any mesh that is not a watertight, consistently oriented
    single solid."""
    triangles = mesh.triangles
    count = int(triangles.shape[0])
    if count == 0:
        raise MeshTopologyError("mesh", "mesh has no triangles")

    scale = float(np.max(np.abs(triangles))) if triangles.size else 1.0
    tol = max(scale, 1.0) * 1e-9
    vertices = triangles.reshape(-1, 3)
    keys = np.round(vertices / tol).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    welded = np.asarray(inverse).reshape(-1)[: count * 3].reshape(count, 3)

    # Degenerate faces: repeated welded vertex or near-zero area.
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    area_eps = (max(scale, 1.0) ** 2) * 1e-18
    for tri_index in range(count):
        a, b, c = (int(value) for value in welded[tri_index])
        if len({a, b, c}) != 3:
            raise MeshTopologyError(
                f"triangle[{tri_index}]", "has a repeated (degenerate) vertex"
            )
        if float(areas[tri_index]) <= area_eps:
            raise MeshTopologyError(
                f"triangle[{tri_index}]", "has near-zero area (degenerate)"
            )

    directed: dict[tuple[int, int], int] = {}
    undirected: dict[tuple[int, int], int] = {}
    parent: list[int] = list(range(count))
    edge_owner: dict[tuple[int, int], int] = {}

    for tri_index in range(count):
        a, b, c = (int(value) for value in welded[tri_index])
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
            key = (u, v) if u < v else (v, u)
            undirected[key] = undirected.get(key, 0) + 1
            if key in edge_owner:
                _union(parent, edge_owner[key], tri_index)
            else:
                edge_owner[key] = tri_index

    for (u, v), total in undirected.items():
        if total == 1:
            raise MeshTopologyError(
                "mesh", "surface is open / not watertight (an edge borders one face)"
            )
        if total > 2:
            raise MeshTopologyError(
                "mesh",
                "surface is non-manifold (an edge borders more than two faces)",
                count=total,
            )
        forward = directed.get((u, v), 0)
        backward = directed.get((v, u), 0)
        if forward != 1 or backward != 1:
            raise MeshTopologyError(
                "mesh",
                "inconsistent triangle orientation across a shared edge",
            )

    roots = {_find(parent, index) for index in range(count)}
    if len(roots) != 1:
        raise MeshTopologyError(
            "mesh",
            "mesh contains multiple disconnected solids",
            count=len(roots),
        )


def _classify(
    triangles_m: npt.NDArray[np.float64],
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
    material_id: int,
) -> npt.NDArray[np.uint16]:
    nz, ny, nx = shape_zyx
    sx, sy, sz = voxel_size
    ox, oy, oz = origin

    x0 = triangles_m[:, 0, 0]
    y0 = triangles_m[:, 0, 1]
    z0 = triangles_m[:, 0, 2]
    x1 = triangles_m[:, 1, 0]
    y1 = triangles_m[:, 1, 1]
    z1 = triangles_m[:, 1, 2]
    x2 = triangles_m[:, 2, 0]
    y2 = triangles_m[:, 2, 1]
    z2 = triangles_m[:, 2, 2]

    denom = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
    denom_scale = float(np.max(np.abs(denom))) if denom.size else 0.0
    area_eps = denom_scale * 1e-12
    facing = np.abs(denom) > area_eps
    sign = np.sign(denom)
    safe_denom = np.where(facing, denom, 1.0)

    # Deterministic sub-voxel offset of the YZ sample point used by the
    # winding ray (historical ADR-006 D8). It only perturbs a centre off an
    # interior triangulation diagonal (a non-physical shared edge); real
    # surfaces are at least half a voxel from any cell centre for well-posed
    # inputs, so classification is unchanged. The surface test below uses the
    # unjittered centres.
    jitter_y = sy * _SAMPLE_JITTER_Y
    jitter_z = sz * _SAMPLE_JITTER_Z

    xc = ox + (np.arange(nx) + 0.5) * sx
    label = np.zeros((nz, ny, nx), dtype=np.uint16)

    for k in range(nz):
        centre_z = oz + (k + 0.5) * sz
        zc = centre_z + jitter_z
        for j in range(ny):
            centre_y = oy + (j + 0.5) * sy
            yc = centre_y + jitter_y
            centres = np.column_stack(
                (
                    xc,
                    np.full(nx, centre_y, dtype=np.float64),
                    np.full(nx, centre_z, dtype=np.float64),
                )
            )
            on_surface = _points_on_surface(centres, triangles_m)
            w1 = ((yc - y0) * (z2 - z0) - (y2 - y0) * (zc - z0)) / safe_denom
            w2 = ((y1 - y0) * (zc - z0) - (yc - y0) * (z1 - z0)) / safe_denom
            w0 = 1.0 - w1 - w2
            inside = (
                facing
                & (w0 >= -_BARYCENTRIC_TOLERANCE)
                & (w1 >= -_BARYCENTRIC_TOLERANCE)
                & (w2 >= -_BARYCENTRIC_TOLERANCE)
            )
            if not np.any(inside):
                if np.any(on_surface):
                    label[k, j, on_surface] = material_id
                continue
            x_int = w0 * x0 + w1 * x1 + w2 * x2
            xi = x_int[inside]
            si = sign[inside]
            ahead = xi[None, :] > xc[:, None]
            winding = (ahead * si[None, :]).sum(axis=1)
            occupied = (np.abs(winding) >= 0.5) | on_surface
            if np.any(occupied):
                label[k, j, occupied] = material_id
    return label


def _points_on_surface(
    points: npt.NDArray[np.float64], triangles: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Return points lying on any triangle under the closed-solid rule.

    Known hotspot; deliberately kept as ported (benchmark before optimizing).
    """
    result = np.zeros(points.shape[0], dtype=np.bool_)
    for triangle in triangles:
        origin = triangle[0]
        edge_u = triangle[1] - origin
        edge_v = triangle[2] - origin
        normal = np.cross(edge_u, edge_v)
        normal_length = float(np.linalg.norm(normal))
        relative = points - origin
        plane_distance = np.abs(relative @ normal) / normal_length
        candidates = plane_distance <= _SURFACE_TOLERANCE_M
        if not np.any(candidates):
            continue

        dot_uu = float(edge_u @ edge_u)
        dot_uv = float(edge_u @ edge_v)
        dot_vv = float(edge_v @ edge_v)
        denominator = dot_uu * dot_vv - dot_uv * dot_uv
        candidate_relative = relative[candidates]
        dot_pu = candidate_relative @ edge_u
        dot_pv = candidate_relative @ edge_v
        bary_u = (dot_vv * dot_pu - dot_uv * dot_pv) / denominator
        bary_v = (dot_uu * dot_pv - dot_uv * dot_pu) / denominator
        on_triangle = (
            (bary_u >= -_BARYCENTRIC_TOLERANCE)
            & (bary_v >= -_BARYCENTRIC_TOLERANCE)
            & (bary_u + bary_v <= 1.0 + _BARYCENTRIC_TOLERANCE)
        )
        candidate_indices = np.flatnonzero(candidates)
        result[candidate_indices[on_triangle]] = True
        if np.all(result):
            break
    return result


def _domain(
    triangles_m: npt.NDArray[np.float64],
    voxel_size: tuple[float, float, float],
    config: MeshVoxelizeConfig,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    explicit = _explicit_bounds(config)
    origin: list[float] = []
    extents_xyz: list[int] = []
    if explicit is None:
        vertices = triangles_m.reshape(-1, 3)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        for axis in range(3):
            size = voxel_size[axis]
            base = floor(float(minimum[axis]) / size + _DOMAIN_SNAP_EPS) * size
            span = float(maximum[axis]) - base
            cells = max(1, ceil(span / size - _DOMAIN_SNAP_EPS))
            base -= config.padding_cells * size
            cells += 2 * config.padding_cells
            origin.append(base)
            extents_xyz.append(cells)
    else:
        domain_min, domain_max = explicit
        for axis in range(3):
            size = voxel_size[axis]
            span = domain_max[axis] - domain_min[axis]
            origin.append(domain_min[axis])
            extents_xyz.append(max(1, ceil(span / size - _DOMAIN_SNAP_EPS)))

    total = extents_xyz[0] * extents_xyz[1] * extents_xyz[2]
    if any(cells > config.max_axis_cells for cells in extents_xyz):
        raise VoxelizationError(
            "voxel_size",
            f"grid extent {tuple(extents_xyz)} exceeds the axis bound of "
            f"{config.max_axis_cells} cells; use a coarser voxel size",
        )
    if total > config.max_total_cells:
        raise VoxelizationError(
            "voxel_size",
            f"grid has {total} cells, exceeding the bound of "
            f"{config.max_total_cells}; use a coarser voxel size",
        )

    shape_zyx = (extents_xyz[2], extents_xyz[1], extents_xyz[0])
    origin_xyz = (origin[0], origin[1], origin[2])
    return origin_xyz, shape_zyx


def _explicit_bounds(
    config: MeshVoxelizeConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if config.domain_min_m is None and config.domain_max_m is None:
        return None
    if config.domain_min_m is None or config.domain_max_m is None:
        raise VoxelizationError(
            "domain_min_m",
            "domain_min_m and domain_max_m must be given together or not at all",
        )
    minimum = tuple(float(v) for v in config.domain_min_m)
    maximum = tuple(float(v) for v in config.domain_max_m)
    if len(minimum) != 3 or len(maximum) != 3:
        raise VoxelizationError(
            "domain_min_m", "domain bounds must contain exactly 3 numbers each"
        )
    if any(hi <= lo for lo, hi in zip(minimum, maximum, strict=True)):
        raise VoxelizationError(
            "domain_max_m", "must be strictly greater than domain_min_m per axis"
        )
    return (minimum, maximum)


def _compose_placement(
    placement: Matrix4, origin: tuple[float, float, float]
) -> Matrix4:
    placement_matrix = np.asarray(placement, dtype=np.float64)
    translation = np.eye(4, dtype=np.float64)
    translation[0, 3] = origin[0]
    translation[1, 3] = origin[1]
    translation[2, 3] = origin[2]
    composed = placement_matrix @ translation
    return cast(
        Matrix4, tuple(tuple(float(value) for value in row) for row in composed)
    )


def _matrix4(rows: tuple[tuple[float, ...], ...]) -> Matrix4:
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise VoxelizationError("placement", "must be a 4x4 matrix")
    return cast(
        Matrix4, tuple(tuple(float(value) for value in row) for row in rows)
    )


def _material_definition(
    block: Mapping[str, object], field: str
) -> MaterialDefinition:
    if not isinstance(block, Mapping):
        raise VoxelizationError(field, "must be an object")
    unknown = sorted(set(block) - {"material_id", "name", "role"})
    if unknown:
        raise VoxelizationError(field, f"unknown fields: {unknown}")
    material_id = block.get("material_id")
    name = block.get("name")
    role = block.get("role")
    if not isinstance(material_id, int) or isinstance(material_id, bool):
        raise VoxelizationError(f"{field}.material_id", "must be an integer")
    if not isinstance(name, str):
        raise VoxelizationError(f"{field}.name", "must be a string")
    if not isinstance(role, str):
        raise VoxelizationError(f"{field}.role", "must be a string")
    try:
        return MaterialDefinition(
            material_id=material_id, name=name, role=MaterialRole(role)
        )
    except (TypeError, ValueError) as error:
        raise VoxelizationError(field, str(error)) from error


def _palette(config: MeshVoxelizeConfig) -> tuple[MaterialDefinition, ...]:
    background = _material_definition(config.background, "background")
    material = _material_definition(config.material, "material")
    if background.role is not MaterialRole.BACKGROUND:
        raise VoxelizationError("background.role", "must be 'background'")
    return (background, material)


def _interior_material_id(material: Mapping[str, object]) -> int:
    definition = _material_definition(material, "material")
    if definition.role is not MaterialRole.MATERIAL:
        raise VoxelizationError("material.role", "must be 'material'")
    if not 1 <= definition.material_id <= 65535:
        raise VoxelizationError(
            "material.material_id",
            "must be a non-background material ID in [1, 65535]",
        )
    return definition.material_id


def _unit_factor(source_unit: str) -> float:
    if source_unit not in _UNIT_TO_METRES:
        raise VoxelizationError(
            "source_unit",
            f"must be one of {sorted(_UNIT_TO_METRES)}, got {source_unit!r}",
        )
    return _UNIT_TO_METRES[source_unit]


def _voxel_size(
    voxel_size_xyz_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    values = tuple(voxel_size_xyz_m)
    if len(values) != 3:
        raise VoxelizationError("voxel_size_xyz_m", "must contain exactly 3 numbers")
    result: list[float] = []
    for axis, item in zip(("x", "y", "z"), values, strict=True):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VoxelizationError(f"voxel_size_xyz_m.{axis}", "must be a number")
        value = float(item)
        if not np.isfinite(value) or value <= 0.0:
            raise VoxelizationError(
                f"voxel_size_xyz_m.{axis}", "must be finite and greater than zero"
            )
        result.append(value)
    return (result[0], result[1], result[2])


def _union(parent: list[int], left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _find(parent: list[int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node
