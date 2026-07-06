"""Tests for the dense reference voxelizer (ported from the recovered vdbmat
suite, plus the additions the plan calls for: explicit domain bounds and
configurable size guards)."""

import struct
from collections.abc import Sequence

import numpy as np
import pytest

from vdbmat_utils.mesh import (
    MeshTopologyError,
    MeshVoxelizeConfig,
    VoxelizationError,
    VoxelizationResult,
    read_stl_bytes,
    voxelize_mesh,
)

Vertex = tuple[float, float, float]

_MATERIAL: dict[str, object] = {
    "material_id": 1,
    "name": "resin",
    "role": "material",
}


def _cube_triangles(lo: Vertex, hi: Vertex) -> list[tuple[Vertex, Vertex, Vertex]]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    # Quads wound counter-clockwise as seen from outside (outward normals).
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    triangles: list[tuple[Vertex, Vertex, Vertex]] = []
    for a, b, c, d in quads:
        triangles.append((corners[a], corners[b], corners[c]))
        triangles.append((corners[a], corners[c], corners[d]))
    return triangles


def _binary_stl(triangles: Sequence[tuple[Vertex, Vertex, Vertex]]) -> bytes:
    data = struct.pack("<80sI", b"", len(triangles))
    for tri in triangles:
        data += struct.pack("<12fH", 0.0, 0.0, 0.0, *tri[0], *tri[1], *tri[2], 0)
    return data


def _ascii_stl(triangles: Sequence[tuple[Vertex, Vertex, Vertex]]) -> bytes:
    lines = ["solid test"]
    for tri in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("outer loop")
        for vertex in tri:
            lines.append(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid test")
    return ("\n".join(lines) + "\n").encode("ascii")


def _config(**overrides: object) -> MeshVoxelizeConfig:
    fields: dict[str, object] = {
        "source_unit": "mm",
        "voxel_size_xyz_m": (0.001, 0.001, 0.001),
        "material": _MATERIAL,
    }
    fields.update(overrides)
    return MeshVoxelizeConfig(**fields)  # type: ignore[arg-type]


def _voxelize(
    triangles: Sequence[tuple[Vertex, Vertex, Vertex]], **overrides: object
) -> VoxelizationResult:
    return voxelize_mesh(read_stl_bytes(_binary_stl(triangles)), _config(**overrides))


def _voxelize_cube(lo: Vertex, hi: Vertex, **overrides: object) -> VoxelizationResult:
    return _voxelize(_cube_triangles(lo, hi), **overrides)


def test_axis_aligned_cube_occupancy() -> None:
    result = _voxelize_cube((0, 0, 0), (3, 3, 3))
    assert result.diagnostics.shape_zyx == (5, 5, 5)
    assert result.diagnostics.occupied_cells == 27
    label = np.asarray(result.volume.material_id)
    # Interior block is exactly padded indices 1..3 on each axis.
    assert label[1:4, 1:4, 1:4].sum() == 27
    assert label.sum() == 27


def test_translated_cube_preserves_count() -> None:
    result = _voxelize_cube((1, 1, 1), (4, 4, 4))
    assert result.diagnostics.occupied_cells == 27


def test_cell_centres_on_mesh_boundary_use_closed_solid_convention() -> None:
    result = _voxelize_cube((0.5, 0.5, 0.5), (2.5, 2.5, 2.5))
    label = np.asarray(result.volume.material_id)
    assert result.diagnostics.occupied_cells == 27
    assert np.all(label[1:4, 1:4, 1:4] == 1)
    assert int(label.sum()) == 27


def test_non_cubic_box_occupancy() -> None:
    result = _voxelize_cube((0, 0, 0), (5, 3, 2))
    assert result.diagnostics.occupied_cells == 5 * 3 * 2


def test_millimetre_and_metre_agree() -> None:
    in_mm = _voxelize_cube((0, 0, 0), (3, 3, 3), source_unit="mm")
    in_m = _voxelize_cube((0.0, 0.0, 0.0), (0.003, 0.003, 0.003), source_unit="m")
    assert in_mm.diagnostics.shape_zyx == in_m.diagnostics.shape_zyx
    assert np.array_equal(
        np.asarray(in_mm.volume.material_id),
        np.asarray(in_m.volume.material_id),
    )


def test_float32_rounding_does_not_add_a_spurious_cell() -> None:
    # 3.0000002 mm survives the float32 STL round-trip slightly above 3.0;
    # without the domain snap the span would gain a fourth cell.
    result = _voxelize_cube((0, 0, 0), (3.0000002, 3, 3))
    assert result.diagnostics.shape_zyx == (5, 5, 5)
    assert result.diagnostics.occupied_cells == 27


def test_anisotropic_voxel_size() -> None:
    result = _voxelize_cube(
        (0, 0, 0), (6, 6, 6), voxel_size_xyz_m=(0.002, 0.001, 0.001)
    )
    # 6 mm cube: 3 cells along X (2 mm), 6 along Y and Z.
    assert result.diagnostics.occupied_cells == 3 * 6 * 6


def test_rigid_rotation_preserves_occupancy_and_moves_world() -> None:
    # 90-degree rotation about Z: (x, y) -> (-y, x).
    rotation = (
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    plain = _voxelize_cube((0, 0, 0), (3, 3, 3))
    rotated = _voxelize_cube((0, 0, 0), (3, 3, 3), placement=rotation)
    assert rotated.diagnostics.occupied_cells == plain.diagnostics.occupied_cells
    # The transform is metadata: the voxel-local array is unchanged.
    assert np.array_equal(
        np.asarray(rotated.volume.material_id),
        np.asarray(plain.volume.material_id),
    )
    plain_world = plain.volume.geometry.cell_center_world((2, 2, 2))
    rotated_world = rotated.volume.geometry.cell_center_world((2, 2, 2))
    assert rotated_world == pytest.approx(
        (-plain_world[1], plain_world[0], plain_world[2])
    )


def test_triangle_reordering_is_invariant() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    a = _voxelize(triangles)
    b = _voxelize(list(reversed(triangles)))
    assert np.array_equal(
        np.asarray(a.volume.material_id), np.asarray(b.volume.material_id)
    )


def test_ascii_and_binary_agree() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    binary = voxelize_mesh(read_stl_bytes(_binary_stl(triangles)), _config())
    ascii_result = voxelize_mesh(read_stl_bytes(_ascii_stl(triangles)), _config())
    assert np.array_equal(
        np.asarray(binary.volume.material_id),
        np.asarray(ascii_result.volume.material_id),
    )


def test_diagnostics_bounds_and_provenance() -> None:
    result = _voxelize_cube((0, 0, 0), (3, 3, 3))
    assert result.diagnostics.triangle_count == 12
    assert result.diagnostics.bounds_min_xyz_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.diagnostics.bounds_max_xyz_m == pytest.approx(
        (0.003, 0.003, 0.003)
    )
    provenance = result.volume.provenance
    assert provenance.generator == "vdbmat-utils.mesh.voxelize"
    assert provenance.sources[0].startswith("sha256:")
    assert provenance.configuration_digest is not None


def test_explicit_domain_bounds_override_autofit() -> None:
    auto = _voxelize_cube((0, 0, 0), (3, 3, 3))
    explicit = _voxelize_cube(
        (0, 0, 0),
        (3, 3, 3),
        domain_min_m=(-0.001, -0.001, -0.001),
        domain_max_m=(0.004, 0.004, 0.004),
    )
    assert explicit.diagnostics.shape_zyx == (5, 5, 5)
    assert np.array_equal(
        np.asarray(explicit.volume.material_id),
        np.asarray(auto.volume.material_id),
    )


def test_domain_bounds_must_come_together() -> None:
    with pytest.raises(VoxelizationError, match="together"):
        _voxelize_cube((0, 0, 0), (3, 3, 3), domain_min_m=(0.0, 0.0, 0.0))


def test_inverted_domain_bounds_rejected() -> None:
    with pytest.raises(VoxelizationError, match="strictly greater"):
        _voxelize_cube(
            (0, 0, 0),
            (3, 3, 3),
            domain_min_m=(0.0, 0.0, 0.0),
            domain_max_m=(0.001, 0.001, 0.0),
        )


def test_open_mesh_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))[:-1]
    with pytest.raises(MeshTopologyError) as info:
        _voxelize(triangles)
    assert "open" in str(info.value).lower()


def test_non_manifold_mesh_is_rejected() -> None:
    # Add a duplicate face so one edge borders three triangles.
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles.append(triangles[0])
    with pytest.raises(MeshTopologyError):
        _voxelize(triangles)


def test_degenerate_triangle_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles[0] = ((0, 0, 0), (0, 0, 0), (1, 0, 0))
    with pytest.raises(MeshTopologyError):
        _voxelize(triangles)


def test_empty_mesh_is_rejected() -> None:
    with pytest.raises(MeshTopologyError):
        _voxelize([])


def test_multi_solid_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles += _cube_triangles((10, 10, 10), (13, 13, 13))
    with pytest.raises(MeshTopologyError) as info:
        _voxelize(triangles)
    assert "disconnected" in str(info.value).lower()


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(VoxelizationError) as info:
        _voxelize_cube((0, 0, 0), (3, 3, 3), source_unit="inch")
    assert info.value.field_path == "source_unit"


def test_background_material_id_is_rejected() -> None:
    with pytest.raises(VoxelizationError) as info:
        _voxelize_cube(
            (0, 0, 0),
            (3, 3, 3),
            material={"material_id": 0, "name": "x", "role": "material"},
        )
    assert info.value.field_path == "material.material_id"


def test_background_role_is_enforced() -> None:
    with pytest.raises(VoxelizationError):
        _voxelize_cube(
            (0, 0, 0),
            (3, 3, 3),
            background={"material_id": 0, "name": "air", "role": "material"},
        )


def test_axis_cell_bound_is_enforced() -> None:
    with pytest.raises(VoxelizationError) as info:
        _voxelize_cube((0, 0, 0), (200, 1, 1))
    assert info.value.field_path == "voxel_size"


def test_size_guards_are_configurable() -> None:
    with pytest.raises(VoxelizationError, match="axis bound of 4"):
        _voxelize_cube((0, 0, 0), (3, 3, 3), max_axis_cells=4)
    with pytest.raises(VoxelizationError, match="exceeding the bound of 100"):
        _voxelize_cube((0, 0, 0), (3, 3, 3), max_total_cells=100)


def test_seed_is_reserved_and_digested() -> None:
    a = _voxelize_cube((0, 0, 0), (3, 3, 3))
    b = _voxelize_cube((0, 0, 0), (3, 3, 3), seed=7)
    assert np.array_equal(
        np.asarray(a.volume.material_id), np.asarray(b.volume.material_id)
    )
    assert (
        a.volume.provenance.configuration_digest
        != b.volume.provenance.configuration_digest
    )
