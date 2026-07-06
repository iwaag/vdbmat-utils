"""Tests for the STL loader (ported from the recovered vdbmat suite)."""

import struct
from pathlib import Path

import numpy as np
import pytest

from vdbmat_utils.mesh import MeshReadError, load_mesh, read_stl_bytes

Vertex = tuple[float, float, float]
_TRIANGLE: tuple[Vertex, Vertex, Vertex] = ((0, 0, 0), (1, 0, 0), (0, 1, 0))


def _binary_stl(triangles: list[tuple[Vertex, Vertex, Vertex]]) -> bytes:
    data = struct.pack("<80sI", b"", len(triangles))
    for tri in triangles:
        data += struct.pack("<12fH", 0.0, 0.0, 0.0, *tri[0], *tri[1], *tri[2], 0)
    return data


def _ascii_stl(triangles: list[tuple[Vertex, Vertex, Vertex]]) -> bytes:
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


def test_binary_and_ascii_agree() -> None:
    binary = read_stl_bytes(_binary_stl([_TRIANGLE]))
    ascii_mesh = read_stl_bytes(_ascii_stl([_TRIANGLE]))
    assert binary.triangle_count == ascii_mesh.triangle_count == 1
    np.testing.assert_allclose(binary.triangles, ascii_mesh.triangles)


def test_binary_starting_with_solid_is_detected_by_length() -> None:
    # A binary file whose 80-byte header begins with "solid" must still parse
    # as binary because the length matches the declared triangle count exactly.
    data = bytearray(_binary_stl([_TRIANGLE]))
    data[:5] = b"solid"
    mesh = read_stl_bytes(bytes(data))
    assert mesh.triangle_count == 1


def test_truncated_binary_stl_is_rejected() -> None:
    data = _binary_stl([_TRIANGLE, _TRIANGLE])
    with pytest.raises(MeshReadError):
        read_stl_bytes(data[:-10])


def test_ascii_non_numeric_vertex_is_rejected() -> None:
    bad = _ascii_stl([_TRIANGLE]).replace(b"vertex 0 0 0", b"vertex x 0 0")
    with pytest.raises(MeshReadError, match="non-numeric"):
        read_stl_bytes(bad)


def test_ascii_truncated_vertex_is_rejected() -> None:
    with pytest.raises(MeshReadError, match="truncated"):
        read_stl_bytes(b"solid t\nvertex 1 2")


def test_ascii_vertex_count_must_be_multiple_of_three() -> None:
    text = b"solid t\nvertex 0 0 0\nvertex 1 0 0\nendsolid t\n"
    with pytest.raises(MeshReadError, match="multiple of 3"):
        read_stl_bytes(text)


def test_empty_meshes_parse_to_zero_triangles() -> None:
    assert read_stl_bytes(_binary_stl([])).triangle_count == 0
    assert read_stl_bytes(b"solid empty\nendsolid empty\n").triangle_count == 0


def test_load_mesh_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MeshReadError, match="file not found"):
        load_mesh(tmp_path / "missing.stl")


def test_source_sha256_recorded(tmp_path: Path) -> None:
    data = _binary_stl([_TRIANGLE])
    path = tmp_path / "tri.stl"
    path.write_bytes(data)
    from_bytes = read_stl_bytes(data)
    from_file = load_mesh(path)
    assert from_bytes.source_sha256 is not None
    assert from_bytes.source_sha256 == from_file.source_sha256
    assert len(from_bytes.source_sha256) == 64
