"""Narrow, repository-owned STL reader (ported from ``vbdmat.io.mesh``).

Parses ASCII or binary STL into a raw triangle soup. Topology inspection and
voxelization semantics live in ``mesh.voxelizer``; this module only turns
bytes into triangle coordinates and rejects malformed payloads. Binary
detection is by exact length match of the declared triangle count — a leading
``solid`` token marks ASCII only when the binary layout does not fit.
"""

import hashlib
import struct
from pathlib import Path

import numpy as np

from vdbmat_utils.mesh import MeshReadError
from vdbmat_utils.mesh.types import TriangleMesh

_BINARY_HEADER_BYTES = 80
_BINARY_COUNT_BYTES = 4
_BINARY_TRIANGLE_BYTES = 50  # 12 float32 + 1 uint16 attribute byte count

_BINARY_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def load_mesh(path: str | Path) -> TriangleMesh:
    """Read an ASCII or binary STL file into a :class:`TriangleMesh`."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except FileNotFoundError as error:
        raise MeshReadError("mesh", f"file not found: {file_path}") from error
    except OSError as error:
        raise MeshReadError("mesh", f"cannot read {file_path}: {error}") from error
    return read_stl_bytes(data)


def read_stl_bytes(data: bytes) -> TriangleMesh:
    """Parse STL bytes, auto-detecting the binary or ASCII encoding."""
    digest = hashlib.sha256(data).hexdigest()
    if _looks_like_binary(data):
        return _read_binary_stl(data, digest)
    return _read_ascii_stl(data, digest)


def _looks_like_binary(data: bytes) -> bool:
    if len(data) < _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES:
        return False
    count = struct.unpack_from("<I", data, _BINARY_HEADER_BYTES)[0]
    expected = (
        _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES + count * _BINARY_TRIANGLE_BYTES
    )
    if len(data) == expected:
        return True
    # A leading "solid" token is the ASCII marker only when the size does not
    # match the exact binary layout above.
    return not data[:5].lstrip().lower().startswith(b"solid")


def _read_binary_stl(data: bytes, digest: str) -> TriangleMesh:
    count = struct.unpack_from("<I", data, _BINARY_HEADER_BYTES)[0]
    expected = (
        _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES + count * _BINARY_TRIANGLE_BYTES
    )
    if len(data) != expected:
        raise MeshReadError(
            "mesh",
            f"binary STL length {len(data)} does not match {count} triangles",
        )
    if count == 0:
        return TriangleMesh(np.empty((0, 3, 3), dtype=np.float64), digest)
    records = np.frombuffer(
        data,
        dtype=_BINARY_RECORD,
        count=count,
        offset=_BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES,
    )
    vertices = records["vertices"].astype(np.float64)
    return TriangleMesh(vertices, digest)


def _read_ascii_stl(data: bytes, digest: str) -> TriangleMesh:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise MeshReadError("mesh", "ASCII STL must be ASCII text") from error

    vertices: list[tuple[float, float, float]] = []
    tokens = text.split()
    index = 0
    length = len(tokens)
    while index < length:
        if tokens[index] == "vertex":
            if index + 3 >= length:
                raise MeshReadError("mesh", "truncated vertex in ASCII STL")
            try:
                vertex = (
                    float(tokens[index + 1]),
                    float(tokens[index + 2]),
                    float(tokens[index + 3]),
                )
            except ValueError as error:
                raise MeshReadError(
                    "mesh", "non-numeric vertex coordinate in ASCII STL"
                ) from error
            vertices.append(vertex)
            index += 4
        else:
            index += 1

    if len(vertices) % 3 != 0:
        raise MeshReadError(
            "mesh", f"ASCII STL vertex count {len(vertices)} is not a multiple of 3"
        )
    if not vertices:
        return TriangleMesh(np.empty((0, 3, 3), dtype=np.float64), digest)
    array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    return TriangleMesh(array, digest)
