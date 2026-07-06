"""Contract tests for the mesh workflow.

Determinism (byte-equal double run through the CLI), checksum stability
(hardcoded payload/manifest digests — a change means the output contract
moved and must be reviewed deliberately), and three-axis ASCII orientation
goldens on the asymmetric L-bracket so any z/y/x confusion in the voxelizer
changes visible text.
"""

import hashlib
from pathlib import Path
from typing import Literal

import pytest
from vdbmat.io import read_material_label_manifest

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import write_mesh_fixture
from vdbmat_utils.preview import slice_ascii

GOLDEN_PAYLOAD_SHA256 = (
    "895617011bff96d0400458cc67b2d4439f745bdd7d6654a15328f7ad911cb2ed"
)
GOLDEN_MANIFEST_SHA256 = (
    "31746580eea5e0e1b45038f9751b49c62e1098c9392845fb4cb5ffcc515f857b"
)

# Middle slice of each axis of the voxelized L-bracket (8x6x4 grid in x/y/z,
# one padding cell on every side). The bracket is 3x2x1 mm with the notch at
# high y / high x, so all three views differ from each other and from any of
# their own transposes.
GOLDEN_SLICES: dict[Literal["z", "y", "x"], str] = {
    "z": (
        "slice z=2  +x →  +y ↓\n"
        "........\n"
        ".111111.\n"
        ".111111.\n"
        ".11.....\n"
        ".11.....\n"
        "........"
    ),
    "y": (
        "slice y=3  +x →  +z ↓\n"
        "........\n"
        ".11.....\n"
        ".11.....\n"
        "........"
    ),
    "x": (
        "slice x=4  +y →  +z ↓\n"
        "......\n"
        ".11...\n"
        ".11...\n"
        "......"
    ),
}


def _run_cli(tmp_path: Path, out_name: str) -> Path:
    mesh_path, config = write_mesh_fixture(tmp_path / "mesh")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / out_name
    assert main(
        [
            "voxelize-mesh",
            str(mesh_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "bracket",
        ]
    ) == 0
    return out_dir


def test_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_cli(tmp_path, "a")
    second = _run_cli(tmp_path, "b")
    for filename in ("bracket.voxels.json", "bracket.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_golden_digests(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, "out")
    payload_sha = hashlib.sha256(
        (out_dir / "bracket.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "bracket.voxels.json").read_bytes()
    ).hexdigest()
    assert payload_sha == GOLDEN_PAYLOAD_SHA256
    assert manifest_sha == GOLDEN_MANIFEST_SHA256


def test_orientation_goldens(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, "out")
    volume = read_material_label_manifest(out_dir / "bracket.voxels.json")
    assert volume.geometry.shape_zyx == (4, 6, 8)
    for axis, golden in GOLDEN_SLICES.items():
        index = volume.geometry.shape_zyx[("z", "y", "x").index(axis)] // 2
        assert slice_ascii(volume, axis, index) == golden, f"axis {axis}"


def test_validate_and_identity(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, "out")
    manifest = out_dir / "bracket.voxels.json"
    assert main(["validate", str(manifest)]) == 0
    assert "sha256:" in manifest.read_text(encoding="utf-8")


def test_open_mesh_fails_with_single_line_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    open_stl = (
        "solid open\n"
        "  facet normal 0 0 0\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid open\n"
    )
    mesh_path = tmp_path / "open.stl"
    mesh_path.write_text(open_stl, encoding="ascii")
    _, config = write_mesh_fixture(tmp_path / "fixture")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    assert main(
        [
            "voxelize-mesh",
            str(mesh_path),
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "out"),
            "--name",
            "open",
        ]
    ) == 1
    stderr = capsys.readouterr().err
    assert stderr.startswith("error: ")
    assert stderr.count("\n") == 1
    assert "watertight" in stderr
