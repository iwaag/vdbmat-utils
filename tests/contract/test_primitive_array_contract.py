"""Contract tests for the primitive-array generator.

Determinism (byte-equal double run through the CLI), checksum stability
(hardcoded payload/manifest digests — a change means the output contract
moved and must be reviewed deliberately), an ASCII preview golden fixing
that A x B primitives are visually countable on a slice, parameter
sensitivity, and seed independence (this generator uses no randomness).
"""

import dataclasses
import hashlib
from pathlib import Path

from vdbmat.io import read_material_label_manifest

from vdbmat_utils.cli.main import main
from vdbmat_utils.core import config_digest
from vdbmat_utils.preview import slice_ascii
from vdbmat_utils.primitives import PrimitiveArrayConfig

CUBE_CONFIG = PrimitiveArrayConfig(
    voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
    primitive="cube",
    counts_xyz=(3, 2, 1),
    primitive_size_m=4e-4,
    gap_m=2e-4,
    margin_m=1e-4,
)

SPHERE_CONFIG = PrimitiveArrayConfig(
    voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
    primitive="sphere",
    counts_xyz=(2, 2, 2),
    primitive_size_m=4e-4,
    gap_m=2e-4,
    margin_m=1e-4,
)

GOLDEN_CUBE_PAYLOAD_SHA256 = (
    "022650026264406aea72d365abbf9cb8a9de8e4df91dbb874e6e05b789bac5d3"
)
GOLDEN_CUBE_MANIFEST_SHA256 = (
    "726268168e99c32a156cf44f63c59f638f186f2e7acafd6867ab3e76255c86e3"
)
GOLDEN_SPHERE_PAYLOAD_SHA256 = (
    "2f39ec799586dc5ce55e0e5094513c31034e56a58865bdbd33a1fbced101a079"
)
GOLDEN_SPHERE_MANIFEST_SHA256 = (
    "d53be766aa195e6b6bf52798a4a024bcc3c8eeadb0cda8007f92d3afdb17fe49"
)


def _run_cli(tmp_path: Path, config: PrimitiveArrayConfig, out_name: str) -> Path:
    config_path = tmp_path / f"{out_name}.config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / out_name
    assert main(
        [
            "generate-primitive-array",
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "demo",
        ]
    ) == 0
    return out_dir


def _digests(out_dir: Path) -> tuple[str, str]:
    payload_sha = hashlib.sha256(
        (out_dir / "demo.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "demo.voxels.json").read_bytes()
    ).hexdigest()
    return payload_sha, manifest_sha


def test_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_cli(tmp_path, CUBE_CONFIG, "a")
    second = _run_cli(tmp_path, CUBE_CONFIG, "b")
    for filename in ("demo.voxels.json", "demo.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_golden_digests_cube(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, CUBE_CONFIG, "cube")
    payload_sha, manifest_sha = _digests(out_dir)
    assert payload_sha == GOLDEN_CUBE_PAYLOAD_SHA256
    assert manifest_sha == GOLDEN_CUBE_MANIFEST_SHA256


def test_golden_digests_sphere(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, SPHERE_CONFIG, "sphere")
    payload_sha, manifest_sha = _digests(out_dir)
    assert payload_sha == GOLDEN_SPHERE_PAYLOAD_SHA256
    assert manifest_sha == GOLDEN_SPHERE_MANIFEST_SHA256


def test_ascii_preview_shows_countable_primitives(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, CUBE_CONFIG, "preview")
    volume = read_material_label_manifest(out_dir / "demo.voxels.json")
    index = volume.geometry.shape_zyx[0] // 2
    preview = slice_ascii(volume, "z", index)
    assert preview == (
        "slice z=3  +x →  +y ↓\n"
        "111111111111111111\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "111111111111111111\n"
        "111111111111111111\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "133331133331133331\n"
        "111111111111111111"
    )


def test_sensitivity_counts_xyz(tmp_path: Path) -> None:
    changed = dataclasses.replace(CUBE_CONFIG, counts_xyz=(2, 2, 1))
    assert config_digest(CUBE_CONFIG) != config_digest(changed)
    out_a = _run_cli(tmp_path, CUBE_CONFIG, "a")
    out_b = _run_cli(tmp_path, changed, "b")
    assert _digests(out_a) != _digests(out_b)


def test_sensitivity_primitive(tmp_path: Path) -> None:
    changed = dataclasses.replace(CUBE_CONFIG, primitive="sphere")
    out_a = _run_cli(tmp_path, CUBE_CONFIG, "a")
    out_b = _run_cli(tmp_path, changed, "b")
    assert _digests(out_a) != _digests(out_b)


def test_sensitivity_primitive_size(tmp_path: Path) -> None:
    changed = dataclasses.replace(CUBE_CONFIG, primitive_size_m=3e-4)
    out_a = _run_cli(tmp_path, CUBE_CONFIG, "a")
    out_b = _run_cli(tmp_path, changed, "b")
    assert _digests(out_a) != _digests(out_b)


def test_seed_independence(tmp_path: Path) -> None:
    changed = dataclasses.replace(CUBE_CONFIG, seed=42)
    assert config_digest(CUBE_CONFIG) != config_digest(changed)
    out_a = _run_cli(tmp_path, CUBE_CONFIG, "a")
    out_b = _run_cli(tmp_path, changed, "b")
    payload_a = (out_a / "demo.material_id.npy").read_bytes()
    payload_b = (out_b / "demo.material_id.npy").read_bytes()
    assert payload_a == payload_b
