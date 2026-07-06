"""Contract tests for the image-stack workflow.

Determinism (byte-equal double run through the CLI), checksum stability
(hardcoded payload/manifest digests — a change means the output contract
moved and must be reviewed deliberately), and material-count conservation
between the source slices and the produced volume.
"""

import hashlib
from pathlib import Path

import numpy as np

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import write_image_stack_fixture
from vdbmat_utils.image import convert_image_stack
from vdbmat_utils.image.pgm import read_pgm
from vdbmat_utils.preview import material_counts

GOLDEN_PAYLOAD_SHA256 = (
    "7ae9ce17a655aaf3f5758eaf95822a290943ef920d640864b7a6d91c33220b5c"
)
GOLDEN_MANIFEST_SHA256 = (
    "0200f3a0001dd1d4451ac39ce98f504d7959e1851c0e04fbd5bdd701c43116a5"
)


def _run_cli(tmp_path: Path, out_name: str) -> Path:
    slices_dir, config = write_image_stack_fixture(tmp_path / "slices")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / out_name
    assert main(
        [
            "convert-image-stack",
            str(slices_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "stack",
        ]
    ) == 0
    return out_dir


def test_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_cli(tmp_path, "a")
    second = _run_cli(tmp_path, "b")
    for filename in ("stack.voxels.json", "stack.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_golden_digests(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, "out")
    payload_sha = hashlib.sha256(
        (out_dir / "stack.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "stack.voxels.json").read_bytes()
    ).hexdigest()
    assert payload_sha == GOLDEN_PAYLOAD_SHA256
    assert manifest_sha == GOLDEN_MANIFEST_SHA256


def test_material_count_conservation(tmp_path: Path) -> None:
    slices_dir, config = write_image_stack_fixture(tmp_path / "slices")
    volume = convert_image_stack(slices_dir, config)

    gray_to_id: dict[int, int] = {}
    for entry in config.levels:
        gray, material_id = entry["gray"], entry["material_id"]
        assert isinstance(gray, int) and isinstance(material_id, int)
        gray_to_id[gray] = material_id
    slice_counts: dict[int, int] = {mid: 0 for mid in gray_to_id.values()}
    for path in sorted(slices_dir.glob("*.pgm")):
        grays, counts = np.unique(read_pgm(path), return_counts=True)
        for gray, count in zip(grays.tolist(), counts.tolist(), strict=True):
            slice_counts[gray_to_id[gray]] += count

    assert material_counts(volume) == slice_counts


def test_validate_and_identity(tmp_path: Path) -> None:
    out_dir = _run_cli(tmp_path, "out")
    manifest = out_dir / "stack.voxels.json"
    assert main(["validate", str(manifest)]) == 0
    assert "sha256:" in manifest.read_text(encoding="utf-8")
