"""Contract tests for the image-stack workflow.

Determinism (byte-equal double run through the CLI), checksum stability
(hardcoded payload/manifest digests — a change means the output contract
moved and must be reviewed deliberately), and material-count conservation
between the source slices and the produced volume.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import write_image_stack_fixture
from vdbmat_utils.image import ImageStackConfig, convert_image_stack
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


# --- rgb levels: purely-additive input path, same contract level as gray ---

_RGB_LEVELS = (
    {"rgb": [10, 10, 10], "material_id": 0, "name": "air", "role": "background"},
    {"rgb": [200, 0, 0], "material_id": 1, "name": "resin_red", "role": "material"},
    {"rgb": [0, 0, 200], "material_id": 2, "name": "resin_blue", "role": "material"},
)
_RGB_STACK_SHAPE_ZYX = (2, 3, 4)
_RGB_GOLDEN_PAYLOAD_SHA256 = (
    "15379abd5ce0766b151cea0052985860ac6e370363aa17ee2060249a0ec2176f"
)
_RGB_GOLDEN_MANIFEST_SHA256 = (
    "137e1e30305399126f79e690044a3896cd191578dc6e7a73d340730a9245a492"
)


def _write_rgb_stack_fixture(directory: Path) -> tuple[Path, ImageStackConfig]:
    Image = pytest.importorskip("PIL.Image")
    directory.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = _RGB_STACK_SHAPE_ZYX
    palette = [tuple(entry["rgb"]) for entry in _RGB_LEVELS]  # type: ignore[misc]
    labels = np.fromfunction(
        lambda z, y, x: (7 * z + 3 * y + x) % len(palette),
        (nz, ny, nx),
        dtype=np.int64,
    )
    for z in range(nz):
        rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
        for material_index, color in enumerate(palette):
            rgb[labels[z] == material_index] = color
        Image.fromarray(rgb, mode="RGB").save(directory / f"slice_{z:04d}.png")
    config = ImageStackConfig(
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0003),
        levels=_RGB_LEVELS,
        format="png",
    )
    return directory, config


def _run_rgb_cli(tmp_path: Path, out_name: str) -> Path:
    slices_dir, config = _write_rgb_stack_fixture(tmp_path / "rgb-slices")
    config_path = tmp_path / "rgb-config.json"
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
            "rgbstack",
        ]
    ) == 0
    return out_dir


def test_rgb_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_rgb_cli(tmp_path, "a")
    second = _run_rgb_cli(tmp_path, "b")
    for filename in ("rgbstack.voxels.json", "rgbstack.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_rgb_golden_digests(tmp_path: Path) -> None:
    out_dir = _run_rgb_cli(tmp_path, "out")
    payload_sha = hashlib.sha256(
        (out_dir / "rgbstack.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "rgbstack.voxels.json").read_bytes()
    ).hexdigest()
    assert payload_sha == _RGB_GOLDEN_PAYLOAD_SHA256
    assert manifest_sha == _RGB_GOLDEN_MANIFEST_SHA256


def test_rgb_config_digest_stable(tmp_path: Path) -> None:
    slices_dir, config = _write_rgb_stack_fixture(tmp_path / "rgb-slices")
    a = convert_image_stack(slices_dir, config)
    b = convert_image_stack(slices_dir, config)
    assert a.provenance.configuration_digest == b.provenance.configuration_digest
    assert a.provenance.configuration_digest is not None


def test_rgb_material_count_conservation(tmp_path: Path) -> None:
    slices_dir, config = _write_rgb_stack_fixture(tmp_path / "rgb-slices")
    volume = convert_image_stack(slices_dir, config)

    from vdbmat_utils.image.png import read_png_rgb

    rgb_to_id = {tuple(entry["rgb"]): entry["material_id"] for entry in config.levels}  # type: ignore[misc]
    slice_counts: dict[int, int] = dict.fromkeys(rgb_to_id.values(), 0)
    for path in sorted(slices_dir.glob("*.png")):
        pixels = read_png_rgb(path).reshape(-1, 3)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        for color, count in zip(colors.tolist(), counts.tolist(), strict=True):
            slice_counts[rgb_to_id[tuple(color)]] += count

    assert material_counts(volume) == slice_counts
