"""Unit tests for the image-stack workflow (PGM parsing, stacking, mapping)."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from vdbmat_utils.image import (
    ImageStackConfig,
    ImageStackError,
    convert_image_stack,
)
from vdbmat_utils.image.pgm import read_pgm

LEVELS: tuple[Mapping[str, object], ...] = (
    {"gray": 0, "material_id": 0, "name": "air", "role": "background"},
    {"gray": 128, "material_id": 1, "name": "resin_clear", "role": "material"},
    {"gray": 255, "material_id": 2, "name": "resin_white", "role": "material"},
)


def _config(**overrides: object) -> ImageStackConfig:
    fields: dict[str, object] = {
        "voxel_size_xyz_m": (0.001, 0.002, 0.003),
        "levels": LEVELS,
    }
    fields.update(overrides)
    return ImageStackConfig(**fields)  # type: ignore[arg-type]


def _write_p5(path: Path, pixels: npt.NDArray[np.uint8]) -> Path:
    rows, cols = pixels.shape
    path.write_bytes(f"P5\n{cols} {rows}\n255\n".encode() + pixels.tobytes())
    return path


def _write_p2(path: Path, pixels: npt.NDArray[np.uint8]) -> Path:
    rows, cols = pixels.shape
    body = "\n".join(" ".join(str(v) for v in row) for row in pixels.tolist())
    path.write_text(f"P2\n# comment\n{cols} {rows}\n255\n{body}\n")
    return path


# Asymmetric 2 (z) x 3 (y) x 4 (x) stack: no transpose maps it onto itself.
_SLICE_0 = np.array(
    [
        [0, 128, 255, 0],
        [128, 128, 0, 0],
        [0, 0, 0, 255],
    ],
    dtype=np.uint8,
)
_SLICE_1 = np.array(
    [
        [255, 255, 255, 128],
        [0, 0, 128, 0],
        [128, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def _write_stack(directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    _write_p5(directory / "slice_0000.pgm", _SLICE_0)
    _write_p5(directory / "slice_0001.pgm", _SLICE_1)
    return directory


def test_read_pgm_p5_and_p2_agree(tmp_path: Path) -> None:
    p5 = read_pgm(_write_p5(tmp_path / "a.pgm", _SLICE_0))
    p2 = read_pgm(_write_p2(tmp_path / "b.pgm", _SLICE_0))
    np.testing.assert_array_equal(p5, p2)
    assert p5.dtype == np.uint8


def test_read_pgm_rejects_non_8bit(tmp_path: Path) -> None:
    path = tmp_path / "deep.pgm"
    path.write_bytes(b"P5\n2 2\n65535\n" + bytes(8))
    with pytest.raises(ImageStackError, match="maxval 255"):
        read_pgm(path)


def test_read_pgm_rejects_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "short.pgm"
    path.write_bytes(b"P5\n2")
    with pytest.raises(ImageStackError, match="truncated"):
        read_pgm(path)


def test_zyx_placement_literal(tmp_path: Path) -> None:
    volume = convert_image_stack(_write_stack(tmp_path / "stack"), _config())
    expected = np.array(
        [
            [[0, 1, 2, 0], [1, 1, 0, 0], [0, 0, 0, 2]],
            [[2, 2, 2, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(volume.material_id, expected)
    assert volume.geometry.shape_zyx == (2, 3, 4)
    assert volume.geometry.voxel_size_xyz_m == (0.001, 0.002, 0.003)


def test_provenance_records_config_and_slice_digests(tmp_path: Path) -> None:
    config = _config()
    volume = convert_image_stack(_write_stack(tmp_path / "stack"), config)
    assert volume.provenance.generator == "vdbmat-utils.image.stack"
    assert len(volume.provenance.sources) == 2
    assert all(s.startswith("sha256:") for s in volume.provenance.sources)
    assert volume.provenance.sources[0] != volume.provenance.sources[1]


def test_local_to_world_passthrough(tmp_path: Path) -> None:
    transform = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    volume = convert_image_stack(
        _write_stack(tmp_path / "stack"), _config(local_to_world=transform)
    )
    assert volume.geometry.local_to_world == transform


def test_undeclared_gray_names_value_and_pixel(tmp_path: Path) -> None:
    directory = _write_stack(tmp_path / "stack")
    bad = _SLICE_1.copy()
    bad[2, 3] = 7
    _write_p5(directory / "slice_0001.pgm", bad)
    with pytest.raises(
        ImageStackError,
        match=r"\[7\].*slice_0001\.pgm row 2, column 3 \(gray 7\)",
    ):
        convert_image_stack(directory, _config())


def test_shape_mismatch(tmp_path: Path) -> None:
    directory = _write_stack(tmp_path / "stack")
    _write_p5(directory / "slice_0001.pgm", _SLICE_0[:, :3].copy())
    with pytest.raises(ImageStackError, match="differs from"):
        convert_image_stack(directory, _config())


def test_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ImageStackError, match=r"no \.pgm files"):
        convert_image_stack(empty, _config())


def test_missing_sequence_index(tmp_path: Path) -> None:
    directory = tmp_path / "stack"
    directory.mkdir()
    _write_p5(directory / "slice_0003.pgm", _SLICE_0)
    _write_p5(directory / "slice_0005.pgm", _SLICE_1)
    with pytest.raises(ImageStackError, match=r"missing index\(es\) \[4\]"):
        convert_image_stack(directory, _config())


def test_non_numeric_names_do_not_trigger_gap_check(tmp_path: Path) -> None:
    directory = tmp_path / "stack"
    directory.mkdir()
    _write_p5(directory / "bottom.pgm", _SLICE_0)
    _write_p5(directory / "top.pgm", _SLICE_1)
    volume = convert_image_stack(directory, _config())
    assert volume.geometry.shape_zyx == (2, 3, 4)


def test_duplicate_material_id_across_levels(tmp_path: Path) -> None:
    levels = (
        *LEVELS[:2],
        {"gray": 255, "material_id": 1, "name": "dup", "role": "material"},
    )
    with pytest.raises(ImageStackError, match="duplicate material_id"):
        convert_image_stack(_write_stack(tmp_path / "stack"), _config(levels=levels))


def test_duplicate_gray_values(tmp_path: Path) -> None:
    levels = (
        *LEVELS,
        {"gray": 255, "material_id": 3, "name": "extra", "role": "material"},
    )
    with pytest.raises(ImageStackError, match="duplicate gray level 255"):
        convert_image_stack(_write_stack(tmp_path / "stack"), _config(levels=levels))


def test_level_field_validation(tmp_path: Path) -> None:
    stack = _write_stack(tmp_path / "stack")
    with pytest.raises(ImageStackError, match="non-empty"):
        convert_image_stack(stack, _config(levels=()))
    with pytest.raises(ImageStackError, match=r"gray: must be an integer"):
        convert_image_stack(
            stack,
            _config(levels=({"gray": "0", "material_id": 0,
                             "name": "air", "role": "background"},)),
        )
    with pytest.raises(ImageStackError, match="unknown fields"):
        convert_image_stack(
            stack,
            _config(levels=({"gray": 0, "material_id": 0, "name": "air",
                             "role": "background", "opacity": 1},)),
        )


def test_png_requires_image_extra(tmp_path: Path) -> None:
    directory = tmp_path / "stack"
    directory.mkdir()
    (directory / "slice_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ImageStackError, match="image' extra"):
        convert_image_stack(directory, _config(format="png"))


def test_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ImageStackError, match="unsupported format 'tiff'"):
        convert_image_stack(tmp_path, _config(format="tiff"))


def test_config_digest_stable_and_seed_reserved(tmp_path: Path) -> None:
    a = convert_image_stack(_write_stack(tmp_path / "stack"), _config())
    b = convert_image_stack(tmp_path / "stack", _config())
    assert a.provenance.configuration_digest == b.provenance.configuration_digest
    assert a.provenance.configuration_digest is not None
    assert _config().seed == 0
