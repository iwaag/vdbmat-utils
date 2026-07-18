"""Unit tests for the image-stack workflow (PGM parsing, stacking, mapping)."""

import sys
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


def test_png_requires_image_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a base install: hide Pillow even if the extra is installed.
    monkeypatch.setitem(sys.modules, "PIL", None)
    directory = tmp_path / "stack"
    directory.mkdir()
    (directory / "slice_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ImageStackError, match="image' extra"):
        convert_image_stack(directory, _config(format="png"))


def test_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ImageStackError, match="unsupported format 'tiff'"):
        convert_image_stack(tmp_path, _config(format="tiff"))


def test_png_matches_pgm(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    pgm_volume = convert_image_stack(_write_stack(tmp_path / "pgm"), _config())
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    for z, pixels in enumerate((_SLICE_0, _SLICE_1)):
        Image.fromarray(pixels, mode="L").save(png_dir / f"slice_{z:04d}.png")
    png_volume = convert_image_stack(png_dir, _config(format="png"))
    np.testing.assert_array_equal(png_volume.material_id, pgm_volume.material_id)


def test_png_rejects_non_grayscale(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(png_dir / "slice_0000.png")
    with pytest.raises(ImageStackError, match="grayscale"):
        convert_image_stack(png_dir, _config(format="png"))


def test_config_digest_stable_and_seed_reserved(tmp_path: Path) -> None:
    a = convert_image_stack(_write_stack(tmp_path / "stack"), _config())
    b = convert_image_stack(tmp_path / "stack", _config())
    assert a.provenance.configuration_digest == b.provenance.configuration_digest
    assert a.provenance.configuration_digest is not None
    assert _config().seed == 0


# --- rgb levels (color-label stack input) --------------------------------

RGB_LEVELS: tuple[Mapping[str, object], ...] = (
    {"rgb": [0, 0, 0], "material_id": 0, "name": "air", "role": "background"},
    {"rgb": [255, 0, 0], "material_id": 1, "name": "resin_red", "role": "material"},
    {"rgb": [0, 255, 0], "material_id": 2, "name": "resin_green", "role": "material"},
)


def _rgb_config(**overrides: object) -> ImageStackConfig:
    fields: dict[str, object] = {
        "voxel_size_xyz_m": (0.001, 0.002, 0.003),
        "levels": RGB_LEVELS,
        "format": "png",
    }
    fields.update(overrides)
    return ImageStackConfig(**fields)  # type: ignore[arg-type]


_GRAY_TO_RGB = {0: (0, 0, 0), 128: (255, 0, 0), 255: (0, 255, 0)}


def _write_color_slice(path: Path, gray_pixels: npt.NDArray[np.uint8]) -> None:
    Image = pytest.importorskip("PIL.Image")
    rgb = np.zeros((*gray_pixels.shape, 3), dtype=np.uint8)
    for gray, color in _GRAY_TO_RGB.items():
        rgb[gray_pixels == gray] = color
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_color_stack(directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    _write_color_slice(directory / "slice_0000.png", _SLICE_0)
    _write_color_slice(directory / "slice_0001.png", _SLICE_1)
    return directory


def test_rgb_levels_match_equivalent_gray_stack(tmp_path: Path) -> None:
    gray_volume = convert_image_stack(_write_stack(tmp_path / "gray"), _config())
    rgb_volume = convert_image_stack(
        _write_color_stack(tmp_path / "rgb"), _rgb_config()
    )
    np.testing.assert_array_equal(rgb_volume.material_id, gray_volume.material_id)


def test_rgb_levels_reject_indexed_png_round_trip(tmp_path: Path) -> None:
    from vdbmat_utils.image.png import write_indexed_png

    directory = tmp_path / "indexed"
    directory.mkdir()
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    indices = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    write_indexed_png(directory / "slice_0000.png", indices, palette)
    volume = convert_image_stack(directory, _rgb_config())
    expected = np.array([[[0, 1], [2, 0]]], dtype=np.uint16)
    np.testing.assert_array_equal(volume.material_id, expected)


def test_rgb_and_gray_entries_cannot_mix(tmp_path: Path) -> None:
    levels = (*RGB_LEVELS[:2], LEVELS[2])
    directory = _write_color_stack(tmp_path / "rgb")
    with pytest.raises(ImageStackError, match="must not mix"):
        convert_image_stack(directory, _rgb_config(levels=levels))


def test_level_entry_cannot_have_both_gray_and_rgb(tmp_path: Path) -> None:
    levels = (
        {
            "gray": 0,
            "rgb": [0, 0, 0],
            "material_id": 0,
            "name": "air",
            "role": "background",
        },
    )
    directory = _write_color_stack(tmp_path / "rgb")
    with pytest.raises(ImageStackError, match="must not have both"):
        convert_image_stack(directory, _rgb_config(levels=levels))


def test_level_entry_requires_gray_or_rgb(tmp_path: Path) -> None:
    levels = ({"material_id": 0, "name": "air", "role": "background"},)
    directory = _write_color_stack(tmp_path / "rgb")
    with pytest.raises(ImageStackError, match="exactly one of 'gray' or 'rgb'"):
        convert_image_stack(directory, _rgb_config(levels=levels))


def test_duplicate_rgb_values(tmp_path: Path) -> None:
    levels = (
        *RGB_LEVELS,
        {"rgb": [0, 255, 0], "material_id": 3, "name": "extra", "role": "material"},
    )
    directory = _write_color_stack(tmp_path / "rgb")
    with pytest.raises(ImageStackError, match=r"duplicate rgb level \[0, 255, 0\]"):
        convert_image_stack(directory, _rgb_config(levels=levels))


def test_rgb_value_out_of_range(tmp_path: Path) -> None:
    levels = (
        {"rgb": [0, 0, 0], "material_id": 0, "name": "air", "role": "background"},
        {"rgb": [256, 0, 0], "material_id": 1, "name": "bad", "role": "material"},
    )
    directory = _write_color_stack(tmp_path / "rgb")
    with pytest.raises(ImageStackError, match=r"rgb\.r: must be an integer"):
        convert_image_stack(directory, _rgb_config(levels=levels))


def test_rgb_levels_reject_pgm_format(tmp_path: Path) -> None:
    directory = _write_stack(tmp_path / "stack")
    with pytest.raises(ImageStackError, match=r"require config\.format 'png'"):
        convert_image_stack(directory, _rgb_config(format="pgm"))


def test_undeclared_rgb_names_value_and_pixel(tmp_path: Path) -> None:
    directory = _write_color_stack(tmp_path / "rgb")
    Image = pytest.importorskip("PIL.Image")
    rgb = np.array(Image.open(directory / "slice_0001.png").convert("RGB"))
    rgb[2, 3] = (1, 2, 3)
    Image.fromarray(rgb, mode="RGB").save(directory / "slice_0001.png")
    with pytest.raises(
        ImageStackError,
        match=r"slice_0001\.png row 2, column 3 \(RGB \[1, 2, 3\]\)",
    ):
        convert_image_stack(directory, _rgb_config())
