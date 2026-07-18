"""Unit tests for the indexed-palette PNG writer/reader helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

from vdbmat_utils.image import ImageStackError
from vdbmat_utils.image.png import read_indexed_png, write_indexed_png


def test_write_then_read_round_trips_indices(tmp_path: Path) -> None:
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    path = tmp_path / "slice_0000.png"
    write_indexed_png(path, indices, palette)

    decoded_indices, decoded_palette = read_indexed_png(path)
    np.testing.assert_array_equal(decoded_indices, indices)
    assert decoded_palette[:3] == palette


def test_write_pads_unused_palette_entries_black(tmp_path: Path) -> None:
    indices = np.zeros((2, 2), dtype=np.uint8)
    palette = [(0, 0, 0), (10, 20, 30)]
    path = tmp_path / "slice_0000.png"
    write_indexed_png(path, indices, palette)
    _, decoded_palette = read_indexed_png(path)
    assert decoded_palette[2] == (0, 0, 0)
    assert decoded_palette[255] == (0, 0, 0)


def test_write_rejects_palette_over_256(tmp_path: Path) -> None:
    indices = np.zeros((1, 1), dtype=np.uint8)
    palette = [(0, 0, 0)] * 257
    with pytest.raises(ImageStackError, match="256"):
        write_indexed_png(tmp_path / "slice_0000.png", indices, palette)


def test_double_write_is_byte_equal(tmp_path: Path) -> None:
    indices = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    palette = [(0, 0, 0), (255, 255, 255)]
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    write_indexed_png(first, indices, palette)
    write_indexed_png(second, indices, palette)
    assert first.read_bytes() == second.read_bytes()


def test_write_requires_image_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    indices = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ImageStackError, match="image' extra"):
        write_indexed_png(tmp_path / "slice_0000.png", indices, [(0, 0, 0)])


def test_read_rejects_non_indexed_png(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "gray.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(path)
    with pytest.raises(ImageStackError, match="indexed-palette"):
        read_indexed_png(path)
