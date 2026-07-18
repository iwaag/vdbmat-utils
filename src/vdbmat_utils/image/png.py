"""PNG slice reader and indexed-palette writer, behind the ``image`` extra
(Pillow).

Pillow is imported lazily so the base install stays dependency-free; the PGM
path never touches this module.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vdbmat_utils.image import ImageStackError

_PALETTE_ENTRIES = 256
_PNG_COMPRESS_LEVEL = 6


def read_png(path: Path) -> npt.NDArray[np.uint8]:
    """Read one 8-bit grayscale PNG slice."""
    try:
        from PIL import Image
    except ImportError as error:
        raise ImageStackError(
            "PNG input requires the 'image' extra: "
            "pip install 'vdbmat-utils[image]'"
        ) from error
    try:
        image: Any = Image.open(path)
        image.load()
    except OSError as error:
        raise ImageStackError(f"{path.name}: cannot read PNG: {error}") from error
    with image:
        if image.mode == "L":
            return np.asarray(image, dtype=np.uint8)
        if image.mode in ("I;16", "I", "F"):
            raise ImageStackError(
                f"{path.name}: only 8-bit grayscale PNG is supported, "
                f"got mode {image.mode!r}"
            )
        raise ImageStackError(
            f"{path.name}: only grayscale (mode 'L') PNG is supported, "
            f"got mode {image.mode!r}"
        )


def read_png_rgb(path: Path) -> npt.NDArray[np.uint8]:
    """Read a color-label PNG slice as an ``(H, W, 3)`` uint8 RGB array.

    Accepts mode "P" (indexed palette, expanded via the palette table
    directly — not Pillow's ``convert("RGB")``, whose transparency/palette
    handling has varied across versions) and mode "RGB". Other modes
    (grayscale, RGBA, 16-bit, etc.) are rejected explicitly rather than
    silently converted, since a dropped alpha channel or widened bit depth
    would silently change the declared color set.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise ImageStackError(
            "PNG input requires the 'image' extra: "
            "pip install 'vdbmat-utils[image]'"
        ) from error
    try:
        image: Any = Image.open(path)
        image.load()
    except OSError as error:
        raise ImageStackError(f"{path.name}: cannot read PNG: {error}") from error
    with image:
        if image.mode == "RGB":
            return np.asarray(image, dtype=np.uint8)
        if image.mode == "P":
            indices = np.asarray(image, dtype=np.uint8)
            raw_palette = list(image.getpalette() or [])
            raw_palette.extend([0] * (_PALETTE_ENTRIES * 3 - len(raw_palette)))
            palette_table = np.asarray(raw_palette, dtype=np.uint8).reshape(
                _PALETTE_ENTRIES, 3
            )
            return palette_table[indices]
        raise ImageStackError(
            f"{path.name}: only indexed-palette ('P') or 'RGB' PNG is "
            f"supported, got mode {image.mode!r}"
        )


def write_indexed_png(
    path: Path,
    indices: npt.NDArray[np.uint8],
    palette_rgb: Sequence[tuple[int, int, int]],
) -> None:
    """Write ``indices`` as an indexed-palette (mode "P") PNG.

    ``palette_rgb[i]`` is the colour for pixel value ``i``; entries beyond
    ``len(palette_rgb)`` are padded black. No antialiasing, resizing, or ICC
    handling is applied, so no intermediate colours can appear. Compression
    parameters are fixed for a deterministic double-run, but the encoded PNG
    bytes may still differ across Pillow versions/builds — see
    ``docs/print-slices.md``.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise ImageStackError(
            "PNG output requires the 'image' extra: "
            "pip install 'vdbmat-utils[image]'"
        ) from error
    if len(palette_rgb) > _PALETTE_ENTRIES:
        raise ImageStackError(
            f"palette has {len(palette_rgb)} entries, exceeds the PNG "
            f"indexed-palette limit of {_PALETTE_ENTRIES}"
        )
    flat_palette: list[int] = []
    for rgb in palette_rgb:
        flat_palette.extend(int(channel) for channel in rgb)
    flat_palette.extend([0] * (_PALETTE_ENTRIES - len(palette_rgb)) * 3)

    image = Image.fromarray(indices, mode="P")
    image.putpalette(flat_palette)
    image.save(
        path, format="PNG", optimize=False, compress_level=_PNG_COMPRESS_LEVEL
    )


def read_indexed_png(
    path: Path,
) -> tuple[npt.NDArray[np.uint8], list[tuple[int, int, int]]]:
    """Read an indexed-palette PNG back to its index array and palette.

    Test/decode-verification helper: the palette is always returned with
    exactly 256 entries (Pillow's in-memory representation), regardless of
    how many were meaningfully written.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise ImageStackError(
            "PNG input requires the 'image' extra: "
            "pip install 'vdbmat-utils[image]'"
        ) from error
    try:
        image: Any = Image.open(path)
        image.load()
    except OSError as error:
        raise ImageStackError(f"{path.name}: cannot read PNG: {error}") from error
    with image:
        if image.mode != "P":
            raise ImageStackError(
                f"{path.name}: expected indexed-palette PNG, got mode "
                f"{image.mode!r}"
            )
        indices = np.asarray(image, dtype=np.uint8)
        raw_palette = list(image.getpalette() or [])
        raw_palette.extend([0] * (_PALETTE_ENTRIES * 3 - len(raw_palette)))
        palette = [
            (raw_palette[i * 3], raw_palette[i * 3 + 1], raw_palette[i * 3 + 2])
            for i in range(_PALETTE_ENTRIES)
        ]
    return indices, palette
