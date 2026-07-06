"""Zero-dependency reader for 8-bit grayscale PGM slices (P5 binary, P2 ASCII).

Kept dependency-free so the base image workflow works without the ``image``
extra; PNG input lives in ``image/png.py`` behind that extra.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from vdbmat_utils.image import ImageStackError


def read_pgm(path: Path) -> npt.NDArray[np.uint8]:
    """Read one 8-bit grayscale PGM (P5 binary or P2 ASCII) slice."""
    data = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    # PGM header tokens may be separated by whitespace and '#' comments.
    while len(tokens) < 4 and index < len(data):
        if data[index : index + 1].isspace():
            index += 1
            continue
        if data[index : index + 1] == b"#":
            end = data.find(b"\n", index)
            index = len(data) if end == -1 else end + 1
            continue
        start = index
        while index < len(data) and not data[index : index + 1].isspace():
            index += 1
        tokens.append(data[start:index])
    if len(tokens) < 4:
        raise ImageStackError(f"{path.name}: truncated PGM header")
    magic, width_token, height_token, maxval_token = tokens
    try:
        width, height, maxval = (
            int(width_token),
            int(height_token),
            int(maxval_token),
        )
    except ValueError as error:
        raise ImageStackError(f"{path.name}: non-numeric PGM header") from error
    if width <= 0 or height <= 0:
        raise ImageStackError(f"{path.name}: image dimensions must be positive")
    if maxval != 255:
        raise ImageStackError(
            f"{path.name}: only 8-bit PGM (maxval 255) is supported"
        )

    if magic == b"P5":
        pixels = data[index + 1 :]
        expected = width * height
        if len(pixels) != expected:
            raise ImageStackError(
                f"{path.name}: expected {expected} pixel bytes, got {len(pixels)}"
            )
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
    if magic == b"P2":
        values = data[index:].split()
        if len(values) != width * height:
            raise ImageStackError(
                f"{path.name}: expected {width * height} pixels, got {len(values)}"
            )
        try:
            flat = np.asarray([int(item) for item in values], dtype=np.int64)
        except ValueError as error:
            raise ImageStackError(f"{path.name}: non-numeric P2 pixel") from error
        if flat.min() < 0 or flat.max() > 255:
            raise ImageStackError(f"{path.name}: P2 pixels must lie in [0, 255]")
        return flat.astype(np.uint8).reshape(height, width)
    raise ImageStackError(f"{path.name}: unsupported PGM magic {magic!r}")
