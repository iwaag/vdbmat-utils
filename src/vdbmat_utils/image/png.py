"""PNG slice reader, behind the ``image`` extra (Pillow).

Pillow is imported lazily so the base install stays dependency-free; the PGM
path never touches this module.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vdbmat_utils.image import ImageStackError


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
