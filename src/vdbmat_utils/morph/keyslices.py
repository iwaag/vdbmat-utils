"""Key-slice discovery: filename z-index parsing, reading, level mapping.

Reuses the Phase 1 slice readers and ``levels`` validation from
``vdbmat_utils.image`` — the morph contract for gray→material mapping is
exactly the image-stack contract; only the meaning of the filename number
changes (it *is* the output z index, and gaps are the point).
"""

import dataclasses
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialPalette

from vdbmat_utils.image import ImageStackError
from vdbmat_utils.image.stack import _parse_levels, _read_slice
from vdbmat_utils.morph import MorphError

_NUMERIC_GROUPS = re.compile(r"\d+")


@dataclasses.dataclass(frozen=True, slots=True)
class KeySlices:
    """Validated key slices: strictly increasing z indices, uniform shape."""

    z_indices: tuple[int, ...]
    labels: tuple[npt.NDArray[np.uint16], ...]  # one 2-D (y, x) layer per key
    digests: tuple[str, ...]  # sha256:<hex> per slice file, in z order
    palette: MaterialPalette


def _parse_z_index(path: Path) -> int:
    groups = _NUMERIC_GROUPS.findall(path.stem)
    if len(groups) != 1:
        raise MorphError(
            f"{path.name}: key-slice filenames must contain exactly one "
            f"numeric group (the z index), found {len(groups)}"
        )
    return int(groups[0])


def load_key_slices(
    slices_dir: Path,
    *,
    levels: tuple[Mapping[str, object], ...],
    image_format: str,
) -> KeySlices:
    """Read and validate every key slice under ``slices_dir``.

    Filenames declare output z indices; duplicates and lexicographic order
    disagreeing with numeric order are errors. Gray values are mapped through
    the shared ``levels`` table; an undeclared value is an error naming the
    file and value (same behavior as ``convert-image-stack``).
    """
    try:
        mode, gray_to_id, palette = _parse_levels(levels)
    except ImageStackError as error:
        raise MorphError(str(error)) from error
    if mode == "rgb":
        raise MorphError(
            "config.levels: 'rgb' entries are not supported by morph-stack; "
            "use 'gray' levels"
        )

    paths = sorted(slices_dir.glob(f"*.{image_format}"))
    if not paths:
        raise MorphError(f"slices: no .{image_format} files under {slices_dir}")

    z_indices = [_parse_z_index(path) for path in paths]
    for position in range(1, len(paths)):
        z, next_z = z_indices[position - 1], z_indices[position]
        if next_z == z:
            raise MorphError(
                f"slices: {paths[position - 1].name} and "
                f"{paths[position].name} declare the same z index {z}"
            )
        if next_z < z:
            raise MorphError(
                f"slices: filename order is not monotonic in z "
                f"({paths[position - 1].name} → {z}, "
                f"{paths[position].name} → {next_z})"
            )

    lookup = np.zeros(256, dtype=np.uint16)
    declared = np.zeros(256, dtype=np.bool_)
    for gray, material_id in gray_to_id.items():
        lookup[gray] = material_id
        declared[gray] = True

    labels: list[npt.NDArray[np.uint16]] = []
    digests: list[str] = []
    for path in paths:
        digests.append(f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}")
        try:
            pixels = _read_slice(path, image_format)
        except ImageStackError as error:
            raise MorphError(str(error)) from error
        if labels and pixels.shape != labels[0].shape:
            raise MorphError(
                f"{path.name}: slice shape {pixels.shape} differs from "
                f"{paths[0].name} {labels[0].shape}"
            )
        undeclared_mask = ~declared[pixels]
        if undeclared_mask.any():
            row, col = (int(v) for v in np.argwhere(undeclared_mask)[0])
            raise MorphError(
                f"{path.name}: gray value {int(pixels[row, col])} at row {row}, "
                f"column {col} is not declared in config.levels"
            )
        labels.append(lookup[pixels])

    return KeySlices(
        z_indices=tuple(z_indices),
        labels=tuple(labels),
        digests=tuple(digests),
        palette=palette,
    )
