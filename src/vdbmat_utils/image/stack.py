"""Slice assembly, gray→material-level mapping, and stack validation."""

import dataclasses
import hashlib
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import (
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialPalette,
    MaterialRole,
)

from vdbmat_utils.core import (
    ConfigError,
    GeneratorConfig,
    build_material_label_volume,
    build_provenance,
    provenance_identity,
)
from vdbmat_utils.image import ImageStackError
from vdbmat_utils.image.pgm import read_pgm

GENERATOR = "vdbmat-utils.image.stack"
GENERATOR_VERSION = "0.1.0"

_FORMATS = ("pgm", "png")
_LEVEL_FIELDS = {"gray", "rgb", "material_id", "name", "role"}
_TRAILING_INT = re.compile(r"(\d+)\D*$")


def _pack_rgb(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (r << 16) | (g << 8) | b


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ImageStackConfig(GeneratorConfig):
    """Configuration for the image-stack workflow.

    ``levels`` declares every gray value that may appear in the slices, each
    as a mapping with keys ``gray`` (0..255), ``material_id``, ``name``, and
    ``role``. ``seed`` is inherited, unused in Phase 1, and reserved.
    """

    voxel_size_xyz_m: tuple[float, float, float]
    levels: tuple[Mapping[str, object], ...]
    local_to_world: tuple[tuple[float, ...], ...] | None = None
    format: str = "pgm"


def _parse_rgb_value(field: str, value: object) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ImageStackError(f"{field}: must be an array of 3 integers")
    channels = []
    for name, channel in zip(("r", "g", "b"), value, strict=True):
        valid = (
            isinstance(channel, int)
            and not isinstance(channel, bool)
            and 0 <= channel <= 255
        )
        if not valid:
            raise ImageStackError(f"{field}.{name}: must be an integer in [0, 255]")
        channels.append(channel)
    return (channels[0], channels[1], channels[2])


def _parse_levels(
    levels: tuple[Mapping[str, object], ...],
) -> tuple[str, dict[int, int], MaterialPalette]:
    """Validate ``levels`` into a mode, a key→material-id map, and a palette.

    ``mode`` is ``"gray"`` or ``"rgb"``; the key map is keyed by the gray
    value (0..255) or the packed RGB value (``r<<16 | g<<8 | b``)
    respectively. A config's entries must be all-gray or all-rgb — a stack
    is either grayscale or color, and mixing would make the reader's mode
    selection pixel-dependent.
    """
    if not levels:
        raise ImageStackError("config.levels: must be a non-empty array")

    modes = {
        "gray" if "gray" in entry else "rgb" if "rgb" in entry else None
        for entry in levels
    }
    if None in modes:
        raise ImageStackError(
            "config.levels: each entry must have exactly one of 'gray' or 'rgb'"
        )
    if len(modes) > 1:
        raise ImageStackError(
            "config.levels: entries must not mix 'gray' and 'rgb' within one config"
        )
    mode = modes.pop()

    key_to_id: dict[int, int] = {}
    definitions: list[MaterialDefinition] = []
    for index, entry in enumerate(levels):
        field = f"config.levels[{index}]"
        if not isinstance(entry, Mapping):
            raise ImageStackError(f"{field}: must be an object")
        unknown = sorted(set(entry) - _LEVEL_FIELDS)
        if unknown:
            raise ImageStackError(f"{field}: unknown fields: {unknown}")
        if mode == "gray":
            if "rgb" in entry:
                raise ImageStackError(f"{field}: must not have both 'gray' and 'rgb'")
            gray = entry.get("gray")
            if (
                not isinstance(gray, int)
                or isinstance(gray, bool)
                or not 0 <= gray <= 255
            ):
                raise ImageStackError(f"{field}.gray: must be an integer in [0, 255]")
            if gray in key_to_id:
                raise ImageStackError(f"{field}.gray: duplicate gray level {gray}")
            key = gray
        else:
            if "gray" in entry:
                raise ImageStackError(f"{field}: must not have both 'gray' and 'rgb'")
            rgb = _parse_rgb_value(f"{field}.rgb", entry.get("rgb"))
            key = _pack_rgb(rgb)
            if key in key_to_id:
                raise ImageStackError(f"{field}.rgb: duplicate rgb level {list(rgb)}")
        material_id = entry.get("material_id")
        name = entry.get("name")
        role = entry.get("role")
        if not isinstance(material_id, int) or isinstance(material_id, bool):
            raise ImageStackError(f"{field}.material_id: must be an integer")
        if not isinstance(name, str):
            raise ImageStackError(f"{field}.name: must be a string")
        if not isinstance(role, str):
            raise ImageStackError(f"{field}.role: must be a string")
        try:
            definition = MaterialDefinition(
                material_id=material_id,
                name=name,
                role=MaterialRole(role),
            )
        except (TypeError, ValueError) as error:
            raise ImageStackError(f"{field}: {error}") from error
        key_to_id[key] = definition.material_id
        definitions.append(definition)
    try:
        palette = MaterialPalette.from_sequence(definitions)
    except (TypeError, ValueError) as error:
        raise ImageStackError(f"config.levels: {error}") from error
    return mode, key_to_id, palette


def _check_sequence_gaps(paths: list[Path]) -> None:
    """Reject numerically-named sequences with missing indices.

    Only applies when every filename ends in a number (before the suffix);
    interpolation over gaps is Phase 2, so a gap is an error, not a request.
    """
    matches = [_TRAILING_INT.search(path.stem) for path in paths]
    if any(match is None for match in matches):
        return
    numbers = sorted(int(match.group(1)) for match in matches if match is not None)
    expected = range(numbers[0], numbers[0] + len(numbers))
    missing = sorted(set(expected) - set(numbers))
    if missing:
        raise ImageStackError(
            f"slices: numeric sequence has missing index(es) {missing} "
            f"between {numbers[0]} and {numbers[-1]}"
        )


def _first_undeclared_pixel(
    layers: list[npt.NDArray[np.int64]],
    paths: list[Path],
    declared: npt.NDArray[np.int64],
    describe_value: Callable[[int], str],
) -> str:
    for layer, path in zip(layers, paths, strict=True):
        undeclared_mask = ~np.isin(layer, declared)
        if undeclared_mask.any():
            row, col = (int(v) for v in np.argwhere(undeclared_mask)[0])
            value = int(layer[row, col])
            return (
                f"first at {path.name} row {row}, column {col} "
                f"({describe_value(value)})"
            )
    raise AssertionError("no undeclared pixel found")  # pragma: no cover


def _describe_gray(value: int) -> str:
    return f"gray {value}"


def _describe_rgb(value: int) -> str:
    return f"RGB {[(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]}"


def _read_slice(path: Path, image_format: str) -> npt.NDArray[np.uint8]:
    if image_format == "pgm":
        return read_pgm(path)
    if image_format == "png":
        try:
            from vdbmat_utils.image.png import read_png
        except ImportError as error:
            raise ImageStackError(
                "PNG input requires the 'image' extra: "
                "pip install 'vdbmat-utils[image]'"
            ) from error
        return read_png(path)
    raise ImageStackError(
        f"config.format: unsupported format {image_format!r}; "
        f"expected one of {', '.join(_FORMATS)}"
    )


def _read_color_slice(path: Path) -> npt.NDArray[np.int64]:
    try:
        from vdbmat_utils.image.png import read_png_rgb
    except ImportError as error:
        raise ImageStackError(
            "PNG input requires the 'image' extra: pip install 'vdbmat-utils[image]'"
        ) from error
    rgb = read_png_rgb(path).astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _apply_lookup(
    stack: npt.NDArray[np.int64], key_to_id: dict[int, int]
) -> npt.NDArray[np.uint16]:
    sorted_keys = np.asarray(sorted(key_to_id), dtype=np.int64)
    sorted_values = np.asarray(
        [key_to_id[int(key)] for key in sorted_keys], dtype=np.uint16
    )
    positions = np.searchsorted(sorted_keys, stack)
    return sorted_values[positions]


def stack_identity(volume: MaterialLabelVolume) -> str:
    """Asset identity per D6: SHA-256 over the concatenated per-slice digests
    (provenance ``sources``, in stack order) plus the configuration digest."""
    try:
        return provenance_identity(volume.provenance)
    except ConfigError as error:  # pragma: no cover - convert always sets it
        raise ImageStackError(f"volume provenance: {error}") from error


def convert_image_stack(
    slices_dir: Path, config: ImageStackConfig
) -> MaterialLabelVolume:
    """Build a canonical label volume from a directory of labeled 2D slices."""
    if config.format not in _FORMATS:
        raise ImageStackError(
            f"config.format: unsupported format {config.format!r}; "
            f"expected one of {', '.join(_FORMATS)}"
        )
    mode, key_to_id, palette = _parse_levels(tuple(config.levels))
    if mode == "rgb" and config.format != "png":
        raise ImageStackError(
            "config.levels: 'rgb' entries require config.format 'png' "
            f"(PGM is a grayscale-only format), got {config.format!r}"
        )

    slice_paths = sorted(slices_dir.glob(f"*.{config.format}"))
    if not slice_paths:
        raise ImageStackError(
            f"slices: no .{config.format} files under {slices_dir}"
        )
    _check_sequence_gaps(slice_paths)

    digests: list[str] = []
    layers: list[npt.NDArray[np.int64]] = []
    for path in slice_paths:
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        layer = (
            _read_color_slice(path)
            if mode == "rgb"
            else _read_slice(path, config.format)
        )
        if layers and layer.shape != layers[0].shape:
            raise ImageStackError(
                f"{path.name}: slice shape {layer.shape} differs from "
                f"{slice_paths[0].name} {layers[0].shape}"
            )
        layers.append(np.asarray(layer, dtype=np.int64))
    stack = np.stack(layers, axis=0)  # [z, y, x]

    declared = np.asarray(sorted(key_to_id), dtype=np.int64)
    present = np.unique(stack)
    undeclared = sorted(int(v) for v in present if int(v) not in key_to_id)
    if undeclared:
        describe = _describe_rgb if mode == "rgb" else _describe_gray
        location = _first_undeclared_pixel(layers, slice_paths, declared, describe)
        value_kind = "RGB values" if mode == "rgb" else "gray values"
        described = (
            [describe(v) for v in undeclared] if mode == "rgb" else undeclared
        )
        raise ImageStackError(
            f"slices: {value_kind} {described} are not declared in "
            f"config.levels; {location}"
        )

    label = _apply_lookup(stack, key_to_id)

    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=tuple(f"sha256:{digest}" for digest in digests),
        # No directory name here: the manifest digest must depend only on the
        # slice bytes and the configuration (checksum-stability contract).
        notes=(
            "layered color-label image stack; rows=+Y, columns=+X"
            if mode == "rgb"
            else "layered image stack; rows=+Y, columns=+X"
        ),
    )
    return build_material_label_volume(
        material_id=label,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        palette=palette,
        provenance=provenance,
        local_to_world=config.local_to_world,
    )
