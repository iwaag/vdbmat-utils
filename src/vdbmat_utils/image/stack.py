"""Slice assembly, gray→material-level mapping, and stack validation."""

import dataclasses
import hashlib
import re
from collections.abc import Mapping
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
    GeneratorConfig,
    build_material_label_volume,
    build_provenance,
)
from vdbmat_utils.image import ImageStackError
from vdbmat_utils.image.pgm import read_pgm

GENERATOR = "vdbmat-utils.image.stack"
GENERATOR_VERSION = "0.1.0"

_FORMATS = ("pgm", "png")
_LEVEL_FIELDS = {"gray", "material_id", "name", "role"}
_TRAILING_INT = re.compile(r"(\d+)\D*$")


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


def _parse_levels(
    levels: tuple[Mapping[str, object], ...],
) -> tuple[dict[int, int], MaterialPalette]:
    """Validate ``levels`` into a gray→material-id map and a palette."""
    if not levels:
        raise ImageStackError("config.levels: must be a non-empty array")
    gray_to_id: dict[int, int] = {}
    definitions: list[MaterialDefinition] = []
    for index, entry in enumerate(levels):
        field = f"config.levels[{index}]"
        if not isinstance(entry, Mapping):
            raise ImageStackError(f"{field}: must be an object")
        unknown = sorted(set(entry) - _LEVEL_FIELDS)
        if unknown:
            raise ImageStackError(f"{field}: unknown fields: {unknown}")
        gray = entry.get("gray")
        if not isinstance(gray, int) or isinstance(gray, bool) or not 0 <= gray <= 255:
            raise ImageStackError(f"{field}.gray: must be an integer in [0, 255]")
        if gray in gray_to_id:
            raise ImageStackError(f"{field}.gray: duplicate gray level {gray}")
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
        gray_to_id[gray] = definition.material_id
        definitions.append(definition)
    try:
        palette = MaterialPalette.from_sequence(definitions)
    except (TypeError, ValueError) as error:
        raise ImageStackError(f"config.levels: {error}") from error
    return gray_to_id, palette


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
    layers: list[npt.NDArray[np.uint8]],
    paths: list[Path],
    declared: npt.NDArray[np.int64],
) -> str:
    for layer, path in zip(layers, paths, strict=True):
        undeclared_mask = ~np.isin(layer, declared)
        if undeclared_mask.any():
            row, col = (int(v) for v in np.argwhere(undeclared_mask)[0])
            value = int(layer[row, col])
            return f"first at {path.name} row {row}, column {col} (gray {value})"
    raise AssertionError("no undeclared pixel found")  # pragma: no cover


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


def stack_identity(volume: MaterialLabelVolume) -> str:
    """Asset identity per D6: SHA-256 over the concatenated per-slice digests
    (provenance ``sources``, in stack order) plus the configuration digest."""
    combined = hashlib.sha256()
    for source in volume.provenance.sources:
        combined.update(source.encode("utf-8"))
    configuration_digest = volume.provenance.configuration_digest
    if configuration_digest is None:  # pragma: no cover - convert always sets it
        raise ImageStackError("volume provenance has no configuration digest")
    combined.update(configuration_digest.encode("utf-8"))
    return f"sha256:{combined.hexdigest()}"


def convert_image_stack(
    slices_dir: Path, config: ImageStackConfig
) -> MaterialLabelVolume:
    """Build a canonical label volume from a directory of labeled 2D slices."""
    if config.format not in _FORMATS:
        raise ImageStackError(
            f"config.format: unsupported format {config.format!r}; "
            f"expected one of {', '.join(_FORMATS)}"
        )
    gray_to_id, palette = _parse_levels(tuple(config.levels))

    slice_paths = sorted(slices_dir.glob(f"*.{config.format}"))
    if not slice_paths:
        raise ImageStackError(
            f"slices: no .{config.format} files under {slices_dir}"
        )
    _check_sequence_gaps(slice_paths)

    digests: list[str] = []
    layers: list[npt.NDArray[np.uint8]] = []
    for path in slice_paths:
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        layer = _read_slice(path, config.format)
        if layers and layer.shape != layers[0].shape:
            raise ImageStackError(
                f"{path.name}: slice shape {layer.shape} differs from "
                f"{slice_paths[0].name} {layers[0].shape}"
            )
        layers.append(layer)
    stack = np.stack(layers, axis=0)  # [z, y, x] grayscale

    declared = np.asarray(sorted(gray_to_id), dtype=np.int64)
    present = np.unique(stack)
    undeclared = sorted(int(v) for v in present if int(v) not in gray_to_id)
    if undeclared:
        location = _first_undeclared_pixel(layers, slice_paths, declared)
        raise ImageStackError(
            f"slices: gray values {undeclared} are not declared in "
            f"config.levels; {location}"
        )

    lookup = np.zeros(256, dtype=np.uint16)
    for gray, material_id in gray_to_id.items():
        lookup[gray] = material_id
    label = lookup[stack]

    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=tuple(f"sha256:{digest}" for digest in digests),
        # No directory name here: the manifest digest must depend only on the
        # slice bytes and the configuration (checksum-stability contract).
        notes="layered image stack; rows=+Y, columns=+X",
    )
    return build_material_label_volume(
        material_id=label,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        palette=palette,
        provenance=provenance,
        local_to_world=config.local_to_world,
    )
