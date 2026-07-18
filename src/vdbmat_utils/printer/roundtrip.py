"""Derive a ``convert-image-stack`` config from a print-slices manifest.

``export-print-slices`` writes a color-label PNG stack plus a sidecar
manifest (``<name>.printslices.json``). Reading that stack back through
``convert-image-stack`` needs an ``ImageStackConfig`` whose ``levels`` match
the manifest's ``palette`` (rgb → material_id/name/role, background
included) — this module derives that config mechanically instead of
requiring a hand-written mapping, so the round-trip contract
(``export-print-slices`` → ``convert-image-stack`` → printer-grid material
array) has no manual step that could silently drift from the exporter.
"""

from collections.abc import Mapping

from vdbmat_utils.image import ImageStackConfig
from vdbmat_utils.printer import PrintSlicesError

_EXPECTED_FORMAT = "vdbmat.print-slices"
_EXPECTED_FORMAT_VERSION = "1.0.0"


def image_stack_config_from_print_manifest(
    manifest: Mapping[str, object],
) -> ImageStackConfig:
    """Derive the ``ImageStackConfig`` that reads back a print-slices export.

    ``voxel_size_xyz_m`` is recomputed from the manifest's ``dpi_x``/
    ``dpi_y``/``layer_thickness_m`` using the exporter's own pitch formula
    (``0.0254 / dpi``), not read back from the manifest's ``pitch_*_mm``
    millimetre values — round-tripping through mm risks a float mismatch
    with the sampler's pitch that this bit-identical recomputation avoids.
    """
    _validate_format(manifest)
    printer = _require_mapping(manifest, "printer")
    dpi_x = _require_number(printer, "dpi_x", "printer.dpi_x")
    dpi_y = _require_number(printer, "dpi_y", "printer.dpi_y")
    layer_thickness_m = _require_number(
        printer, "layer_thickness_m", "printer.layer_thickness_m"
    )
    voxel_size_xyz_m = (0.0254 / dpi_x, 0.0254 / dpi_y, layer_thickness_m)

    levels = _derive_levels(manifest)

    return ImageStackConfig(
        voxel_size_xyz_m=voxel_size_xyz_m,
        levels=levels,
        format="png",
    )


def _validate_format(manifest: Mapping[str, object]) -> None:
    format_name = manifest.get("format")
    if format_name != _EXPECTED_FORMAT:
        raise PrintSlicesError(
            "format",
            f"expected {_EXPECTED_FORMAT!r}, got {format_name!r}",
        )
    format_version = manifest.get("format_version")
    if format_version != _EXPECTED_FORMAT_VERSION:
        raise PrintSlicesError(
            "format_version",
            f"expected {_EXPECTED_FORMAT_VERSION!r}, got {format_version!r}",
        )


def _derive_levels(
    manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    palette = _require_mapping(manifest, "palette")
    if not palette:
        raise PrintSlicesError("palette", "must be a non-empty object")

    entries: list[tuple[int, Mapping[str, object]]] = []
    for key, entry in palette.items():
        field = f"palette[{key}]"
        if not isinstance(key, str) or not key.isdigit():
            raise PrintSlicesError(
                field, f"key must be a decimal-string material id, got {key!r}"
            )
        if not isinstance(entry, Mapping):
            raise PrintSlicesError(field, "must be an object")
        name = entry.get("name")
        if not isinstance(name, str):
            raise PrintSlicesError(f"{field}.name", "must be a string")
        role = entry.get("role")
        if not isinstance(role, str):
            raise PrintSlicesError(f"{field}.role", "must be a string")
        rgb = entry.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            raise PrintSlicesError(
                f"{field}.rgb", "must be an array of 3 integers"
            )
        channels = []
        for channel_name, channel in zip(("r", "g", "b"), rgb, strict=True):
            if not isinstance(channel, int) or not 0 <= channel <= 255:
                raise PrintSlicesError(
                    f"{field}.rgb.{channel_name}",
                    "must be an integer in [0, 255]",
                )
            channels.append(channel)
        material_id = int(key)
        entries.append(
            (
                material_id,
                {
                    "rgb": channels,
                    "material_id": material_id,
                    "name": name,
                    "role": role,
                },
            )
        )

    entries.sort(key=lambda item: item[0])
    return tuple(entry for _, entry in entries)


def _require_mapping(
    manifest: Mapping[str, object], field: str
) -> Mapping[str, object]:
    value = manifest.get(field)
    if not isinstance(value, Mapping):
        raise PrintSlicesError(field, "must be an object")
    return value


def _require_number(mapping: Mapping[str, object], key: str, field: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PrintSlicesError(field, "must be a number")
    if float(value) <= 0.0:
        raise PrintSlicesError(field, "must be greater than zero")
    return float(value)
