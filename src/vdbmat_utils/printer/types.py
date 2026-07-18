"""Configuration contract for the print-slices exporter."""

import dataclasses
from collections.abc import Mapping

from vdbmat_utils.core import GeneratorConfig
from vdbmat_utils.printer import PrintSlicesError

_AXES = ("x", "y")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PrintSlicesConfig(GeneratorConfig):
    """Flat configuration for a GrabCAD PNG-method slice export.

    ``palette`` maps a non-background ``material_id`` (as a decimal string,
    since canonical JSON requires string dict keys) to an RGB triplet. The
    background material (``material_id`` 0) is never a palette key; its
    colour is ``background_rgb``. ``seed`` is inherited from
    ``GeneratorConfig`` and reserved; this exporter uses no randomness.
    """

    layer_thickness_m: float
    palette: Mapping[str, tuple[int, int, int]]
    dpi_x: float = 600.0
    dpi_y: float = 300.0
    max_materials: int = 6
    background_rgb: tuple[int, int, int] = (0, 0, 0)
    printer_x_axis: str = "x"
    printer_y_axis: str = "y"
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    name_prefix: str = "slice_"
    index_start: int = 0
    min_slices: int = 30
    max_total_pixels: int = 4_000_000_000

    def __post_init__(self) -> None:
        _validate_positive_float("dpi_x", self.dpi_x)
        _validate_positive_float("dpi_y", self.dpi_y)
        _validate_positive_float("layer_thickness_m", self.layer_thickness_m)
        _validate_int_range("max_materials", self.max_materials, minimum=1, maximum=6)

        if self.printer_x_axis not in _AXES:
            raise PrintSlicesError(
                "printer_x_axis", f"must be one of {_AXES}, got {self.printer_x_axis!r}"
            )
        if self.printer_y_axis not in _AXES:
            raise PrintSlicesError(
                "printer_y_axis", f"must be one of {_AXES}, got {self.printer_y_axis!r}"
            )
        if self.printer_x_axis == self.printer_y_axis:
            raise PrintSlicesError(
                "printer_y_axis",
                f"must differ from printer_x_axis ({self.printer_x_axis!r})",
            )

        background_rgb = _validate_rgb("background_rgb", self.background_rgb)
        object.__setattr__(self, "background_rgb", background_rgb)

        palette = _validate_palette(self.palette, background_rgb)
        object.__setattr__(self, "palette", palette)

        if len(palette) > self.max_materials:
            raise PrintSlicesError(
                "palette",
                f"has {len(palette)} entries, exceeds max_materials "
                f"({self.max_materials})",
            )

        for field in ("flip_x", "flip_y", "flip_z"):
            if not isinstance(getattr(self, field), bool):
                raise PrintSlicesError(field, "must be a boolean")

        if not isinstance(self.name_prefix, str) or not self.name_prefix:
            raise PrintSlicesError("name_prefix", "must be a non-empty string")

        _validate_int_range("index_start", self.index_start, minimum=0)
        _validate_int_range("min_slices", self.min_slices, minimum=1)
        _validate_int_range("max_total_pixels", self.max_total_pixels, minimum=1)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    fvalue = float(value)
    return fvalue == fvalue and fvalue not in (float("inf"), float("-inf"))


def _validate_positive_float(field: str, value: object) -> None:
    if not _is_finite_number(value) or float(value) <= 0.0:  # type: ignore[arg-type]
        raise PrintSlicesError(field, "must be a finite number greater than zero")


def _validate_int_range(
    field: str, value: object, *, minimum: int, maximum: int | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrintSlicesError(field, "must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise PrintSlicesError(field, f"must be {bound}, got {value}")


def _validate_rgb(field: str, value: object) -> tuple[int, int, int]:
    values = tuple(value) if isinstance(value, (list, tuple)) else None
    if values is None or len(values) != 3:
        raise PrintSlicesError(field, "must contain exactly 3 integers")
    normalized = []
    for channel, item in zip(("r", "g", "b"), values, strict=True):
        valid = (
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
        )
        if not valid:
            raise PrintSlicesError(
                f"{field}.{channel}", "must be an integer in [0, 255]"
            )
        normalized.append(item)
    return (normalized[0], normalized[1], normalized[2])


def _validate_palette(
    value: object, background_rgb: tuple[int, int, int]
) -> dict[str, tuple[int, int, int]]:
    if not isinstance(value, Mapping) or not value:
        raise PrintSlicesError("palette", "must be a non-empty object")

    normalized: dict[str, tuple[int, int, int]] = {}
    seen_rgb: dict[tuple[int, int, int], str] = {background_rgb: "background_rgb"}
    for key, rgb in value.items():
        if not isinstance(key, str) or not key.isdigit():
            raise PrintSlicesError(
                "palette", f"keys must be decimal-string material ids, got {key!r}"
            )
        material_id = int(key)
        if material_id == 0:
            raise PrintSlicesError(
                "palette", "must not contain the background material id (0)"
            )
        normalized_rgb = _validate_rgb(f"palette[{key}]", rgb)
        conflict = seen_rgb.get(normalized_rgb)
        if conflict is not None:
            raise PrintSlicesError(
                f"palette[{key}]",
                f"RGB {normalized_rgb} duplicates {conflict}",
            )
        seen_rgb[normalized_rgb] = f"palette[{key}]"
        normalized[key] = normalized_rgb

    return normalized
