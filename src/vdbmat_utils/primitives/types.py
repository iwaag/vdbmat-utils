"""Configuration contract for the primitive-array generator."""

import dataclasses

from vdbmat_utils.core import GeneratorConfig
from vdbmat_utils.primitives import PrimitiveArrayError

#: Built-in, non-background material names usable as base or inclusion.
#: The id mapping is pinned to vdbmat's built-in materials (see
#: ``vdbmat.optics.config.phase0_provisional_mapping``). If vdbmat's
#: built-in set changes, this table, its tests, and the docs are updated in
#: the same change — no compatibility table is kept (see roadmap policy).
BUILTIN_MATERIAL_IDS: dict[str, int] = {
    "transparent-resin": 1,
    "white-resin": 2,
    "black-opaque-resin": 3,
}

_PRIMITIVES = ("cube", "sphere")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PrimitiveArrayConfig(GeneratorConfig):
    """Flat configuration for an A x B x C primitive-array volume.

    Grid shape is derived from ``counts_xyz``, ``primitive_size_m``,
    ``gap_m``, and ``margin_m`` — never given directly. ``gap_m`` and
    ``margin_m`` have no defaults: the config must state the intended
    spacing explicitly rather than rely on an assumed convention ("touching"
    and "one size apart" are equally plausible defaults). ``seed`` is
    inherited from ``GeneratorConfig`` and reserved; this generator uses no
    randomness and its output does not depend on it.
    """

    voxel_size_xyz_m: tuple[float, float, float]
    primitive: str
    counts_xyz: tuple[int, int, int]
    primitive_size_m: float
    gap_m: float
    margin_m: float
    base_material_name: str = "transparent-resin"
    inclusion_material_name: str = "black-opaque-resin"
    max_axis_cells: int = 256
    max_total_cells: int = 8_000_000

    def __post_init__(self) -> None:
        _validate_triplet("voxel_size_xyz_m", self.voxel_size_xyz_m, positive=True)
        object.__setattr__(
            self, "voxel_size_xyz_m", tuple(float(v) for v in self.voxel_size_xyz_m)
        )
        if self.primitive not in _PRIMITIVES:
            raise PrimitiveArrayError(
                "primitive", f"must be one of {_PRIMITIVES}, got {self.primitive!r}"
            )
        _validate_counts(self.counts_xyz)
        object.__setattr__(
            self, "counts_xyz", tuple(int(v) for v in self.counts_xyz)
        )
        _validate_number(
            "primitive_size_m", self.primitive_size_m, minimum=0.0, inclusive=False
        )
        _validate_number("gap_m", self.gap_m, minimum=0.0, inclusive=True)
        _validate_number("margin_m", self.margin_m, minimum=0.0, inclusive=True)
        _validate_material_name("base_material_name", self.base_material_name)
        _validate_material_name("inclusion_material_name", self.inclusion_material_name)
        if self.base_material_name == self.inclusion_material_name:
            raise PrimitiveArrayError(
                "inclusion_material_name",
                f"must differ from base_material_name ({self.base_material_name!r})",
            )
        _validate_positive_int("max_axis_cells", self.max_axis_cells)
        _validate_positive_int("max_total_cells", self.max_total_cells)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    fvalue = float(value)
    return fvalue == fvalue and fvalue not in (float("inf"), float("-inf"))


def _validate_number(
    field: str, value: object, *, minimum: float, inclusive: bool
) -> None:
    if not _is_finite_number(value):
        raise PrimitiveArrayError(field, "must be a finite number")
    fvalue = float(value)  # type: ignore[arg-type]
    ok = fvalue >= minimum if inclusive else fvalue > minimum
    if not ok:
        bound = "greater than or equal to" if inclusive else "greater than"
        raise PrimitiveArrayError(field, f"must be {bound} {minimum}")


def _validate_triplet(field: str, value: object, *, positive: bool) -> None:
    values = tuple(value) if isinstance(value, (list, tuple)) else None
    if values is None or len(values) != 3:
        raise PrimitiveArrayError(field, "must contain exactly 3 numbers")
    for axis, item in zip(("x", "y", "z"), values, strict=True):
        if not _is_finite_number(item) or (positive and float(item) <= 0.0):
            raise PrimitiveArrayError(
                f"{field}.{axis}", "must be finite and greater than zero"
            )


def _validate_counts(value: object) -> None:
    values = tuple(value) if isinstance(value, (list, tuple)) else None
    if values is None or len(values) != 3:
        raise PrimitiveArrayError("counts_xyz", "must contain exactly 3 integers")
    for axis, item in zip(("x", "y", "z"), values, strict=True):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise PrimitiveArrayError(f"counts_xyz.{axis}", "must be an integer >= 1")


def _validate_positive_int(field: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrimitiveArrayError(field, "must be a positive integer")


def _validate_material_name(field: str, value: object) -> None:
    if not isinstance(value, str) or value not in BUILTIN_MATERIAL_IDS:
        raise PrimitiveArrayError(
            field, f"must be one of {sorted(BUILTIN_MATERIAL_IDS)}, got {value!r}"
        )
