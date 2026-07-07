"""Continuous scalar fields, kept structurally apart from label volumes.

Plan D3: material-id arrays must never be interpolated as numeric
intensities. All smooth math (distance transforms, interpolation) happens on
``ScalarField`` data; the only sanctioned scalar→label conversion is
``quantize_to_labels``.
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from vdbmat_utils.core.errors import VdbmatUtilsError

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FieldError(VdbmatUtilsError):
    """A scalar-field input violates its contract."""


@dataclasses.dataclass(frozen=True)
class ScalarField:
    """A dense float64 scalar field on a canonical ``z, y, x`` grid.

    Carries grid geometry but no material palette — a field has no material
    semantics until it is quantized.
    """

    values: npt.NDArray[np.float64]
    voxel_size_xyz_m: tuple[float, float, float]
    local_to_world: tuple[tuple[float, ...], ...] = _IDENTITY

    def __post_init__(self) -> None:
        array = np.asarray(self.values)
        if array.ndim != 3:
            raise FieldError(
                f"field values must be a 3-D array in z, y, x order, "
                f"got {array.ndim}-D"
            )
        if array.dtype != np.float64:
            raise FieldError(f"field dtype must be float64, got {array.dtype}")
        if len(self.voxel_size_xyz_m) != 3 or any(
            not (float(v) > 0) for v in self.voxel_size_xyz_m
        ):
            raise FieldError("voxel_size_xyz_m must be 3 positive components")


from .edt import signed_distance, squared_edt  # noqa: E402
from .quantize import quantize_to_labels  # noqa: E402

__all__ = [
    "FieldError",
    "ScalarField",
    "quantize_to_labels",
    "signed_distance",
    "squared_edt",
]
