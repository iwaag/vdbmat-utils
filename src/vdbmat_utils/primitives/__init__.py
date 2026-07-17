"""Deterministic transparent-block + opaque-primitive-array generator.

A minimal generator method for designlab Phase 1: build a material-label
volume of a transparent base block containing an A x B x C grid of opaque
cube or sphere inclusions, from one flat config with no input files. See
``docs/primitive-arrays.md`` for the config shape and sampling rules.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class PrimitiveArrayError(VdbmatUtilsError):
    """A primitive-array config or derived grid violates the contract."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


from .generator import generate_primitive_array  # noqa: E402
from .types import BUILTIN_MATERIAL_IDS, PrimitiveArrayConfig  # noqa: E402

__all__ = [
    "BUILTIN_MATERIAL_IDS",
    "PrimitiveArrayConfig",
    "PrimitiveArrayError",
    "generate_primitive_array",
]
