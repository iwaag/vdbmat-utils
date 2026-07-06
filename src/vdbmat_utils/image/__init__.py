"""Labeled 2D image stacks → canonical material-label volumes.

Ported from the historical ``vdbmat/tools/image_stack_generator`` reference
generator (ADR-009 D2) and rebuilt on the shared ``core`` conventions:
frozen-dataclass configuration with canonical-JSON digest, provenance
assembly, and the common volume builder. Slices stack in ascending filename
order as z = 0, 1, …; image rows map to +Y and columns to +X. Every gray
value present in the slices must be declared — an undeclared value is an
error, never a guess.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class ImageStackError(VdbmatUtilsError):
    """A slice stack or its configuration violates the generator contract."""


from .stack import (  # noqa: E402
    ImageStackConfig,
    convert_image_stack,
    stack_identity,
)

__all__ = [
    "ImageStackConfig",
    "ImageStackError",
    "convert_image_stack",
    "stack_identity",
]
