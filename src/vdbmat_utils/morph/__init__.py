"""Sparse key-slice morphing → canonical material-label volumes (plan D4).

A slice directory declares labeled key slices at explicit z indices
(``slice_0000.pgm``, ``slice_0008.pgm``, …); intermediate slices are
interpolated per label through signed distance fields, so material ids are
never numerically interpolated — all smoothness lives in
``vdbmat_utils.fields``.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class MorphError(VdbmatUtilsError):
    """A morph key-slice stack or its configuration violates the contract."""


from .interpolate import MorphStackConfig, morph_stack  # noqa: E402

__all__ = [
    "MorphError",
    "MorphStackConfig",
    "morph_stack",
]
