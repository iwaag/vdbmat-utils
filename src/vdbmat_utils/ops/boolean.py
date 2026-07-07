"""Boolean composition of two label volumes (plan D5)."""

import numpy as np
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError

from ._common import BACKGROUND_ID, rebuild, require_matching_geometry
from .palette import merge_palettes

_MODES = ("union", "intersect", "subtract")


def compose(
    base: MaterialLabelVolume,
    overlay: MaterialLabelVolume,
    *,
    mode: str,
) -> MaterialLabelVolume:
    """Combine two volumes that share the exact same geometry.

    Foreground means "not background (id 0)". Modes:

    - ``union`` — overlay foreground wins over base ("last writer wins");
    - ``intersect`` — keep base labels only where overlay is foreground;
    - ``subtract`` — clear base labels where overlay is foreground.

    Union merges the palettes (conflicting shared ids are an error pointing
    at ``remap-materials``); intersect and subtract keep the base palette.
    Provenance passes through from ``base``.
    """
    if mode not in _MODES:
        raise OpsError(f"compose mode must be one of {_MODES}, got {mode!r}")
    require_matching_geometry(base, overlay, other_name="overlay")
    overlay_foreground = overlay.material_id != BACKGROUND_ID
    background = np.uint16(BACKGROUND_ID)
    if mode == "union":
        array = np.where(overlay_foreground, overlay.material_id, base.material_id)
        return rebuild(
            base,
            array.astype(np.uint16),
            palette=merge_palettes(base.palette, overlay.palette),
        )
    if mode == "intersect":
        array = np.where(overlay_foreground, base.material_id, background)
    else:  # subtract
        array = np.where(overlay_foreground, background, base.material_id)
    return rebuild(base, array.astype(np.uint16))
