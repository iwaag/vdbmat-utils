"""Masking a label volume with a second label volume (plan D5)."""

import numpy as np
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import OpsError

from ._common import BACKGROUND_ID, rebuild, require_matching_geometry

_MODES = ("keep", "clear")


def apply_mask(
    volume: MaterialLabelVolume,
    mask: MaterialLabelVolume,
    *,
    mode: str = "keep",
    fill_material_id: int | None = None,
) -> MaterialLabelVolume:
    """Keep or clear the voxels selected by ``mask`` (nonzero = selected).

    ``mode="keep"`` fills everything outside the selection with
    ``fill_material_id`` (default: background); ``mode="clear"`` fills the
    selection instead. The mask must share the volume's exact geometry —
    there is no auto-alignment in Phase 2.
    """
    if mode not in _MODES:
        raise OpsError(f"apply_mask mode must be one of {_MODES}, got {mode!r}")
    require_matching_geometry(volume, mask, other_name="mask")
    if fill_material_id is None:
        fill = BACKGROUND_ID
    else:
        fill = int(fill_material_id)
        if fill not in volume.palette.material_ids:
            raise OpsError(
                f"apply_mask fill_material_id {fill} is not in the palette"
            )
    selected = mask.material_id != 0
    if mode == "clear":
        selected = ~selected
    array = np.where(selected, volume.material_id, np.uint16(fill)).astype(np.uint16)
    return rebuild(volume, array)
