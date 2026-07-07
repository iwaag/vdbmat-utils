"""Bulk material-id remapping (plan D5)."""

from collections.abc import Mapping

import numpy as np
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialPalette

from vdbmat_utils.core.errors import PaletteError
from vdbmat_utils.ops import OpsError

from ._common import BACKGROUND_ID, rebuild


def remap_materials(
    volume: MaterialLabelVolume,
    mapping: Mapping[int, int],
    *,
    definitions: Mapping[int, Mapping[str, str]] | None = None,
    prune_palette: bool = True,
) -> MaterialLabelVolume:
    """Rewrite material ids through a lookup table.

    ``mapping`` sends existing ids to new ids; unmapped ids pass through.
    Two source materials may collapse onto one id only if their definitions
    agree (after ``definitions`` overrides). ``definitions`` optionally
    renames materials by *new* id: ``{new_id: {"name": ...}}`` (role changes
    are not supported — background is structurally id 0 and only id 0).
    ``prune_palette`` drops palette entries that no longer label any voxel
    (background id 0 is always kept).
    """
    overrides = definitions or {}
    palette_ids = set(volume.palette.material_ids)
    for old_id in mapping:
        if old_id not in palette_ids:
            raise OpsError(f"remap: material_id {old_id} is not in the palette")
    for new_id in mapping.values():
        if not 0 <= int(new_id) <= np.iinfo(np.uint16).max:
            raise OpsError(f"remap: target id {new_id} is outside uint16 range")

    lut_size = max(palette_ids) + 1
    lut = np.arange(lut_size, dtype=np.uint16)
    for old_id, new_id in mapping.items():
        lut[old_id] = np.uint16(new_id)
    array = lut[volume.material_id]

    merged: dict[int, MaterialDefinition] = {}
    order: list[int] = []
    for definition in volume.palette.materials:
        new_id = int(mapping.get(definition.material_id, definition.material_id))
        override = overrides.get(new_id, {})
        candidate = MaterialDefinition(
            material_id=new_id,
            name=str(override.get("name", definition.name)),
            role=definition.role,
            external_id=definition.external_id,
        )
        existing = merged.get(new_id)
        if existing is None:
            merged[new_id] = candidate
            order.append(new_id)
        elif (existing.name, existing.role, existing.external_id) != (
            candidate.name,
            candidate.role,
            candidate.external_id,
        ):
            raise PaletteError(
                f"remap: materials {existing.name!r} and {definition.name!r} both "
                f"map to id {new_id} with conflicting definitions"
            )

    if prune_palette:
        used = set(np.unique(array).tolist())
        used.add(BACKGROUND_ID)
        order = [new_id for new_id in order if new_id in used]

    for new_id in overrides:
        if new_id not in merged:
            raise OpsError(
                f"remap: definitions override for id {new_id}, which no material "
                "maps to"
            )

    return rebuild(
        volume,
        array,
        palette=MaterialPalette.from_sequence([merged[i] for i in order]),
    )
