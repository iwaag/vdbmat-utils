"""Palette merge rules shared by composition operations (plan D5)."""

from vdbmat.core import MaterialDefinition, MaterialPalette

from vdbmat_utils.core.errors import PaletteError


def _identity(definition: MaterialDefinition) -> tuple[object, ...]:
    return (definition.name, definition.role, definition.external_id)


def merge_palettes(
    base: MaterialPalette, overlay: MaterialPalette
) -> MaterialPalette:
    """Merge two palettes by material id.

    A shared id merges only when name, role, and external id all match;
    a conflicting definition is an error — remap one side's ids first
    (``remap-materials``). Base entries keep their order; new overlay entries
    follow in overlay order.
    """
    merged: list[MaterialDefinition] = list(base.materials)
    by_id = {definition.material_id: definition for definition in merged}
    for definition in overlay.materials:
        existing = by_id.get(definition.material_id)
        if existing is None:
            merged.append(definition)
            by_id[definition.material_id] = definition
        elif _identity(existing) != _identity(definition):
            raise PaletteError(
                f"material_id {definition.material_id} means "
                f"{existing.name!r}/{existing.role.value} in the base palette but "
                f"{definition.name!r}/{definition.role.value} in the overlay; "
                "reconcile ids with remap-materials before composing"
            )
    return MaterialPalette.from_sequence(merged)
