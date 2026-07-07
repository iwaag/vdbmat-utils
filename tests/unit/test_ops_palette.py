"""Unit tests for palette merging (plan Step 0.2)."""

import pytest
from vdbmat.core import MaterialDefinition, MaterialPalette, MaterialRole

from vdbmat_utils.core import PaletteError
from vdbmat_utils.ops import merge_palettes


def _palette(*entries: tuple[int, str]) -> MaterialPalette:
    definitions = [
        MaterialDefinition(material_id=0, name="air", role=MaterialRole.BACKGROUND)
    ]
    definitions += [
        MaterialDefinition(
            material_id=material_id, name=name, role=MaterialRole.MATERIAL
        )
        for material_id, name in entries
    ]
    return MaterialPalette.from_sequence(definitions)


def test_merge_disjoint_and_matching_entries() -> None:
    merged = merge_palettes(_palette((1, "a"), (2, "b")), _palette((2, "b"), (3, "c")))
    assert merged.material_ids == (0, 1, 2, 3)
    assert merged.by_id(3).name == "c"


def test_merge_conflicting_shared_id_is_an_error() -> None:
    with pytest.raises(PaletteError, match=r"remap-materials"):
        merge_palettes(_palette((1, "a")), _palette((1, "different")))
