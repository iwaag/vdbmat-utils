"""Unit tests for remap_materials (plan Step 0.2)."""

from collections.abc import Callable

import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.core import PaletteError
from vdbmat_utils.ops import OpsError, remap_materials

VolumeFactory = Callable[..., MaterialLabelVolume]

_SOURCE = [[[0, 1, 2], [3, 1, 0]]]


def test_remap_rewrites_ids_and_palette(make_volume: VolumeFactory) -> None:
    result = remap_materials(make_volume(_SOURCE), {1: 5, 2: 1})
    assert result.material_id.tolist() == [[[0, 5, 1], [3, 5, 0]]]
    assert result.palette.by_id(5).name == "resin_a"
    assert result.palette.by_id(1).name == "resin_b"


def test_remap_collapse_identical_definitions(make_volume: VolumeFactory) -> None:
    volume = make_volume(
        _SOURCE,
        palette=(
            (0, "air", "background"),
            (1, "resin", "material"),
            (2, "resin", "material"),
            (3, "other", "material"),
        ),
    )
    result = remap_materials(volume, {2: 1})
    assert result.material_id.tolist() == [[[0, 1, 1], [3, 1, 0]]]
    assert result.palette.material_ids == (0, 1, 3)


def test_remap_collision_with_conflicting_definitions(
    make_volume: VolumeFactory,
) -> None:
    with pytest.raises(PaletteError, match=r"conflicting definitions"):
        remap_materials(make_volume(_SOURCE), {2: 1})


def test_remap_unknown_source_id_is_an_error(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"material_id 7"):
        remap_materials(make_volume(_SOURCE), {7: 1})


def test_remap_prunes_unused_palette_entries(make_volume: VolumeFactory) -> None:
    volume = make_volume([[[0, 1, 1]]])  # materials 2 and 3 declared but unused
    assert remap_materials(volume, {}).palette.material_ids == (0, 1)
    assert remap_materials(
        volume, {}, prune_palette=False
    ).palette.material_ids == (0, 1, 2, 3)


def test_remap_rename_via_definitions(make_volume: VolumeFactory) -> None:
    result = remap_materials(
        make_volume(_SOURCE), {1: 4}, definitions={4: {"name": "glass"}}
    )
    assert result.palette.by_id(4).name == "glass"


def test_remap_definitions_for_unused_id_is_an_error(
    make_volume: VolumeFactory,
) -> None:
    with pytest.raises(OpsError, match=r"id 9"):
        remap_materials(make_volume(_SOURCE), {}, definitions={9: {"name": "x"}})
