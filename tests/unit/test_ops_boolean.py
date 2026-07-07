"""Unit tests for boolean composition (plan Step 1.3)."""

from collections.abc import Callable

import pytest
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.core import PaletteError
from vdbmat_utils.ops import OpsError, compose

VolumeFactory = Callable[..., MaterialLabelVolume]

_BASE = [[[1, 1, 0], [2, 0, 0]]]
_OVERLAY = [[[0, 3, 3], [3, 0, 0]]]


def test_union_overlay_foreground_wins(make_volume: VolumeFactory) -> None:
    result = compose(make_volume(_BASE), make_volume(_OVERLAY), mode="union")
    assert result.material_id.tolist() == [[[1, 3, 3], [3, 0, 0]]]


_AIR = (0, "air", "background")


def test_union_merges_palettes(make_volume: VolumeFactory) -> None:
    base = make_volume(
        _BASE, palette=(_AIR, (1, "a", "material"), (2, "b", "material"))
    )
    overlay = make_volume(_OVERLAY, palette=(_AIR, (3, "c", "material")))
    result = compose(base, overlay, mode="union")
    assert result.palette.material_ids == (0, 1, 2, 3)


def test_union_palette_conflict_names_remap(make_volume: VolumeFactory) -> None:
    base = make_volume(
        _BASE,
        palette=(
            _AIR,
            (1, "a", "material"),
            (2, "b", "material"),
            (3, "c", "material"),
        ),
    )
    overlay = make_volume(_OVERLAY, palette=(_AIR, (3, "not_c", "material")))
    with pytest.raises(PaletteError, match=r"remap-materials"):
        compose(base, overlay, mode="union")


def test_intersect_keeps_base_labels_inside_overlay(
    make_volume: VolumeFactory,
) -> None:
    result = compose(make_volume(_BASE), make_volume(_OVERLAY), mode="intersect")
    assert result.material_id.tolist() == [[[0, 1, 0], [2, 0, 0]]]


def test_subtract_clears_base_under_overlay(make_volume: VolumeFactory) -> None:
    result = compose(make_volume(_BASE), make_volume(_OVERLAY), mode="subtract")
    assert result.material_id.tolist() == [[[1, 0, 0], [0, 0, 0]]]


def test_geometry_mismatch_is_an_error(make_volume: VolumeFactory) -> None:
    shifted = make_volume(
        _OVERLAY,
        local_to_world=(
            (1.0, 0.0, 0.0, 0.001),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    with pytest.raises(OpsError, match=r"overlay\.local_to_world"):
        compose(make_volume(_BASE), shifted, mode="union")


def test_unknown_mode_is_an_error(make_volume: VolumeFactory) -> None:
    with pytest.raises(OpsError, match=r"mode"):
        compose(make_volume(_BASE), make_volume(_OVERLAY), mode="xor")
