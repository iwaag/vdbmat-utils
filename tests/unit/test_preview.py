"""Unit tests for the preview diagnostics (counts, ASCII slices, PGM slices).

The ASCII and PGM goldens use the ``multimaterial`` fixture preset, whose
label pattern ``(7z + 3y + x) % 4`` is asymmetric along every axis, so any
z/y/x transposition inside the preview code changes the golden text.
"""

from pathlib import Path

import numpy as np
import pytest
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from vdbmat_utils.core import build_material_label_volume, build_provenance
from vdbmat_utils.fixtures import FixtureConfig, build_fixture
from vdbmat_utils.preview import (
    PreviewError,
    material_counts,
    slice_ascii,
    slice_pgm,
)


@pytest.fixture(scope="module")
def multimaterial() -> MaterialLabelVolume:
    return build_fixture("multimaterial")


def test_material_counts_anisotropic() -> None:
    assert material_counts(build_fixture("anisotropic")) == {0: 30, 1: 30}


def test_material_counts_multimaterial(multimaterial: MaterialLabelVolume) -> None:
    assert material_counts(multimaterial) == {0: 15, 1: 15, 2: 15, 3: 15}


def test_material_counts_includes_unused_palette_entries() -> None:
    volume = build_material_label_volume(
        material_id=np.zeros((2, 2, 2), dtype=np.uint16),
        voxel_size_xyz_m=(0.001, 0.001, 0.001),
        palette=(
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(5, "resin_clear", MaterialRole.MATERIAL),
        ),
        provenance=build_provenance(
            generator="test", generator_version="0.0.0", config=FixtureConfig()
        ),
    )
    assert material_counts(volume) == {0: 8, 5: 0}


def test_slice_ascii_z_golden(multimaterial: MaterialLabelVolume) -> None:
    assert slice_ascii(multimaterial, "z", 2) == "\n".join(
        [
            "slice z=2  +x →  +y ↓",
            "23.",
            "123",
            ".12",
            "3.1",
        ]
    )


def test_slice_ascii_y_golden(multimaterial: MaterialLabelVolume) -> None:
    assert slice_ascii(multimaterial, "y", 2) == "\n".join(
        [
            "slice y=2  +x →  +z ↓",
            "23.",
            "123",
            ".12",
            "3.1",
            "23.",
        ]
    )


def test_slice_ascii_x_golden(multimaterial: MaterialLabelVolume) -> None:
    assert slice_ascii(multimaterial, "x", 1) == "\n".join(
        [
            "slice x=1  +y →  +z ↓",
            "1.32",
            ".321",
            "321.",
            "21.3",
            "1.32",
        ]
    )


def test_slice_pgm_byte_golden(
    multimaterial: MaterialLabelVolume, tmp_path: Path
) -> None:
    path = slice_pgm(multimaterial, "z", 2, tmp_path / "slice.pgm")
    # Palette ids 0..3 map to grays 0, 85, 170, 255 by ascending-id rank.
    expected_pixels = bytes([170, 255, 0, 85, 170, 255, 0, 85, 170, 255, 0, 85])
    assert path.read_bytes() == b"P5\n3 4\n255\n" + expected_pixels


def test_slice_index_out_of_range(multimaterial: MaterialLabelVolume) -> None:
    with pytest.raises(PreviewError, match="out of range"):
        slice_ascii(multimaterial, "z", 5)
    with pytest.raises(PreviewError, match="out of range"):
        slice_ascii(multimaterial, "x", -1)


def test_unknown_axis_rejected(multimaterial: MaterialLabelVolume) -> None:
    with pytest.raises(PreviewError, match="unknown axis"):
        slice_ascii(multimaterial, "w", 0)  # type: ignore[arg-type]
