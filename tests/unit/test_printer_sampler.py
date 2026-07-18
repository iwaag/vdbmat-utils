"""Unit tests for print-slices output-grid derivation and sampling."""

import numpy as np
import pytest

from vdbmat_utils.printer import PrintSlicesConfig, PrintSlicesError
from vdbmat_utils.printer.sampler import (
    build_sampling_plan,
    derive_output_grid,
    sample_slice,
)


def _config(**overrides: object) -> PrintSlicesConfig:
    kwargs = dict(
        layer_thickness_m=14e-6,
        palette={"1": [255, 0, 0]},
        min_slices=1,
    )
    kwargs.update(overrides)
    return PrintSlicesConfig(**kwargs)


# --- grid derivation ---------------------------------------------------


def test_exact_ratio_shape() -> None:
    # dpi_y=300 -> pitch_y = 25.4/300 mm exactly; voxel_size_y set to that
    # pitch so the ratio is an exact integer.
    pitch_y = 0.0254 / 300.0
    config = _config(dpi_x=600.0, dpi_y=300.0, layer_thickness_m=27e-6)
    grid = derive_output_grid(
        shape_zyx=(2, 5, 4),
        voxel_size_xyz_m=(0.0254 / 600.0, pitch_y, 27e-6),
        config=config,
    )
    assert grid.width == 4
    assert grid.height == 5
    assert grid.n_slices == 2


def test_non_integer_ratio_isotropic_source() -> None:
    # Isotropic 100um source resampled onto 600/300 dpi + 27um profile.
    config = _config(dpi_x=600.0, dpi_y=300.0, layer_thickness_m=27e-6)
    voxel_size = (100e-6, 100e-6, 100e-6)
    grid = derive_output_grid(
        shape_zyx=(3, 2, 2), voxel_size_xyz_m=voxel_size, config=config
    )
    pitch_x = 0.0254 / 600.0
    pitch_y = 0.0254 / 300.0
    # extent_x = 2 * 100e-6 = 200e-6; ceil(200e-6/pitch_x - eps)
    import math

    assert grid.width == math.ceil(200e-6 / pitch_x - 1e-6)
    assert grid.height == math.ceil(200e-6 / pitch_y - 1e-6)
    assert grid.n_slices == math.ceil(300e-6 / 27e-6 - 1e-6)


def test_min_slices_guard() -> None:
    config = _config(layer_thickness_m=1e-3, min_slices=30)
    with pytest.raises(PrintSlicesError, match="min_slices"):
        derive_output_grid(
            shape_zyx=(2, 2, 2),
            voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
            config=config,
        )


def test_max_total_pixels_guard_with_suggestion() -> None:
    config = _config(max_total_pixels=10)
    with pytest.raises(PrintSlicesError, match="max_total_pixels"):
        derive_output_grid(
            shape_zyx=(2, 2, 2),
            voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
            config=config,
        )


# --- sampling plan / index arrays ---------------------------------------


def _small_material_id() -> np.ndarray:
    # shape (z=1, y=2, x=3): distinguishable per-cell values.
    return np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint16)


def test_index_array_matches_hand_computed_case() -> None:
    # voxel size chosen equal to pitch so index == output index (identity).
    config = _config(dpi_x=600.0, dpi_y=300.0, layer_thickness_m=14e-6)
    pitch_x = 0.0254 / 600.0
    pitch_y = 0.0254 / 300.0
    plan = build_sampling_plan(
        shape_zyx=(1, 2, 3),
        voxel_size_xyz_m=(pitch_x, pitch_y, 14e-6),
        config=config,
    )
    assert list(plan.printer_x_source_index) == [0, 1, 2]
    assert list(plan.printer_y_source_index) == [0, 1]
    assert list(plan.z_source_index) == [0]

    material_id = _small_material_id()
    out = sample_slice(material_id, plan, 0)
    assert out.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_boundary_tie_uses_floor_low_side() -> None:
    # Two output cells whose centres straddle a source cell boundary exactly.
    # src_voxel_size = 1.0, pitch = 1.0 -> centers 0.5, 1.5 -> floor -> 0, 1
    # (no ties here); construct an explicit tie: pitch = 0.5, src size = 1.0
    # -> output cell 1 center = 1.0 * 0.5 + 0.5*0.5 = 0.75 -> floor(0.75)=0.
    # Use a case where center lands exactly on a source cell boundary:
    # src_voxel_size=2.0, pitch=1.0 -> output index 1 center=1.5 -> /2=0.75->0
    # output index 3 center=3.5 -> /2=1.75->1. Boundary exactly at center=2.0
    # requires pitch=... choose pitch=2.0, src_voxel_size=4.0: output index 1
    # center=(1+0.5)*2=3.0 -> 3.0/4=0.75->floor 0. output index 2
    # center=5.0 -> 5/4=1.25->1.
    # Directly test a value landing exactly on the boundary:
    # center/src_voxel_size == integer N means the physical centre sits on
    # the boundary between source cell N-1 and N; floor gives N (this
    # module's low-side convention is "high" numerically but consistent).
    centers_hit_boundary_pitch = 2.0
    src_voxel_size = 1.0
    # output index i=0 -> center=1.0 exactly on boundary between cell 0/1.
    config = _config(dpi_x=0.0254 / centers_hit_boundary_pitch, layer_thickness_m=14e-6)
    plan = build_sampling_plan(
        shape_zyx=(1, 1, 4),
        voxel_size_xyz_m=(src_voxel_size, 1.0, 14e-6),
        config=config,
    )
    # center for output index 0 = (0+0.5)*pitch = 1.0 -> exactly on boundary
    assert plan.printer_x_source_index[0] == 1


def test_clip_prevents_edge_overhang() -> None:
    # A derived output cell count that slightly overshoots physical extent
    # (due to ceil) must clip to the last valid source index, not go OOB.
    config = _config(dpi_x=600.0, layer_thickness_m=14e-6)
    pitch_x = 0.0254 / 600.0
    # extent slightly more than 2 cells worth so width derives to 3 but only
    # 2 source cells exist along x.
    voxel_size_x = pitch_x * 1.4
    plan = build_sampling_plan(
        shape_zyx=(1, 1, 2),
        voxel_size_xyz_m=(voxel_size_x, 1.0, 14e-6),
        config=config,
    )
    assert plan.printer_x_source_index.max() <= 1


@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("swap_axes", [False, True])
def test_axis_swap_and_flip_combinations(
    flip_x: bool, flip_y: bool, swap_axes: bool
) -> None:
    # 3x2x1 material_id (z=1,y=2,x=3), identity pitch so source==output index
    # space, to check full-array equality for every combination.
    pitch_x = 0.0254 / 600.0
    pitch_y = 0.0254 / 300.0
    material_id = _small_material_id()  # [[1,2,3],[4,5,6]] with shape (1,2,3)

    if not swap_axes:
        config = _config(
            dpi_x=600.0, dpi_y=300.0, layer_thickness_m=14e-6,
            printer_x_axis="x", printer_y_axis="y",
            flip_x=flip_x, flip_y=flip_y,
        )
        voxel_size = (pitch_x, pitch_y, 14e-6)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    else:
        config = _config(
            dpi_x=600.0, dpi_y=300.0, layer_thickness_m=14e-6,
            printer_x_axis="y", printer_y_axis="x",
            flip_x=flip_x, flip_y=flip_y,
        )
        # printer X axis now reads source y (2 cells), printer Y axis reads
        # source x (3 cells); voxel sizes swap accordingly.
        voxel_size = (pitch_y, pitch_x, 14e-6)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16).T

    if flip_x:
        expected = expected[:, ::-1]
    if flip_y:
        expected = expected[::-1, :]

    plan = build_sampling_plan(
        shape_zyx=(1, 2, 3), voxel_size_xyz_m=voxel_size, config=config
    )
    out = sample_slice(material_id, plan, 0)
    assert out.tolist() == expected.tolist()
