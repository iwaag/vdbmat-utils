"""Unit tests for the print-slices exporter (layout, atomicity, manifest)."""

import json
from pathlib import Path

import pytest

from vdbmat_utils.io import write_asset
from vdbmat_utils.primitives import PrimitiveArrayConfig, generate_primitive_array
from vdbmat_utils.printer import (
    PrintSlicesConfig,
    PrintSlicesError,
    export_print_slices,
)


def _write_input(tmp_path: Path) -> Path:
    config = PrimitiveArrayConfig(
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        primitive="cube",
        counts_xyz=(1, 1, 1),
        primitive_size_m=2e-4,
        gap_m=0.0,
        margin_m=1e-4,
    )
    volume = generate_primitive_array(config)
    manifest_path = write_asset(volume, tmp_path / "input", "demo")
    return manifest_path


def _config(**overrides: object) -> PrintSlicesConfig:
    kwargs = dict(
        layer_thickness_m=100e-6,
        palette={"1": [255, 0, 0], "3": [0, 255, 0]},
        min_slices=1,
    )
    kwargs.update(overrides)
    return PrintSlicesConfig(**kwargs)


def test_output_layout(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    result = export_print_slices(
        manifest_path, _config(), tmp_path / "out", "demo"
    )
    assert result.output_dir == tmp_path / "out" / "demo"
    files = sorted(p.name for p in result.output_dir.iterdir())
    assert "demo.printslices.json" in files
    png_files = [f for f in files if f.endswith(".png")]
    assert len(png_files) == result.n_slices
    assert png_files == sorted(png_files)
    # No leftover tmp directories.
    out_entries = (tmp_path / "out").iterdir()
    assert not any(p.name.startswith(".demo.tmp-") for p in out_entries)


def test_digits_derivation_default_four(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    result = export_print_slices(
        manifest_path, _config(), tmp_path / "out", "demo"
    )
    first_png = sorted(p.name for p in result.output_dir.glob("slice_*.png"))[0]
    assert first_png == "slice_0000.png"


def test_rejects_existing_output_dir(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    out_dir = tmp_path / "out"
    export_print_slices(manifest_path, _config(), out_dir, "demo")
    with pytest.raises(PrintSlicesError, match="already exists"):
        export_print_slices(manifest_path, _config(), out_dir, "demo")


def test_missing_palette_entry_leaves_no_output(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    # Palette omits material 3 (black-opaque-resin), present in the input.
    bad_config = _config(palette={"1": [255, 0, 0]})
    out_dir = tmp_path / "out"
    with pytest.raises(PrintSlicesError, match="material ids not declared"):
        export_print_slices(manifest_path, bad_config, out_dir, "demo")
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_extra_palette_entry_rejected(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    bad_config = _config(
        palette={"1": [255, 0, 0], "3": [0, 255, 0], "9": [1, 2, 3]}
    )
    with pytest.raises(PrintSlicesError, match="absent from the input"):
        export_print_slices(manifest_path, bad_config, tmp_path / "out", "demo")


def test_manifest_physical_dimensions_match_hand_calculation(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    config = _config(dpi_x=600.0, dpi_y=300.0, layer_thickness_m=100e-6)
    result = export_print_slices(manifest_path, config, tmp_path / "out", "demo")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    pitch_x_mm = 25.4 / 600.0
    pitch_y_mm = 25.4 / 300.0
    extent_mm = 4 * 1e-4 * 1000.0  # 4-cell cube, 1e-4 m voxels, in mm
    import math

    expected_width = math.ceil((extent_mm / pitch_x_mm) - 1e-3)
    expected_height = math.ceil((extent_mm / pitch_y_mm) - 1e-3)
    assert manifest["grid"]["width_px"] == expected_width
    assert manifest["grid"]["height_px"] == expected_height
    assert manifest["grid"]["physical_mm"]["x"] == pytest.approx(
        expected_width * pitch_x_mm
    )
    assert manifest["grid"]["physical_mm"]["y"] == pytest.approx(
        expected_height * pitch_y_mm
    )
    assert manifest["palette"]["1"]["name"] == "transparent-resin"
    assert manifest["palette"]["3"]["name"] == "black-opaque-resin"
    assert manifest["background_rgb"] == [0, 0, 0]


def test_double_run_is_byte_equal(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    config = _config()
    result_a = export_print_slices(manifest_path, config, tmp_path / "a", "demo")
    result_b = export_print_slices(manifest_path, config, tmp_path / "b", "demo")
    files_a = sorted(result_a.output_dir.iterdir())
    files_b = sorted(result_b.output_dir.iterdir())
    assert [f.name for f in files_a] == [f.name for f in files_b]
    for fa, fb in zip(files_a, files_b, strict=True):
        assert fa.read_bytes() == fb.read_bytes()


def test_material_pixel_counts_sum_to_total_pixels(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path)
    result = export_print_slices(
        manifest_path, _config(), tmp_path / "out", "demo"
    )
    total = result.n_slices * result.width * result.height
    assert sum(result.material_pixel_counts.values()) == total
