"""Unit tests for the print-manifest → ImageStackConfig derivation helper."""

import json
from pathlib import Path

import pytest

from vdbmat_utils.io import write_asset
from vdbmat_utils.primitives import PrimitiveArrayConfig, generate_primitive_array
from vdbmat_utils.printer import (
    PrintSlicesConfig,
    PrintSlicesError,
    export_print_slices,
    image_stack_config_from_print_manifest,
)


def _write_real_manifest(tmp_path: Path) -> dict:
    """Run a real export and return the parsed sidecar manifest."""
    input_config = PrimitiveArrayConfig(
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        primitive="cube",
        counts_xyz=(1, 1, 1),
        primitive_size_m=2e-4,
        gap_m=0.0,
        margin_m=1e-4,
    )
    volume = generate_primitive_array(input_config)
    manifest_path = write_asset(volume, tmp_path / "input", "demo")

    export_config = PrintSlicesConfig(
        dpi_x=600.0,
        dpi_y=300.0,
        layer_thickness_m=27e-6,
        palette={"1": [255, 0, 0], "3": [0, 255, 0]},
        min_slices=1,
    )
    result = export_print_slices(manifest_path, export_config, tmp_path / "out", "demo")
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))


def test_derives_pitch_from_dpi_bit_exact(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    config = image_stack_config_from_print_manifest(manifest)
    assert config.voxel_size_xyz_m == (0.0254 / 600.0, 0.0254 / 300.0, 27e-6)
    assert config.format == "png"


def test_derives_levels_from_palette(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    config = image_stack_config_from_print_manifest(manifest)
    by_material_id = {entry["material_id"]: entry for entry in config.levels}
    assert 0 in by_material_id
    assert by_material_id[0]["role"] == "background"
    assert by_material_id[0]["rgb"] == [0, 0, 0]
    assert by_material_id[1]["rgb"] == [255, 0, 0]
    assert by_material_id[1]["role"] == "material"
    assert by_material_id[3]["rgb"] == [0, 255, 0]
    assert by_material_id[3]["role"] == "material"
    assert all(isinstance(entry["name"], str) for entry in config.levels)


def test_levels_sorted_by_material_id(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    config = image_stack_config_from_print_manifest(manifest)
    ids = [entry["material_id"] for entry in config.levels]
    assert ids == sorted(ids)


def test_rejects_wrong_format(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    manifest["format"] = "something.else"
    with pytest.raises(PrintSlicesError, match="format"):
        image_stack_config_from_print_manifest(manifest)


def test_rejects_wrong_format_version(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    manifest["format_version"] = "9.9.9"
    with pytest.raises(PrintSlicesError, match="format_version"):
        image_stack_config_from_print_manifest(manifest)


def test_rejects_missing_palette(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    del manifest["palette"]
    with pytest.raises(PrintSlicesError, match="palette"):
        image_stack_config_from_print_manifest(manifest)


def test_rejects_palette_entry_missing_rgb(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    del manifest["palette"]["0"]["rgb"]
    with pytest.raises(PrintSlicesError, match=r"palette\[0\]\.rgb"):
        image_stack_config_from_print_manifest(manifest)


def test_rejects_missing_printer_section(tmp_path: Path) -> None:
    manifest = _write_real_manifest(tmp_path)
    del manifest["printer"]["dpi_x"]
    with pytest.raises(PrintSlicesError, match=r"printer\.dpi_x"):
        image_stack_config_from_print_manifest(manifest)
