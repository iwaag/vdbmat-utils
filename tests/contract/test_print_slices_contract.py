"""Contract tests for the print-slices exporter.

Decoded-pixel digest pin (PNG file bytes are not pinned across
environments — see ``docs/print-slices.md``), same-environment double-run
byte equality, decode verification (round-trip half of the future
convert-image-stack contract), parameter sensitivity, seed independence,
and the color-count guard.
"""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
from vdbmat.core import MaterialDefinition, MaterialRole

from vdbmat_utils.cli.main import main
from vdbmat_utils.core import (
    build_material_label_volume,
    build_provenance,
    config_digest,
)
from vdbmat_utils.fixtures import build_fixture
from vdbmat_utils.image.png import read_indexed_png
from vdbmat_utils.io import write_asset
from vdbmat_utils.printer import PrintSlicesConfig
from vdbmat_utils.printer.sampler import build_sampling_plan, sample_slice

# --- representative input 1: single non-background material, HQ profile --

ANISOTROPIC_CONFIG = PrintSlicesConfig(
    dpi_x=600.0,
    dpi_y=300.0,
    layer_thickness_m=14e-6,
    palette={"1": [255, 0, 0]},
    min_slices=1,
)

# --- representative input 2: three non-background materials, HS profile --

MULTIMATERIAL_CONFIG = PrintSlicesConfig(
    dpi_x=600.0,
    dpi_y=300.0,
    layer_thickness_m=27e-6,
    max_materials=3,
    palette={"1": [255, 0, 0], "2": [0, 255, 0], "3": [0, 0, 255]},
    min_slices=1,
)

GOLDEN_ANISOTROPIC_PIXEL_DIGEST = (
    "b31be0c2eface6fc28d16aac0479c124ba3bd09cfad7c1545f835bed996120c0"
)
GOLDEN_MULTIMATERIAL_PIXEL_DIGEST = (
    "b1eea0ebb17b6dd5ae4e779d290130ea5084b488c1b560109be0132baa7e2b9d"
)


def _write_input(tmp_path: Path, preset: str, subdir: str) -> Path:
    volume = build_fixture(preset)
    return write_asset(volume, tmp_path / subdir, preset)


def _run_cli(
    manifest_path: Path, config: PrintSlicesConfig, tmp_path: Path, out_name: str
) -> Path:
    config_path = tmp_path / f"{out_name}.printslices.config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / out_name
    assert main(
        [
            "export-print-slices",
            str(manifest_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "demo",
        ]
    ) == 0
    return out_dir / "demo"


def _decoded_pixel_digest(out_dir: Path) -> str:
    hasher = hashlib.sha256()
    for png_path in sorted(out_dir.glob("*.png")):
        indices, _ = read_indexed_png(png_path)
        hasher.update(indices.tobytes())
    return hasher.hexdigest()


def test_double_run_is_byte_equal_anisotropic(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    first = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "a")
    second = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "b")
    files_first = sorted(p.name for p in first.iterdir())
    files_second = sorted(p.name for p in second.iterdir())
    assert files_first == files_second
    for name in files_first:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_golden_pixel_digest_anisotropic_hq(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    out_dir = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "out")
    assert _decoded_pixel_digest(out_dir) == GOLDEN_ANISOTROPIC_PIXEL_DIGEST


def test_golden_pixel_digest_multimaterial_hs(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "multimaterial", "input")
    out_dir = _run_cli(manifest_path, MULTIMATERIAL_CONFIG, tmp_path, "out")
    assert _decoded_pixel_digest(out_dir) == GOLDEN_MULTIMATERIAL_PIXEL_DIGEST


def test_decode_round_trip_matches_sampler_output(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "multimaterial", "input")
    config = MULTIMATERIAL_CONFIG
    out_dir = _run_cli(manifest_path, config, tmp_path, "out")

    from vdbmat.io import read_material_label_manifest

    volume = read_material_label_manifest(manifest_path)
    plan = build_sampling_plan(
        volume.geometry.shape_zyx, volume.geometry.voxel_size_xyz_m, config
    )

    ordered_ids = sorted(int(key) for key in config.palette)
    index_to_material_id = np.zeros(len(ordered_ids) + 1, dtype=np.uint16)
    for index, material_id in enumerate(ordered_ids, start=1):
        index_to_material_id[index] = material_id

    png_files = sorted(out_dir.glob("slice_*.png"))
    assert len(png_files) == plan.grid.n_slices
    for output_index, png_path in enumerate(png_files):
        decoded_indices, _ = read_indexed_png(png_path)
        decoded_indices = decoded_indices[: plan.grid.height, : plan.grid.width]
        reconstructed = index_to_material_id[decoded_indices]
        expected = sample_slice(volume.material_id, plan, output_index)
        np.testing.assert_array_equal(reconstructed, expected)


def test_sensitivity_dpi_x(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    changed = dataclasses.replace(ANISOTROPIC_CONFIG, dpi_x=300.0)
    assert config_digest(ANISOTROPIC_CONFIG) != config_digest(changed)
    out_a = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "a")
    out_b = _run_cli(manifest_path, changed, tmp_path, "b")
    assert _decoded_pixel_digest(out_a) != _decoded_pixel_digest(out_b)


def test_sensitivity_layer_thickness(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    changed = dataclasses.replace(ANISOTROPIC_CONFIG, layer_thickness_m=27e-6)
    out_a = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "a")
    out_b = _run_cli(manifest_path, changed, tmp_path, "b")
    assert _decoded_pixel_digest(out_a) != _decoded_pixel_digest(out_b)


def test_sensitivity_flip_z(tmp_path: Path) -> None:
    # The anisotropic fixture's z=0/z=2 slabs happen to be pixel-identical
    # (its label formula is periodic in z with period 2, and the derived
    # slice counts are symmetric), so flip_z would be a no-op there; the
    # multimaterial fixture (period-4 label, asymmetric slice counts)
    # actually exercises slice reordering.
    manifest_path = _write_input(tmp_path, "multimaterial", "input")
    changed = dataclasses.replace(MULTIMATERIAL_CONFIG, flip_z=True)
    out_a = _run_cli(manifest_path, MULTIMATERIAL_CONFIG, tmp_path, "a")
    out_b = _run_cli(manifest_path, changed, tmp_path, "b")
    assert _decoded_pixel_digest(out_a) != _decoded_pixel_digest(out_b)


def test_sensitivity_palette_rgb(tmp_path: Path) -> None:
    # Palette RGB only changes the PNG's colour table, never the index
    # array (indices encode material identity, not colour), so this checks
    # the decoded palette table rather than the pixel digest.
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    changed = dataclasses.replace(ANISOTROPIC_CONFIG, palette={"1": [0, 255, 0]})
    out_a = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "a")
    out_b = _run_cli(manifest_path, changed, tmp_path, "b")
    assert _decoded_pixel_digest(out_a) == _decoded_pixel_digest(out_b)

    first_png_a = sorted(out_a.glob("slice_*.png"))[0]
    first_png_b = sorted(out_b.glob("slice_*.png"))[0]
    _, palette_a = read_indexed_png(first_png_a)
    _, palette_b = read_indexed_png(first_png_b)
    assert palette_a[1] == (255, 0, 0)
    assert palette_b[1] == (0, 255, 0)
    assert palette_a != palette_b


def test_seed_independence(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path, "anisotropic", "input")
    changed = dataclasses.replace(ANISOTROPIC_CONFIG, seed=42)
    assert config_digest(ANISOTROPIC_CONFIG) != config_digest(changed)
    out_a = _run_cli(manifest_path, ANISOTROPIC_CONFIG, tmp_path, "a")
    out_b = _run_cli(manifest_path, changed, tmp_path, "b")
    assert _decoded_pixel_digest(out_a) == _decoded_pixel_digest(out_b)


def _seven_material_manifest(tmp_path: Path) -> Path:
    # 2x2x2 volume, one cell per id 0..7 (0 = background); 7 non-background
    # materials, deliberately one more than the PNG-method's 6-color limit.
    material_id = np.arange(8, dtype=np.uint16).reshape((2, 2, 2))
    materials = tuple(
        MaterialDefinition(
            material_id=i,
            name=f"material-{i}",
            role=MaterialRole.BACKGROUND if i == 0 else MaterialRole.MATERIAL,
        )
        for i in range(8)
    )
    volume = build_material_label_volume(
        material_id=material_id,
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        palette=materials,
        provenance=build_provenance(
            generator="print-slices-contract-test",
            generator_version="0.0.0",
            config=PrintSlicesConfig(layer_thickness_m=1e-4, palette={"1": [1, 0, 0]}),
        ),
    )
    return write_asset(volume, tmp_path / "seven", "seven")


def test_seven_materials_exceeds_max_materials_before_publish(tmp_path: Path) -> None:
    import pytest

    from vdbmat_utils.printer import PrintSlicesError

    _seven_material_manifest(tmp_path)
    with pytest.raises(PrintSlicesError, match="max_materials"):
        PrintSlicesConfig(
            layer_thickness_m=1e-4,
            palette={str(i): [i * 30, 0, 0] for i in range(1, 8)},
            min_slices=1,
        )
    out_dir = tmp_path / "out"
    assert not out_dir.exists()
