"""Round-trip contract: export-print-slices → convert-image-stack.

Fixes the roadmap's design principle 4 ("the round-trip is the correctness
bar") as a contract test. Every case is of the form export → derive the
``convert-image-stack`` config from the sidecar manifest → convert →
compare against the *definition* of what the exporter wrote (the
concatenated ``sample_slice()`` output), except the two cases carved out in
``docs/print-slices.md``'s risk notes as sampler-independent external
references (anisotropic resampling and the exact-pitch identity case),
which use an expectation computed without importing ``printer.sampler`` at
all — otherwise a shared sampler bug could pass this test on both sides.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
from vdbmat.core import MaterialDefinition, MaterialRole
from vdbmat.io import read_material_label_manifest

from vdbmat_utils.core import build_material_label_volume, build_provenance
from vdbmat_utils.fixtures import build_fixture
from vdbmat_utils.image import convert_image_stack
from vdbmat_utils.io import write_asset
from vdbmat_utils.printer import (
    PrintSlicesConfig,
    export_print_slices,
    image_stack_config_from_print_manifest,
)
from vdbmat_utils.printer.sampler import build_sampling_plan, sample_slice

MULTIMATERIAL_CONFIG = PrintSlicesConfig(
    dpi_x=600.0,
    dpi_y=300.0,
    layer_thickness_m=27e-6,
    palette={"1": [255, 0, 0], "2": [0, 255, 0], "3": [0, 0, 255]},
    min_slices=1,
)


def _round_trip(
    manifest_path: Path, config: PrintSlicesConfig, tmp_path: Path, name: str
):
    export_out = tmp_path / f"{name}-export"
    result = export_print_slices(manifest_path, config, export_out, "demo")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    derived_config = image_stack_config_from_print_manifest(manifest)
    volume = convert_image_stack(result.output_dir, derived_config)
    return volume, result, derived_config


def _expected_material_id(manifest_path: Path, config: PrintSlicesConfig) -> np.ndarray:
    volume = read_material_label_manifest(manifest_path)
    plan = build_sampling_plan(
        volume.geometry.shape_zyx, volume.geometry.voxel_size_xyz_m, config
    )
    return np.stack(
        [sample_slice(volume.material_id, plan, i) for i in range(plan.grid.n_slices)],
        axis=0,
    )


def _pitch_xyz(config: PrintSlicesConfig) -> tuple[float, float, float]:
    return (0.0254 / config.dpi_x, 0.0254 / config.dpi_y, config.layer_thickness_m)


# --- 1. exact round-trip match at the default axis mapping -----------------


def test_default_axes_round_trip_matches_sampler(tmp_path: Path) -> None:
    volume = build_fixture("multimaterial")
    manifest_path = write_asset(volume, tmp_path / "input", "demo")

    round_tripped, _, derived_config = _round_trip(
        manifest_path, MULTIMATERIAL_CONFIG, tmp_path, "default"
    )
    expected = _expected_material_id(manifest_path, MULTIMATERIAL_CONFIG)

    np.testing.assert_array_equal(round_tripped.material_id, expected)
    assert round_tripped.geometry.voxel_size_xyz_m == _pitch_xyz(MULTIMATERIAL_CONFIG)
    round_tripped_ids = {m.material_id for m in round_tripped.palette}
    source_ids = {m.material_id for m in volume.palette}
    assert round_tripped_ids == source_ids
    for material_id in source_ids:
        assert (
            round_tripped.palette.by_id(material_id).role
            == volume.palette.by_id(material_id).role
        )
    assert derived_config.format == "png"


# --- 2. axis swap / flip variants -------------------------------------------


def _variant_configs() -> dict[str, PrintSlicesConfig]:
    return {
        "swapped_axes": dataclasses.replace(
            MULTIMATERIAL_CONFIG, printer_x_axis="y", printer_y_axis="x"
        ),
        "flip_x": dataclasses.replace(MULTIMATERIAL_CONFIG, flip_x=True),
        "flip_y": dataclasses.replace(MULTIMATERIAL_CONFIG, flip_y=True),
        "flip_z": dataclasses.replace(MULTIMATERIAL_CONFIG, flip_z=True),
    }


def test_axis_swap_and_flip_variants_round_trip_match(tmp_path: Path) -> None:
    volume = build_fixture("multimaterial")
    manifest_path = write_asset(volume, tmp_path / "input", "demo")

    for variant_name, config in _variant_configs().items():
        round_tripped, _, _ = _round_trip(manifest_path, config, tmp_path, variant_name)
        expected = _expected_material_id(manifest_path, config)
        np.testing.assert_array_equal(
            round_tripped.material_id,
            expected,
            err_msg=f"variant {variant_name!r} mismatched",
        )
        assert round_tripped.geometry.voxel_size_xyz_m == _pitch_xyz(config)


# --- 3. anisotropic resampling sensitivity (sampler-independent expected) --


def test_anisotropic_resampling_matches_independent_nearest_neighbour(
    tmp_path: Path,
) -> None:
    # Isotropic 100 um source, default 600/300 dpi + 27 um layer: X gets
    # roughly twice the samples Y does per source voxel (600/300 dpi ratio).
    shape_zyx = (2, 3, 4)
    src_size_m = 100e-6
    nz, ny, nx = shape_zyx
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    material_id = ((z * 7 + y * 3 + x) % 4).astype(np.uint16)
    materials = tuple(
        MaterialDefinition(
            material_id=i,
            name=f"material-{i}",
            role=MaterialRole.BACKGROUND if i == 0 else MaterialRole.MATERIAL,
        )
        for i in range(4)
    )
    config_for_provenance = PrintSlicesConfig(
        layer_thickness_m=27e-6,
        palette={"1": [1, 0, 0], "2": [0, 1, 0], "3": [0, 0, 1]},
    )
    volume = build_material_label_volume(
        material_id=material_id,
        voxel_size_xyz_m=(src_size_m, src_size_m, src_size_m),
        palette=materials,
        provenance=build_provenance(
            generator="print-slices-roundtrip-contract-test",
            generator_version="0.0.0",
            config=config_for_provenance,
        ),
    )
    manifest_path = write_asset(volume, tmp_path / "input", "aniso")

    config = PrintSlicesConfig(
        dpi_x=600.0,
        dpi_y=300.0,
        layer_thickness_m=27e-6,
        palette={"1": [255, 0, 0], "2": [0, 255, 0], "3": [0, 0, 255]},
        min_slices=1,
    )
    round_tripped, result, _ = _round_trip(manifest_path, config, tmp_path, "aniso")

    pitch_x, pitch_y, pitch_z = _pitch_xyz(config)
    extent_x, extent_y, extent_z = nx * src_size_m, ny * src_size_m, nz * src_size_m
    width = int(np.ceil(extent_x / pitch_x - 1e-6))
    height = int(np.ceil(extent_y / pitch_y - 1e-6))
    n_slices = int(np.ceil(extent_z / pitch_z - 1e-6))

    x_index = np.clip(
        np.floor(((np.arange(width) + 0.5) * pitch_x) / src_size_m).astype(np.int64),
        0,
        nx - 1,
    )
    y_index = np.clip(
        np.floor(((np.arange(height) + 0.5) * pitch_y) / src_size_m).astype(np.int64),
        0,
        ny - 1,
    )
    z_index = np.clip(
        np.floor(((np.arange(n_slices) + 0.5) * pitch_z) / src_size_m).astype(np.int64),
        0,
        nz - 1,
    )
    expected = material_id[np.ix_(z_index, y_index, x_index)]

    np.testing.assert_array_equal(round_tripped.material_id, expected)
    # Roughly 2 samples in X per source voxel for every 1 in Y (600/300 dpi).
    assert width / nx > height / ny
    assert result.n_slices == n_slices


# --- 4. exact-pitch identity case ------------------------------------------


def test_exact_pitch_identity_round_trip(tmp_path: Path) -> None:
    dpi_x, dpi_y, layer_thickness_m = 600.0, 300.0, 27e-6
    pitch_x, pitch_y, pitch_z = 0.0254 / dpi_x, 0.0254 / dpi_y, layer_thickness_m
    shape_zyx = (2, 2, 2)
    nz, ny, nx = shape_zyx
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    material_id = ((z * 2 + y + x) % 2).astype(np.uint16)
    materials = (
        MaterialDefinition(material_id=0, name="air", role=MaterialRole.BACKGROUND),
        MaterialDefinition(material_id=1, name="resin", role=MaterialRole.MATERIAL),
    )
    config_for_provenance = PrintSlicesConfig(
        layer_thickness_m=layer_thickness_m, palette={"1": [1, 0, 0]}
    )
    volume = build_material_label_volume(
        material_id=material_id,
        voxel_size_xyz_m=(pitch_x, pitch_y, pitch_z),
        palette=materials,
        provenance=build_provenance(
            generator="print-slices-roundtrip-contract-test",
            generator_version="0.0.0",
            config=config_for_provenance,
        ),
    )
    manifest_path = write_asset(volume, tmp_path / "input", "identity")

    config = PrintSlicesConfig(
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        layer_thickness_m=layer_thickness_m,
        palette={"1": [255, 0, 0]},
        min_slices=1,
    )
    round_tripped, result, _ = _round_trip(manifest_path, config, tmp_path, "identity")

    assert result.width == nx
    assert result.height == ny
    assert result.n_slices == nz
    np.testing.assert_array_equal(round_tripped.material_id, material_id)


# --- 5. double-run: reconstructed manifest digest is stable -----------------


def test_double_run_stack_identity_matches(tmp_path: Path) -> None:
    from vdbmat_utils.image.stack import stack_identity

    volume = build_fixture("multimaterial")
    manifest_path = write_asset(volume, tmp_path / "input", "demo")

    first, _, _ = _round_trip(manifest_path, MULTIMATERIAL_CONFIG, tmp_path, "a")
    second, _, _ = _round_trip(manifest_path, MULTIMATERIAL_CONFIG, tmp_path, "b")

    assert stack_identity(first) == stack_identity(second)
    np.testing.assert_array_equal(first.material_id, second.material_id)
