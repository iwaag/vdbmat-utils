"""Contract test: repeated writes of the same volume are byte-equal.

See ``docs/determinism.md`` for the full determinism rules.
"""

import dataclasses
from pathlib import Path

import numpy as np
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole
from vdbmat.io import read_material_label_manifest

from vdbmat_utils.core import (
    GeneratorConfig,
    build_material_label_volume,
    build_provenance,
    rng_from_seed,
)
from vdbmat_utils.io import write_asset


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticConfig(GeneratorConfig):
    shape_zyx: tuple[int, int, int] = (4, 5, 6)
    material_count: int = 3


def generate_synthetic_volume(config: SyntheticConfig) -> MaterialLabelVolume:
    rng = rng_from_seed(config.seed)
    labels = rng.integers(
        0, config.material_count, size=config.shape_zyx, dtype=np.uint16
    )
    palette = tuple(
        MaterialDefinition(
            material_id=i,
            name=f"synthetic_{i}",
            role=MaterialRole.BACKGROUND if i == 0 else MaterialRole.MATERIAL,
        )
        for i in range(config.material_count)
    )
    return build_material_label_volume(
        material_id=labels,
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0004),
        palette=palette,
        provenance=build_provenance(
            generator="vdbmat-utils-synthetic",
            generator_version="0.1.0",
            config=config,
        ),
    )


def _write_run(directory: Path, seed: int) -> tuple[bytes, bytes]:
    config = SyntheticConfig(seed=seed)
    manifest_path = write_asset(generate_synthetic_volume(config), directory, "asset")
    payload_path = directory / "asset.material_id.npy"
    return manifest_path.read_bytes(), payload_path.read_bytes()


def test_repeated_writes_are_byte_equal(tmp_path: Path) -> None:
    manifest_a, payload_a = _write_run(tmp_path / "a", seed=7)
    manifest_b, payload_b = _write_run(tmp_path / "b", seed=7)
    assert manifest_a == manifest_b
    assert payload_a == payload_b


def test_different_seeds_differ(tmp_path: Path) -> None:
    _, payload_a = _write_run(tmp_path / "a", seed=7)
    _, payload_b = _write_run(tmp_path / "b", seed=8)
    assert payload_a != payload_b


def test_written_asset_round_trips(tmp_path: Path) -> None:
    config = SyntheticConfig(seed=7)
    volume = generate_synthetic_volume(config)
    manifest_path = write_asset(volume, tmp_path, "asset")
    loaded = read_material_label_manifest(manifest_path)
    assert np.array_equal(loaded.material_id, volume.material_id)
    assert loaded.geometry.shape_zyx == volume.geometry.shape_zyx
    assert loaded.geometry.voxel_size_xyz_m == volume.geometry.voxel_size_xyz_m
