"""Unit tests for vdbmat_utils.core conventions."""

import dataclasses

import numpy as np
import pytest
from vdbmat.core import (
    MaterialDefinition,
    MaterialRole,
    Provenance,
    SchemaIdentity,
    SchemaVersion,
    VolumeValidationError,
)

from vdbmat_utils.core import (
    CompatibilityError,
    ConfigError,
    GeneratorConfig,
    GeometryError,
    PaletteError,
    build_material_label_volume,
    build_provenance,
    config_digest,
    require_compatible_volume_schema,
    rng_from_seed,
    spawn_rngs,
)


@dataclasses.dataclass(frozen=True, slots=True)
class DemoConfig(GeneratorConfig):
    size: int = 4
    threshold: float = 0.5
    label: str = "demo"


def demo_palette() -> tuple[MaterialDefinition, ...]:
    return (
        MaterialDefinition(material_id=0, name="void", role=MaterialRole.BACKGROUND),
        MaterialDefinition(
            material_id=1, name="resin_clear", role=MaterialRole.MATERIAL
        ),
    )


def demo_provenance() -> Provenance:
    return build_provenance(
        generator="vdbmat-utils-tests",
        generator_version="0.1.0",
        config=DemoConfig(seed=7),
    )


class TestConfig:
    def test_canonical_json_round_trip(self) -> None:
        config = DemoConfig(seed=3, size=8, threshold=0.25, label="a")
        assert DemoConfig.from_json(config.to_json()) == config

    def test_digest_is_stable_and_seed_sensitive(self) -> None:
        assert config_digest(DemoConfig(seed=1)) == config_digest(DemoConfig(seed=1))
        assert config_digest(DemoConfig(seed=1)) != config_digest(DemoConfig(seed=2))
        assert config_digest(DemoConfig()).startswith("sha256:")

    def test_non_finite_float_rejected(self) -> None:
        with pytest.raises(ConfigError):
            DemoConfig(threshold=float("nan")).to_json()

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ConfigError):
            DemoConfig.from_json('{"seed": 1, "bogus": 2}')


class TestSeeds:
    def test_same_seed_same_stream(self) -> None:
        a = rng_from_seed(42).random(8)
        b = rng_from_seed(42).random(8)
        assert np.array_equal(a, b)

    def test_spawn_streams_are_independent_of_count(self) -> None:
        first = spawn_rngs(rng_from_seed(1), 1)[0].random(4)
        again = spawn_rngs(rng_from_seed(1), 3)[0].random(4)
        assert np.array_equal(first, again)

    def test_invalid_seed_rejected(self) -> None:
        with pytest.raises(ConfigError):
            rng_from_seed(-1)
        with pytest.raises(ConfigError):
            rng_from_seed(True)


class TestCompat:
    def test_pinned_vdbmat_schema_is_supported(self) -> None:
        require_compatible_volume_schema()

    def test_unsupported_major_rejected(self) -> None:
        future = SchemaIdentity(
            name="vdbmat.volume", version=SchemaVersion(major=2, minor=0, patch=0)
        )
        with pytest.raises(CompatibilityError):
            require_compatible_volume_schema(future)


class TestBuilder:
    def test_builds_valid_volume(self) -> None:
        labels = np.zeros((2, 3, 4), dtype=np.uint16)
        labels[0, 0, 0] = 1
        volume = build_material_label_volume(
            material_id=labels,
            voxel_size_xyz_m=(0.0001, 0.0002, 0.0004),
            palette=demo_palette(),
            provenance=demo_provenance(),
        )
        assert volume.geometry.shape_zyx == (2, 3, 4)

    def test_wrong_dtype_rejected(self) -> None:
        with pytest.raises(GeometryError):
            build_material_label_volume(
                material_id=np.zeros((2, 2, 2), dtype=np.int32),
                voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
                palette=demo_palette(),
                provenance=demo_provenance(),
            )

    def test_empty_palette_rejected(self) -> None:
        with pytest.raises(PaletteError):
            build_material_label_volume(
                material_id=np.zeros((2, 2, 2), dtype=np.uint16),
                voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
                palette=(),
                provenance=demo_provenance(),
            )

    def test_undeclared_label_passes_through_vdbmat_error(self) -> None:
        labels = np.full((2, 2, 2), 9, dtype=np.uint16)
        with pytest.raises(VolumeValidationError):
            build_material_label_volume(
                material_id=labels,
                voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
                palette=demo_palette(),
                provenance=demo_provenance(),
            )
