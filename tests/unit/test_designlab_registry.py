"""Browser-free tests for the designlab generator-method registry."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "designlab"
sys.path.insert(0, str(EXAMPLES_DIR))

from designlab_registry import (  # noqa: E402
    PRIMITIVE_ARRAY_METHOD,
    REGISTRY,
    PrimitiveArrayFormBinding,
    method_by_id,
    method_for_config_path,
)

from vdbmat_utils.primitives import (  # noqa: E402
    PrimitiveArrayConfig,
    PrimitiveArrayError,
)


@dataclass
class _FakeWidget:
    value: Any


def _fake_binding(**overrides: Any) -> PrimitiveArrayFormBinding:
    defaults = dict(
        voxel_size_x=0.0001,
        voxel_size_y=0.0001,
        voxel_size_z=0.0001,
        primitive="cube",
        counts_x=2,
        counts_y=2,
        counts_z=2,
        primitive_size_m=0.0004,
        gap_m=0.0002,
        margin_m=0.0001,
        base_material_name="transparent-resin",
        inclusion_material_name="black-opaque-resin",
        max_axis_cells=256,
        max_total_cells=8_000_000,
        seed=0,
    )
    defaults.update(overrides)
    return PrimitiveArrayFormBinding(**{k: _FakeWidget(v) for k, v in defaults.items()})


def test_registry_enumerates_primitive_array() -> None:
    assert REGISTRY == (PRIMITIVE_ARRAY_METHOD,)
    assert PRIMITIVE_ARRAY_METHOD.method_id == "primitive-array"
    assert PRIMITIVE_ARRAY_METHOD.config_suffix == ".primarray.json"
    assert PRIMITIVE_ARRAY_METHOD.config_cls is PrimitiveArrayConfig


def test_method_by_id_round_trips() -> None:
    assert method_by_id("primitive-array") is PRIMITIVE_ARRAY_METHOD
    with pytest.raises(KeyError):
        method_by_id("no-such-method")


def test_method_for_config_path_matches_suffix() -> None:
    assert method_for_config_path(Path("demo.primarray.json")) is PRIMITIVE_ARRAY_METHOD
    assert method_for_config_path(Path("demo.formation.json")) is None
    assert method_for_config_path(Path("demo.json")) is None


def test_generator_argv_golden() -> None:
    argv = PRIMITIVE_ARRAY_METHOD.generator_argv(
        Path("/cfg/demo.primarray.json"), Path("/out"), "demo"
    )
    assert argv == [
        sys.executable,
        "-m",
        "vdbmat_utils.cli.main",
        "generate-primitive-array",
        "--config",
        "/cfg/demo.primarray.json",
        "--out",
        "/out",
        "--name",
        "demo",
    ]


def test_form_to_config_builds_expected_config() -> None:
    binding = _fake_binding()
    config = PRIMITIVE_ARRAY_METHOD.form_to_config(binding)
    assert config == PrimitiveArrayConfig(
        voxel_size_xyz_m=(0.0001, 0.0001, 0.0001),
        primitive="cube",
        counts_xyz=(2, 2, 2),
        primitive_size_m=0.0004,
        gap_m=0.0002,
        margin_m=0.0001,
        base_material_name="transparent-resin",
        inclusion_material_name="black-opaque-resin",
        max_axis_cells=256,
        max_total_cells=8_000_000,
        seed=0,
    )


def test_form_to_config_propagates_field_named_error_untouched() -> None:
    binding = _fake_binding(primitive="cylinder")
    with pytest.raises(PrimitiveArrayError) as excinfo:
        PRIMITIVE_ARRAY_METHOD.form_to_config(binding)
    assert excinfo.value.field_path == "primitive"


def test_config_to_form_then_form_to_config_round_trips() -> None:
    config = PrimitiveArrayConfig(
        voxel_size_xyz_m=(0.0002, 0.0003, 0.0004),
        primitive="sphere",
        counts_xyz=(3, 1, 2),
        primitive_size_m=0.0005,
        gap_m=0.0,
        margin_m=0.0001,
        base_material_name="white-resin",
        inclusion_material_name="black-opaque-resin",
        seed=7,
    )
    binding = _fake_binding()
    PRIMITIVE_ARRAY_METHOD.config_to_form(binding, config)
    round_tripped = PRIMITIVE_ARRAY_METHOD.form_to_config(binding)
    assert round_tripped == config
