"""Unit tests for the print-slices config contract."""

import json

import pytest

from vdbmat_utils.core import ConfigError, config_digest
from vdbmat_utils.printer import PrintSlicesConfig, PrintSlicesError


def _base_kwargs() -> dict:
    return dict(
        layer_thickness_m=14e-6,
        palette={"1": [255, 0, 0], "2": [0, 255, 0]},
    )


def _config(**overrides: object) -> PrintSlicesConfig:
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    return PrintSlicesConfig(**kwargs)


# --- defaults and round-trip -------------------------------------------------


def test_defaults_are_accepted() -> None:
    config = _config()
    assert config.dpi_x == 600.0
    assert config.dpi_y == 300.0
    assert config.max_materials == 6
    assert config.background_rgb == (0, 0, 0)
    assert config.printer_x_axis == "x"
    assert config.printer_y_axis == "y"
    assert config.flip_x is False
    assert config.flip_y is False
    assert config.flip_z is False
    assert config.name_prefix == "slice_"
    assert config.index_start == 0
    assert config.min_slices == 30
    assert config.max_total_pixels == 4_000_000_000
    assert config.palette == {"1": (255, 0, 0), "2": (0, 255, 0)}


def test_valid_config_round_trips() -> None:
    config = _config()
    text = config.to_json()
    restored = PrintSlicesConfig.from_json(text)
    assert restored == config
    assert restored.to_json() == text


def test_unknown_field_rejected() -> None:
    payload = _config().to_json()
    data = json.loads(payload)
    data["bogus"] = 1
    with pytest.raises(ConfigError, match="unknown configuration fields"):
        PrintSlicesConfig.from_json(json.dumps(data))


def test_config_digest_stable_for_equal_configs() -> None:
    a = _config()
    b = _config()
    assert config_digest(a) == config_digest(b)


def test_config_digest_changes_with_seed() -> None:
    a = _config(seed=0)
    b = _config(seed=1)
    assert config_digest(a) != config_digest(b)


# --- rejected configs ---------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        dict(dpi_x=0.0),
        dict(dpi_x=-1.0),
        dict(dpi_y=0.0),
        dict(layer_thickness_m=0.0),
        dict(layer_thickness_m=-1e-6),
        dict(max_materials=0),
        dict(max_materials=7),
        dict(printer_x_axis="z"),
        dict(printer_y_axis="z"),
        dict(printer_x_axis="x", printer_y_axis="x"),
        dict(background_rgb=(0, 0)),
        dict(background_rgb=(-1, 0, 0)),
        dict(background_rgb=(256, 0, 0)),
        dict(background_rgb=(0.5, 0, 0)),
        dict(palette={}),
        dict(palette={"0": [1, 2, 3]}),
        dict(palette={"a": [1, 2, 3]}),
        dict(palette={"1": [1, 2]}),
        dict(palette={"1": [-1, 0, 0]}),
        dict(palette={"1": [256, 0, 0]}),
        # duplicate RGB across two palette entries
        dict(palette={"1": [1, 2, 3], "2": [1, 2, 3]}),
        # palette entry collides with background_rgb
        dict(palette={"1": [0, 0, 0]}),
        dict(name_prefix=""),
        dict(index_start=-1),
        dict(min_slices=0),
        dict(max_total_pixels=0),
        # palette larger than max_materials
        dict(
            palette={str(i): [i, 0, 0] for i in range(1, 8)},
            max_materials=6,
        ),
    ],
)
def test_rejected_configs(overrides: dict) -> None:
    with pytest.raises(PrintSlicesError):
        _config(**overrides)


def test_axis_swap_is_accepted() -> None:
    config = _config(printer_x_axis="y", printer_y_axis="x")
    assert config.printer_x_axis == "y"
    assert config.printer_y_axis == "x"


def test_palette_exactly_at_max_materials_is_accepted() -> None:
    config = _config(
        palette={str(i): [i, 0, 0] for i in range(1, 7)},
        max_materials=6,
    )
    assert len(config.palette) == 6
