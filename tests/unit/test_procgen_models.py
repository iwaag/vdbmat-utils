import json
from pathlib import Path

import numpy as np
import pytest
from vdbmat.optics import load_optical_mapping

from vdbmat_utils.cli.main import main
from vdbmat_utils.core.errors import ConfigError, PaletteError
from vdbmat_utils.procgen.models import (
    FormationConfig,
    generate_formation,
    write_formation,
)


def _base_config(**overrides: object) -> FormationConfig:
    payload = {
        "seed": 4,
        "shape_zyx": [4, 3, 2],
        "voxel_size_xyz_m": [1.0, 1.0, 1.0],
        "palette": [
            {"material_id": 0, "name": "air", "role": "background"},
            {"material_id": 1, "name": "transparent-resin"},
            {"material_id": 2, "name": "white-resin"},
        ],
        "layers": [{"kind": "host", "material_id": 1}],
    }
    payload.update(overrides)
    return FormationConfig.from_json(json.dumps(payload))


def test_unwarped_strata_has_exact_expected_bands() -> None:
    config = _base_config(
        layers=[
            {"kind": "host", "material_id": 1},
            {
                "kind": "strata",
                "axis": "z",
                "thickness_m": 2.0,
                "material_ids": [1, 2],
            },
        ]
    )
    result = generate_formation(config, name="bands")
    expected = np.empty((4, 3, 2), dtype=np.uint16)
    expected[0:2, :, :] = 1
    expected[2:4, :, :] = 2
    np.testing.assert_array_equal(result.volume.material_id, expected)


def test_painter_precedence_later_layer_overwrites() -> None:
    config = _base_config(
        shape_zyx=[2, 1, 3],
        layers=[
            {"kind": "host", "material_id": 1},
            {
                "kind": "veins",
                "axis": "x",
                "offset_m": 1.5,
                "width_m": 1.0,
                "material_id": 2,
            },
        ],
    )
    labels = generate_formation(config, name="vein").volume.material_id
    np.testing.assert_array_equal(labels[:, :, 1], np.full((2, 1), 2))
    np.testing.assert_array_equal(labels[:, :, [0, 2]], np.full((2, 1, 2), 1))


def test_palette_violation_names_layer() -> None:
    config = _base_config(
        layers=[
            {"kind": "host", "material_id": 1},
            {
                "kind": "veins",
                "axis": "x",
                "offset_m": 0.5,
                "width_m": 1.0,
                "material_id": 99,
            },
        ]
    )
    with pytest.raises(PaletteError, match=r"layers\[1\]"):
        generate_formation(config, name="bad")


def test_all_builtin_palette_writes_no_mapping(tmp_path: Path) -> None:
    written = write_formation(_base_config(), out=tmp_path, name="builtin")
    assert written.mapping_path is None
    assert not (tmp_path / "builtin.optical-mapping.json").exists()


def test_custom_mapping_is_written_and_loadable(tmp_path: Path) -> None:
    config = FormationConfig.from_json(
        json.dumps(
            {
                "seed": 1,
                "shape_zyx": [2, 2, 2],
                "voxel_size_xyz_m": [0.001, 0.001, 0.001],
                "palette": [
                    {"material_id": 0, "name": "air", "role": "background"},
                    {"material_id": 1, "name": "host-rock"},
                ],
                "layers": [{"kind": "host", "material_id": 1}],
                "mapping": {
                    "materials": [
                        {
                            "name": "host-rock",
                            "sigma_a_rgb_per_m": [1.0, 1.0, 1.0],
                            "sigma_s_rgb_per_m": [2.0, 2.0, 2.0],
                            "g": 0.0,
                            "ior": 1.5,
                        }
                    ]
                },
            }
        )
    )
    written = write_formation(config, out=tmp_path, name="custom")
    assert written.mapping_path is not None
    mapping = load_optical_mapping(written.mapping_path)
    assert mapping.digest == written.mapping_digest
    assert set(mapping.material_ids) == {0, 1}


def test_missing_custom_mapping_coefficients_fail() -> None:
    config = FormationConfig.from_json(
        json.dumps(
            {
                "shape_zyx": [1, 1, 1],
                "voxel_size_xyz_m": [1, 1, 1],
                "palette": [
                    {"material_id": 0, "name": "air", "role": "background"},
                    {"material_id": 1, "name": "host-rock"},
                ],
                "layers": [{"kind": "host", "material_id": 1}],
            }
        )
    )
    with pytest.raises(ConfigError, match="missing"):
        write_formation(config, out="/tmp/unused", name="missing")


def test_generate_formation_strict_returns_failure_on_constraint(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "formation.json"
    config_path.write_text(
        json.dumps(
            {
                "shape_zyx": [1, 1, 1],
                "voxel_size_xyz_m": [1, 1, 1],
                "palette": [
                    {"material_id": 0, "name": "air", "role": "background"},
                    {"material_id": 1, "name": "transparent-resin"},
                ],
                "layers": [{"kind": "host", "material_id": 1}],
                "constraints": [
                    {
                        "kind": "volume-fraction",
                        "material_id": 1,
                        "min": 0.0,
                        "max": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "generate-formation",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "out"),
                "--name",
                "strict",
                "--strict",
            ]
        )
        == 1
    )
