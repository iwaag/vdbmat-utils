"""Unit tests for the pipeline engine and apply-pipeline CLI (plan D6)."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from vdbmat_utils.cli.main import main
from vdbmat_utils.core import build_material_label_volume, build_provenance
from vdbmat_utils.io import write_asset
from vdbmat_utils.ops import compose, crop, remap_materials
from vdbmat_utils.pipeline import (
    PipelineConfig,
    PipelineError,
    run_pipeline,
    validate_pipeline,
)

_PALETTE = (
    (0, "air", "background"),
    (1, "resin_a", "material"),
    (2, "resin_b", "material"),
    (3, "resin_c", "material"),
)


def _volume(array: object) -> MaterialLabelVolume:
    return build_material_label_volume(
        material_id=np.asarray(array, dtype=np.uint16),
        voxel_size_xyz_m=(0.001, 0.002, 0.003),
        palette=tuple(
            MaterialDefinition(material_id=i, name=name, role=MaterialRole(role))
            for i, name, role in _PALETTE
        ),
        provenance=build_provenance(
            generator="vdbmat-utils.tests", generator_version="0.0.0"
        ),
    )


def _write_input(directory: Path, name: str, array: object) -> Path:
    return write_asset(_volume(array), directory, name)


def _config(
    inputs: tuple[Mapping[str, object], ...],
    steps: tuple[Mapping[str, object], ...],
    output_ref: str,
) -> PipelineConfig:
    return PipelineConfig(inputs=inputs, steps=steps, output={"ref": output_ref})


_CROP_STEP: Mapping[str, object] = {
    "op": "crop",
    "from": "a",
    "min_zyx": [0, 0, 0],
    "max_zyx": [1, 1, 1],
    "as": "b",
}


def _one_input(tmp_path: Path) -> tuple[Mapping[str, object], ...]:
    _write_input(tmp_path, "a", [[[1, 2], [0, 3]]])
    return ({"id": "a", "manifest_path": "a.voxels.json"},)


def test_unknown_op_names_step_index(tmp_path: Path) -> None:
    config = _config(
        _one_input(tmp_path), ({"op": "sharpen", "from": "a", "as": "b"},), "b"
    )
    with pytest.raises(PipelineError, match=r"steps\[0\]\.op: unknown op 'sharpen'"):
        validate_pipeline(config)


def test_unknown_parameter_names_step_index(tmp_path: Path) -> None:
    step = dict(_CROP_STEP)
    step["sigma"] = 2
    config = _config(_one_input(tmp_path), (step,), "b")
    with pytest.raises(
        PipelineError, match=r"steps\[0\]: unknown parameters for op 'crop': sigma"
    ):
        validate_pipeline(config)


def test_unbound_reference_names_step_index(tmp_path: Path) -> None:
    step = dict(_CROP_STEP)
    step["from"] = "ghost"
    config = _config(_one_input(tmp_path), (step,), "b")
    with pytest.raises(PipelineError, match=r"steps\[0\]\.from: id 'ghost'"):
        validate_pipeline(config)


def test_rebinding_an_id_is_an_error(tmp_path: Path) -> None:
    step = dict(_CROP_STEP)
    step["as"] = "a"
    config = _config(_one_input(tmp_path), (step,), "a")
    with pytest.raises(PipelineError, match=r"steps\[0\]\.as: id 'a' is already"):
        validate_pipeline(config)


def test_unused_input_is_an_error(tmp_path: Path) -> None:
    _write_input(tmp_path, "a", [[[1]]])
    _write_input(tmp_path, "extra", [[[1]]])
    config = _config(
        (
            {"id": "a", "manifest_path": "a.voxels.json"},
            {"id": "extra", "manifest_path": "extra.voxels.json"},
        ),
        (_CROP_STEP,),
        "b",
    )
    with pytest.raises(PipelineError, match="input id\\(s\\) extra are never used"):
        validate_pipeline(config)


def test_missing_input_file_is_an_error(tmp_path: Path) -> None:
    config = _config(
        ({"id": "a", "manifest_path": "nope.voxels.json"},), (_CROP_STEP,), "b"
    )
    with pytest.raises(PipelineError, match="manifest for id 'a' not found"):
        run_pipeline(config, base_dir=tmp_path)


def test_runtime_op_error_names_step_index(tmp_path: Path) -> None:
    step = dict(_CROP_STEP)
    step["max_zyx"] = [9, 9, 9]  # out of range for the 1x2x2 input
    config = _config(_one_input(tmp_path), (step,), "b")
    with pytest.raises(PipelineError, match=r"step 0 \(crop\):"):
        run_pipeline(config, base_dir=tmp_path)


def test_three_step_pipeline_matches_direct_python_api(tmp_path: Path) -> None:
    base_array = np.zeros((2, 3, 3), dtype=np.uint16)
    base_array[:, 0:2, 0:2] = 1
    overlay_array = np.zeros((1, 2, 2), dtype=np.uint16)
    overlay_array[0, 1, 1] = 3
    _write_input(tmp_path, "base", base_array)
    _write_input(tmp_path, "overlay", overlay_array)
    config = _config(
        (
            {"id": "base", "manifest_path": "base.voxels.json"},
            {"id": "overlay", "manifest_path": "overlay.voxels.json"},
        ),
        (
            {
                "op": "crop",
                "from": "base",
                "min_zyx": [0, 0, 0],
                "max_zyx": [1, 2, 2],
                "as": "cropped",
            },
            {
                "op": "remap-materials",
                "from": "cropped",
                "mapping": {"1": 5},
                "as": "remapped",
            },
            {
                "op": "compose",
                "base": "remapped",
                "overlay": "overlay",
                "mode": "union",
                "as": "final",
            },
        ),
        "final",
    )
    result = run_pipeline(config, base_dir=tmp_path)

    volume = _volume(base_array)
    direct = crop(volume, min_zyx=(0, 0, 0), max_zyx=(1, 2, 2))
    direct = remap_materials(direct, {1: 5})
    direct = compose(direct, _volume(overlay_array), mode="union")
    np.testing.assert_array_equal(result.material_id, direct.material_id)
    assert result.geometry == direct.geometry

    assert result.provenance.generator == "vdbmat-utils.pipeline"
    assert len(result.provenance.sources) == 2
    second = run_pipeline(config, base_dir=tmp_path)
    assert second.provenance == result.provenance
    np.testing.assert_array_equal(second.material_id, result.material_id)


def _write_pipeline_json(tmp_path: Path) -> Path:
    _write_input(tmp_path, "a", [[[1, 2], [0, 3]]])
    payload = {
        "inputs": [{"id": "a", "manifest_path": "a.voxels.json"}],
        "steps": [dict(_CROP_STEP)],
        "output": {"ref": "b"},
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_cli_dry_run_touches_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_pipeline_json(tmp_path)
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "apply-pipeline",
                "--config",
                str(config_path),
                "--out",
                str(out_dir),
                "--name",
                "result",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "step 0: crop(from=a) -> b" in output
    assert "output: b" in output
    assert not out_dir.exists()


def test_cli_apply_pipeline_writes_valid_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_pipeline_json(tmp_path)
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "apply-pipeline",
                "--config",
                str(config_path),
                "--out",
                str(out_dir),
                "--name",
                "result",
            ]
        )
        == 0
    )
    assert "wrote" in capsys.readouterr().out
    manifest = out_dir / "result.voxels.json"
    assert main(["validate", str(manifest)]) == 0


def test_cli_invalid_pipeline_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_pipeline_json(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["steps"][0]["op"] = "sharpen"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        main(
            [
                "apply-pipeline",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "out"),
                "--name",
                "result",
                "--dry-run",
            ]
        )
        == 1
    )
    assert "unknown op 'sharpen'" in capsys.readouterr().err
