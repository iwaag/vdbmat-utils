"""CLI tests for inspect, validate, and generate-fixture."""

import json
from pathlib import Path

import pytest

from vdbmat_utils.cli.main import main


def _generate(tmp_path: Path, preset: str = "anisotropic") -> Path:
    assert main(["generate-fixture", preset, "-o", str(tmp_path)]) == 0
    return tmp_path / f"{preset}.voxels.json"


def test_generate_and_validate(tmp_path: Path) -> None:
    manifest = _generate(tmp_path)
    assert manifest.exists()
    assert (tmp_path / "anisotropic.material_id.npy").exists()
    assert main(["validate", str(manifest)]) == 0


def test_inspect_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    capsys.readouterr()
    assert main(["inspect", str(manifest), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape_zyx"] == [3, 4, 5]
    assert payload["material_ids"] == [0, 1]


def test_inspect_human_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    capsys.readouterr()
    assert main(["inspect", str(manifest)]) == 0
    assert "shape_zyx" in capsys.readouterr().out


def test_validate_corrupt_asset_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    payload = tmp_path / "anisotropic.material_id.npy"
    payload.write_bytes(payload.read_bytes() + b"\0")
    assert main(["validate", str(manifest)]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_manifest_returns_1(tmp_path: Path) -> None:
    assert main(["validate", str(tmp_path / "missing.voxels.json")]) == 1


def test_usage_error_returns_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["generate-fixture", "no-such-preset", "-o", "/tmp/x"])
    assert excinfo.value.code == 2


def test_material_counts_human_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    capsys.readouterr()
    assert main(["material-counts", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "0 (void, background): 30" in out
    assert "1 (resin_clear, material): 30" in out


def test_material_counts_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    capsys.readouterr()
    assert main(["material-counts", str(manifest), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"0": 30, "1": 30}


def test_preview_slices_ascii_default_middle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    capsys.readouterr()
    assert main(["preview-slices", str(manifest)]) == 0
    out = capsys.readouterr().out
    # anisotropic preset has shape_zyx (3, 4, 5) -> middle z slice is index 1
    assert out.startswith("slice z=1  +x →  +y ↓\n")
    assert len(out.strip().splitlines()) == 1 + 4


def test_preview_slices_pgm_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    target = tmp_path / "slice.pgm"
    capsys.readouterr()
    assert main(
        ["preview-slices", str(manifest), "--axis", "y", "--index", "0",
         "--out", str(target)]
    ) == 0
    assert target.read_bytes().startswith(b"P5\n5 3\n255\n")


def test_preview_slices_bad_index_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _generate(tmp_path)
    assert main(["preview-slices", str(manifest), "--index", "99"]) == 1
    assert "out of range" in capsys.readouterr().err


def test_convert_image_stack_with_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdbmat_utils.fixtures import write_image_stack_fixture

    slices_dir, config = write_image_stack_fixture(tmp_path / "slices")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(
        [
            "convert-image-stack", str(slices_dir),
            "--config", str(config_path),
            "--out", str(out_dir),
            "--name", "stack",
            "--voxel-size", "0.001", "0.002", "0.004",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "stack.voxels.json" in output
    assert "(air, background):" in output
    capsys.readouterr()
    assert main(["inspect", str(out_dir / "stack.voxels.json"), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["voxel_size_xyz_m"] == [0.001, 0.002, 0.004]


def test_convert_image_stack_bad_stack_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdbmat_utils.fixtures import write_image_stack_fixture

    slices_dir, config = write_image_stack_fixture(tmp_path / "slices")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    (slices_dir / "slice_0001.pgm").unlink()
    assert main(
        [
            "convert-image-stack", str(slices_dir),
            "--config", str(config_path),
            "--out", str(tmp_path / "out"),
            "--name", "stack",
        ]
    ) == 1
    assert "missing index" in capsys.readouterr().err


def test_voxelize_mesh_with_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdbmat_utils.fixtures import write_mesh_fixture

    mesh_path, config = write_mesh_fixture(tmp_path / "mesh")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(
        [
            "voxelize-mesh", str(mesh_path),
            "--config", str(config_path),
            "--out", str(out_dir),
            "--name", "bracket",
            "--voxel-size", "0.001", "0.001", "0.001",
            "--material-id", "2",
            "--material-name", "white-resin",
            "--padding", "0",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "bracket.voxels.json" in output
    # 1 mm voxels, no padding: 3x2x1 mm bracket -> 3x2x1 grid.
    assert "shape_zyx: (1, 2, 3)" in output
    assert "2 (white-resin, material):" in output
    capsys.readouterr()
    assert main(["inspect", str(out_dir / "bracket.voxels.json"), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["voxel_size_xyz_m"] == [0.001, 0.001, 0.001]
    assert inspected["material_ids"] == [0, 2]
    assert inspected["source_identity"].startswith("sha256:")


def test_voxelize_mesh_bad_source_unit_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdbmat_utils.fixtures import write_mesh_fixture

    mesh_path, config = write_mesh_fixture(tmp_path / "mesh")
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(
        config.to_json().replace('"mm"', '"furlong"'), encoding="utf-8"
    )
    assert main(
        [
            "voxelize-mesh", str(mesh_path),
            "--config", str(config_path),
            "--out", str(tmp_path / "out"),
            "--name", "bracket",
        ]
    ) == 1
    assert "source_unit" in capsys.readouterr().err


def _primitive_array_config_json() -> str:
    from vdbmat_utils.primitives import PrimitiveArrayConfig

    config = PrimitiveArrayConfig(
        voxel_size_xyz_m=(1e-4, 1e-4, 1e-4),
        primitive="cube",
        counts_xyz=(3, 2, 1),
        primitive_size_m=4e-4,
        gap_m=2e-4,
        margin_m=1e-4,
    )
    return config.to_json()


def test_generate_primitive_array(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(_primitive_array_config_json(), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(
        [
            "generate-primitive-array",
            "--config", str(config_path),
            "--out", str(out_dir),
            "--name", "demo",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "demo.voxels.json" in output
    assert "(transparent-resin, material):" in output
    assert "(black-opaque-resin, material):" in output
    capsys.readouterr()
    assert main(["validate", str(out_dir / "demo.voxels.json")]) == 0


def test_generate_primitive_array_bad_config_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        _primitive_array_config_json().replace('"cube"', '"cylinder"'),
        encoding="utf-8",
    )
    assert main(
        [
            "generate-primitive-array",
            "--config", str(config_path),
            "--out", str(tmp_path / "out"),
            "--name", "demo",
        ]
    ) == 1
    assert "primitive" in capsys.readouterr().err


def test_fixture_seed_changes_nothing_for_seedless_presets(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    assert main(["generate-fixture", "anisotropic", "-o", str(a)]) == 0
    assert main(["generate-fixture", "anisotropic", "-o", str(b), "--seed", "0"]) == 0
    assert (
        (a / "anisotropic.material_id.npy").read_bytes()
        == (b / "anisotropic.material_id.npy").read_bytes()
    )
