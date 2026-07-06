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


def test_fixture_seed_changes_nothing_for_seedless_presets(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    assert main(["generate-fixture", "anisotropic", "-o", str(a)]) == 0
    assert main(["generate-fixture", "anisotropic", "-o", str(b), "--seed", "0"]) == 0
    assert (
        (a / "anisotropic.material_id.npy").read_bytes()
        == (b / "anisotropic.material_id.npy").read_bytes()
    )
