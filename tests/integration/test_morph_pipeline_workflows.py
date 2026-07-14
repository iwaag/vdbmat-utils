"""Integration tests: Phase 2 fixtures → ``vdbmat import-voxels`` → ``convert``."""

import subprocess
import sys
from pathlib import Path

import pytest

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import write_morph_fixture, write_pipeline_fixture


def _vdbmat(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vdbmat.cli.main", *args],
        capture_output=True,
        text=True,
    )


def _import_and_convert(manifest: Path, tmp_path: Path, name: str) -> None:
    material_zarr = tmp_path / f"{name}.zarr"
    imported = _vdbmat("import-voxels", str(manifest), str(material_zarr))
    assert imported.returncode == 0, imported.stderr
    optical_zarr = tmp_path / f"{name}-optical.zarr"
    converted = _vdbmat("convert", str(material_zarr), str(optical_zarr))
    assert converted.returncode == 0, converted.stderr
    assert optical_zarr.exists()


@pytest.mark.integration
def test_morphed_fixture_to_optical_end_to_end(tmp_path: Path) -> None:
    slices_dir, config = write_morph_fixture(tmp_path / "slices")
    config_path = tmp_path / "morph.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / "asset"
    assert main(
        [
            "morph-stack",
            str(slices_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "morphed",
        ]
    ) == 0
    _import_and_convert(out_dir / "morphed.voxels.json", tmp_path, "morphed")


@pytest.mark.integration
def test_composed_fixture_to_optical_end_to_end(tmp_path: Path) -> None:
    config_path = write_pipeline_fixture(tmp_path / "assets")
    out_dir = tmp_path / "asset"
    assert main(
        [
            "apply-pipeline",
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "composed",
        ]
    ) == 0
    _import_and_convert(out_dir / "composed.voxels.json", tmp_path, "composed")
