"""Integration tests: fixture assets import cleanly via ``vdbmat import-voxels``."""

import subprocess
import sys
from pathlib import Path

import pytest

from vdbmat_utils.fixtures import FIXTURE_PRESETS, build_fixture
from vdbmat_utils.io import write_asset


def _import_voxels(manifest: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vdbmat.cli.main",
            "import-voxels",
            str(manifest),
            str(output),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize("preset", FIXTURE_PRESETS)
def test_fixture_imports_end_to_end(preset: str, tmp_path: Path) -> None:
    manifest = write_asset(build_fixture(preset), tmp_path, preset)
    result = _import_voxels(manifest, tmp_path / f"{preset}.zarr")
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_tampered_manifest_is_rejected_end_to_end(tmp_path: Path) -> None:
    manifest = write_asset(build_fixture("anisotropic"), tmp_path, "anisotropic")
    payload = tmp_path / "anisotropic.material_id.npy"
    payload.write_bytes(payload.read_bytes() + b"\0")
    result = _import_voxels(manifest, tmp_path / "out.zarr")
    assert result.returncode != 0
