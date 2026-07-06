"""Integration test: image-stack fixture → import-voxels → optical convert."""

import subprocess
import sys
from pathlib import Path

import pytest

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import write_image_stack_fixture


def _vdbmat(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vdbmat.cli.main", *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
def test_image_stack_to_optical_end_to_end(tmp_path: Path) -> None:
    slices_dir, config = write_image_stack_fixture(tmp_path / "slices")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / "asset"
    assert main(
        [
            "convert-image-stack",
            str(slices_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--name",
            "stack",
        ]
    ) == 0
    manifest = out_dir / "stack.voxels.json"

    material_zarr = tmp_path / "material.zarr"
    imported = _vdbmat("import-voxels", str(manifest), str(material_zarr))
    assert imported.returncode == 0, imported.stderr

    optical_zarr = tmp_path / "optical.zarr"
    converted = _vdbmat("convert", str(material_zarr), str(optical_zarr))
    assert converted.returncode == 0, converted.stderr
    assert optical_zarr.exists()
