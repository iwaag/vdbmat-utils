"""Integration tests: Phase 3 reference formations through vdbmat."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vdbmat_utils.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIGS = (
    ("marble-like", ROOT / "examples/formation_generation/marble-like.formation.json"),
    ("granite-like", ROOT / "examples/formation_generation/granite-like.formation.json"),
)


def _vdbmat(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vdbmat.cli.main", *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize(("name", "config"), REFERENCE_CONFIGS)
def test_reference_formation_to_optical_end_to_end(
    tmp_path: Path, name: str, config: Path
) -> None:
    out = tmp_path / name
    assert main(
        [
            "generate-formation",
            "--config",
            str(config),
            "--out",
            str(out),
            "--name",
            name,
            "--strict",
        ]
    ) == 0
    manifest = out / f"{name}.voxels.json"
    mapping = out / f"{name}.optical-mapping.json"
    assert main(["validate", str(manifest)]) == 0

    material_zarr = tmp_path / f"{name}.zarr"
    imported = _vdbmat("import-voxels", str(manifest), str(material_zarr))
    assert imported.returncode == 0, imported.stderr

    digest = _vdbmat("mapping-digest", str(mapping), "--json")
    assert digest.returncode == 0, digest.stderr
    digest_payload = json.loads(digest.stdout)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert digest_payload["digest"] in manifest_payload["source"]["notes"]

    optical_zarr = tmp_path / f"{name}-optical.zarr"
    converted = _vdbmat(
        "convert",
        str(material_zarr),
        str(optical_zarr),
        "--mapping-file",
        str(mapping),
        "--json",
    )
    assert converted.returncode == 0, converted.stderr
    assert json.loads(converted.stdout)["mapping_digest"] == digest_payload["digest"]
    assert optical_zarr.exists()
