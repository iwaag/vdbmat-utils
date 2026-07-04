"""Golden-fixture contract tests against the pinned vdbmat version.

The hardcoded digests pin the exact bytes each preset produces; a change to
either digest means the output contract moved and must be reviewed (and the
goldens deliberately updated) rather than silently absorbed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from vdbmat.core import VolumeValidationError
from vdbmat.io import (
    VoxelManifestError,
    inspect_material_label_manifest,
    read_material_label_manifest,
)

from vdbmat_utils.fixtures import FIXTURE_PRESETS, build_fixture
from vdbmat_utils.io import write_asset

GOLDEN_PAYLOAD_SHA256 = {
    "anisotropic": "1e72ce29769c013f5248f85d30576a2eec24ca03fcbd750a411983739d536b52",
    "transformed": "670ec0325af58e088580d170ca84f16f00bae2fb5e8ce41e95550a55ba713c72",
    "multimaterial": "1daa4c47e8f2d38706d934e2be5f7a4e3400a46449ab97cfed4824e43930ff18",
}

GOLDEN_MANIFEST_SHA256 = {
    "anisotropic": "67c2a14d59929b5e5e6db46bf3dce561ed82f9dfba43a62973b7bfc912e5ec7f",
    "transformed": "7620cd5608df350ebf986d758083408badba8be0bb74014fbd28efd2fc6b0bca",
    "multimaterial": "c469901e5f2e094e697598c2495511f6c50e74865cacaecb479cfc22577ea049",
}


def _write(preset: str, directory: Path) -> Path:
    return write_asset(build_fixture(preset), directory, preset)


@pytest.mark.parametrize("preset", FIXTURE_PRESETS)
def test_golden_digests(preset: str, tmp_path: Path) -> None:
    manifest_path = _write(preset, tmp_path)
    payload_path = tmp_path / f"{preset}.material_id.npy"
    payload_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert payload_sha == GOLDEN_PAYLOAD_SHA256[preset]
    assert manifest_sha == GOLDEN_MANIFEST_SHA256[preset]


@pytest.mark.parametrize("preset", FIXTURE_PRESETS)
def test_round_trip_through_pinned_vdbmat(preset: str, tmp_path: Path) -> None:
    manifest_path = _write(preset, tmp_path)
    volume = build_fixture(preset)
    loaded = read_material_label_manifest(manifest_path)
    assert np.array_equal(loaded.material_id, volume.material_id)
    assert loaded.geometry == volume.geometry
    inspection = inspect_material_label_manifest(manifest_path)
    assert inspection is not None


def test_transformed_preset_has_rotation_and_offset(tmp_path: Path) -> None:
    volume = build_fixture("transformed")
    matrix = volume.geometry.local_to_world
    assert matrix[0][3] != 0.0
    assert matrix[0][1] == -1.0


def test_multimaterial_includes_non_builtin_name() -> None:
    names = {definition.name for definition in build_fixture("multimaterial").palette}
    assert "quartz_vein" in names


class TestInvalidMetadata:
    """Tampered manifests must be rejected by the pinned vdbmat reader."""

    def _manifest(self, tmp_path: Path) -> tuple[Path, dict[str, object]]:
        manifest_path = _write("anisotropic", tmp_path)
        return manifest_path, json.loads(manifest_path.read_text())

    def _expect_rejection(
        self, manifest_path: Path, tampered: dict[str, object]
    ) -> None:
        manifest_path.write_text(json.dumps(tampered))
        with pytest.raises((VoxelManifestError, VolumeValidationError)):
            read_material_label_manifest(manifest_path)

    def test_payload_checksum_mismatch(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        payload = manifest["payload"]
        assert isinstance(payload, dict)
        payload["sha256"] = "0" * 64
        self._expect_rejection(manifest_path, manifest)

    def test_wrong_dtype(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        payload = manifest["payload"]
        assert isinstance(payload, dict)
        payload["dtype"] = "int32"
        self._expect_rejection(manifest_path, manifest)

    def test_wrong_axis_order(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        payload = manifest["payload"]
        assert isinstance(payload, dict)
        payload["dimensions"] = ["x", "y", "z"]
        self._expect_rejection(manifest_path, manifest)

    def test_undeclared_material_reference(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        materials = manifest["materials"]
        assert isinstance(materials, list)
        del materials[1]
        self._expect_rejection(manifest_path, manifest)

    def test_non_rigid_transform(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        manifest["local_to_world"] = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self._expect_rejection(manifest_path, manifest)

    def test_missing_voxel_size(self, tmp_path: Path) -> None:
        manifest_path, manifest = self._manifest(tmp_path)
        del manifest["voxel_size_xyz_m"]
        self._expect_rejection(manifest_path, manifest)
