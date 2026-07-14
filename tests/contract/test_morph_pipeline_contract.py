"""Contract tests for the Phase 2 workflows (plan Step 4.2).

Determinism (byte-equal double runs through the CLI), payload/manifest
digest goldens, axis-orientation ASCII goldens for the morphed fixture, and
the material-conservation identities that genuinely hold (remap voxel-count
transfer; mask/compose count identities). Conservation is deliberately *not*
claimed for resample (see ``docs/volume-ops.md``).
"""

import hashlib
from pathlib import Path

import numpy as np
from vdbmat.io import read_material_label_manifest

from vdbmat_utils.cli.main import main
from vdbmat_utils.fixtures import (
    build_fixture,
    write_morph_fixture,
    write_pipeline_fixture,
)
from vdbmat_utils.ops import apply_mask, compose, remap_materials
from vdbmat_utils.preview import material_counts, slice_ascii

MORPH_GOLDEN_PAYLOAD_SHA256 = (
    "04d4c5855951d800b48d5c8de4c63295dcfa67f40ebeebee87c3c0935521d936"
)
MORPH_GOLDEN_MANIFEST_SHA256 = (
    "55b03f2d5cab5c2383da49ce8fcf7777add14169056b14130508427f341da06d"
)
PIPELINE_GOLDEN_PAYLOAD_SHA256 = (
    "bf2cdf84becaa4f5b4005a4abd331a7107837d02e00ff72485fe0a5723adf10e"
)
PIPELINE_GOLDEN_MANIFEST_SHA256 = (
    "82b9d381b8c8ea20e89c322b62006220549278205c37350c0f75cf64ae60cc07"
)

# Orientation goldens: one slice per axis of the morphed fixture. The legends
# pin the row/column axis mapping; the bodies pin the interpolation result
# (including the merge already completed by z=5).
MORPH_ASCII_Z5 = """\
slice z=5  +x →  +y ↓
............
.1111111111.
.1111111111.
.1111111111.
.1111111111.
............
...22222....
...22222....
...22222....
............"""

MORPH_ASCII_Y3 = """\
slice y=3  +x →  +z ↓
.1111..1111.
.1111..1111.
.11111.1111.
.11111.1111.
.1111111111.
.1111111111.
.1111111111.
.1111111111."""

MORPH_ASCII_X3 = """\
slice x=3  +y →  +z ↓
.1111.222.
.1111.222.
.1111.222.
.1111.222.
.1111.222.
.1111.222.
.1111.....
.1111....."""


def _run_morph_cli(tmp_path: Path, out_name: str) -> Path:
    slices_dir, config = write_morph_fixture(tmp_path / "slices")
    config_path = tmp_path / "morph.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    out_dir = tmp_path / out_name
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
    return out_dir


def _run_pipeline_cli(tmp_path: Path, out_name: str) -> Path:
    config_path = write_pipeline_fixture(tmp_path / "assets")
    out_dir = tmp_path / out_name
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
    return out_dir


def test_morph_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_morph_cli(tmp_path, "a")
    second = _run_morph_cli(tmp_path, "b")
    for filename in ("morphed.voxels.json", "morphed.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_pipeline_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _run_pipeline_cli(tmp_path, "a")
    second = _run_pipeline_cli(tmp_path, "b")
    for filename in ("composed.voxels.json", "composed.material_id.npy"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_morph_golden_digests(tmp_path: Path) -> None:
    out_dir = _run_morph_cli(tmp_path, "out")
    payload_sha = hashlib.sha256(
        (out_dir / "morphed.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "morphed.voxels.json").read_bytes()
    ).hexdigest()
    assert payload_sha == MORPH_GOLDEN_PAYLOAD_SHA256
    assert manifest_sha == MORPH_GOLDEN_MANIFEST_SHA256


def test_pipeline_golden_digests(tmp_path: Path) -> None:
    out_dir = _run_pipeline_cli(tmp_path, "out")
    payload_sha = hashlib.sha256(
        (out_dir / "composed.material_id.npy").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256(
        (out_dir / "composed.voxels.json").read_bytes()
    ).hexdigest()
    assert payload_sha == PIPELINE_GOLDEN_PAYLOAD_SHA256
    assert manifest_sha == PIPELINE_GOLDEN_MANIFEST_SHA256


def test_morph_orientation_ascii_goldens(tmp_path: Path) -> None:
    out_dir = _run_morph_cli(tmp_path, "out")
    volume = read_material_label_manifest(out_dir / "morphed.voxels.json")
    assert volume.geometry.shape_zyx == (8, 10, 12)
    assert slice_ascii(volume, "z", 5) == MORPH_ASCII_Z5
    assert slice_ascii(volume, "y", 3) == MORPH_ASCII_Y3
    assert slice_ascii(volume, "x", 3) == MORPH_ASCII_X3


def test_both_outputs_validate(tmp_path: Path) -> None:
    morph_dir = _run_morph_cli(tmp_path, "m")
    pipeline_dir = _run_pipeline_cli(tmp_path, "p")
    assert main(["validate", str(morph_dir / "morphed.voxels.json")]) == 0
    assert main(["validate", str(pipeline_dir / "composed.voxels.json")]) == 0


def test_remap_conserves_and_transfers_counts() -> None:
    volume = build_fixture("multimaterial")
    before = material_counts(volume)
    remapped = remap_materials(volume, {2: 7})
    after = material_counts(remapped)
    assert sum(after.values()) == sum(before.values())  # total conserved
    assert after[7] == before[2]  # counts move as mapped
    assert 2 not in after  # pruned


def test_compose_count_identities(tmp_path: Path) -> None:
    config_path = write_pipeline_fixture(tmp_path / "assets")
    base = read_material_label_manifest(config_path.parent / "base.voxels.json")
    overlay = read_material_label_manifest(
        config_path.parent / "overlay.voxels.json"
    )
    base_fg = base.material_id != 0
    overlay_fg = overlay.material_id != 0

    union = compose(base, overlay, mode="union")
    assert int((union.material_id != 0).sum()) == int((base_fg | overlay_fg).sum())
    intersect = compose(base, overlay, mode="intersect")
    assert int((intersect.material_id != 0).sum()) == int(
        (base_fg & overlay_fg).sum()
    )
    subtract = compose(base, overlay, mode="subtract")
    assert int((subtract.material_id != 0).sum()) == int(
        (base_fg & ~overlay_fg).sum()
    )
    # keep/clear masks partition the base foreground exactly.
    kept = apply_mask(base, overlay, mode="keep")
    cleared = apply_mask(base, overlay, mode="clear")
    kept_fg = int((kept.material_id != 0).sum())
    cleared_fg = int((cleared.material_id != 0).sum())
    assert kept_fg + cleared_fg == int(base_fg.sum())
    np.testing.assert_array_equal(
        kept.material_id, intersect.material_id
    )  # keep-mask ≡ intersect for label masks