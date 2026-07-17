"""Integration tests: the designlab generate pipeline through the pinned CLIs.

Every case here drives ``run_generate_job`` for real (subprocess calls to
both ``vdbmat-utils`` and ``vdbmat``), so inputs are kept tiny (a single
4x4x4-voxel primitive) to stay fast.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "designlab"
sys.path.insert(0, str(EXAMPLES_DIR))

from designlab_pipeline import PublishError, run_generate_job  # noqa: E402
from designlab_registry import PRIMITIVE_ARRAY_METHOD  # noqa: E402

from vdbmat_utils.primitives import PrimitiveArrayConfig  # noqa: E402

_TINY_CONFIG = PrimitiveArrayConfig(
    voxel_size_xyz_m=(0.001, 0.001, 0.001),
    primitive="cube",
    counts_xyz=(1, 1, 1),
    primitive_size_m=0.002,
    gap_m=0.0,
    margin_m=0.001,
)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "out"
    output_root.mkdir()
    work_root = tmp_path / "work"
    work_root.mkdir()
    return output_root, work_root


@pytest.mark.integration
def test_happy_path_publishes_bundle_viewer_can_detect(tmp_path: Path) -> None:
    output_root, work_root = _roots(tmp_path)

    result = run_generate_job(
        method=PRIMITIVE_ARRAY_METHOD,
        config=_TINY_CONFIG,
        name="demo",
        output_root=output_root,
        work_root=work_root,
        seq=1,
    )

    assert result.reused is False
    assert result.publish_path == output_root / result.publish_name
    assert (result.publish_path / "run.json").is_file()
    assert (result.publish_path / "optical.zarr").exists()

    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "vdbmat.cli.main",
            "validate",
            str(result.publish_path),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stderr
    document = json.loads(validate.stdout)
    assert document["status"] == "ok"

    # The transaction's own work directory is cleaned up on success.
    assert list(work_root.iterdir()) == []


@pytest.mark.integration
def test_gui_saved_config_reproduces_bundle_payload_digest(tmp_path: Path) -> None:
    output_root, work_root = _roots(tmp_path)
    result = run_generate_job(
        method=PRIMITIVE_ARRAY_METHOD,
        config=_TINY_CONFIG,
        name="repro",
        output_root=output_root,
        work_root=work_root,
        seq=1,
    )

    run_manifest_path = result.publish_path / "run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    bundle_payload_sha256 = run_manifest["input_payload_sha256"]
    assert isinstance(bundle_payload_sha256, str)
    assert bundle_payload_sha256.startswith("sha256:")

    config_path = tmp_path / "repro.primarray.json"
    config_path.write_text(_TINY_CONFIG.to_json(), encoding="utf-8")
    cli_out = tmp_path / "cli-out"
    cli_out.mkdir()
    cli_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vdbmat_utils.cli.main",
            "generate-primitive-array",
            "--config",
            str(config_path),
            "--out",
            str(cli_out),
            "--name",
            "repro",
        ],
        capture_output=True,
        text=True,
    )
    assert cli_result.returncode == 0, cli_result.stderr
    payload = (cli_out / "repro.material_id.npy").read_bytes()
    cli_payload_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"

    assert cli_payload_sha256 == bundle_payload_sha256


@pytest.mark.integration
def test_size_guard_failure_publishes_nothing(tmp_path: Path) -> None:
    output_root, work_root = _roots(tmp_path)
    oversized = PrimitiveArrayConfig(
        voxel_size_xyz_m=(0.001, 0.001, 0.001),
        primitive="cube",
        counts_xyz=(1, 1, 1),
        primitive_size_m=0.002,
        gap_m=0.0,
        margin_m=0.001,
        max_axis_cells=1,
    )

    with pytest.raises(PublishError) as excinfo:
        run_generate_job(
            method=PRIMITIVE_ARRAY_METHOD,
            config=oversized,
            name="toobig",
            output_root=output_root,
            work_root=work_root,
            seq=1,
        )
    assert excinfo.value.stage == "generate"
    assert list(output_root.iterdir()) == []
    # The failed job's work directory is left behind for inspection.
    assert list(work_root.iterdir()) != []


@pytest.mark.integration
def test_second_call_with_same_config_reuses_published_bundle(tmp_path: Path) -> None:
    output_root, work_root = _roots(tmp_path)
    first = run_generate_job(
        method=PRIMITIVE_ARRAY_METHOD,
        config=_TINY_CONFIG,
        name="again",
        output_root=output_root,
        work_root=work_root,
        seq=1,
    )
    assert first.reused is False

    second = run_generate_job(
        method=PRIMITIVE_ARRAY_METHOD,
        config=_TINY_CONFIG,
        name="again",
        output_root=output_root,
        work_root=work_root,
        seq=2,
    )
    assert second.reused is True
    assert second.publish_path == first.publish_path
    assert second.publish_name == first.publish_name
