"""Browser-free tests for designlab_pipeline: no subprocess is ever invoked here.

Only the reuse/collision branches of ``run_generate_job`` return before the
``generate`` stage, so those are the only ``run_generate_job`` paths
exercised without a real subprocess; the full happy path is covered by the
``integration``-marked tests in ``tests/integration/test_designlab_pipeline.py``.
"""

import json
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "designlab"
sys.path.insert(0, str(EXAMPLES_DIR))

from designlab_pipeline import (  # noqa: E402
    PublishError,
    _write_run_config,
    check_roots,
    default_work_root,
    publish_name_for,
    run_generate_job,
    sweep_stale_work_dirs,
    validate_name,
)
from designlab_registry import PRIMITIVE_ARRAY_METHOD  # noqa: E402

from vdbmat_utils.core import config_digest  # noqa: E402
from vdbmat_utils.primitives import PrimitiveArrayConfig  # noqa: E402

_CONFIG = PrimitiveArrayConfig(
    voxel_size_xyz_m=(0.0001, 0.0001, 0.0001),
    primitive="cube",
    counts_xyz=(2, 2, 2),
    primitive_size_m=0.0004,
    gap_m=0.0002,
    margin_m=0.0001,
)


def test_validate_name_accepts_and_rejects() -> None:
    validate_name("demo-1")
    for bad in ("Demo", "1demo!", "-demo", "", "demo/other"):
        with pytest.raises(PublishError) as excinfo:
            validate_name(bad)
        assert excinfo.value.stage == "validate"


def test_publish_name_for_is_method_name_digest12() -> None:
    name = publish_name_for(PRIMITIVE_ARRAY_METHOD, "demo", _CONFIG)
    digest12 = config_digest(_CONFIG).removeprefix("sha256:")[:12]
    assert name == f"primitive-array-demo-{digest12}"
    assert len(digest12) == 12


def test_default_work_root_is_output_root_sibling(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    work_root = default_work_root(output_root)
    assert work_root == tmp_path / "out.designlab-work"


def test_check_roots_rejects_work_root_inside_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    work_root = output_root / "work"
    work_root.mkdir()
    with pytest.raises(PublishError) as excinfo:
        check_roots(output_root, work_root)
    assert excinfo.value.stage == "validate"


def test_check_roots_rejects_missing_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    with pytest.raises(PublishError):
        check_roots(output_root, tmp_path / "missing-work")
    with pytest.raises(PublishError):
        check_roots(tmp_path / "missing-out", tmp_path)


def test_write_run_config_golden(tmp_path: Path) -> None:
    run_config_path = _write_run_config(tmp_path, "demo", "demo.voxels.json")
    document = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert document == {
        "schema": {"name": "vdbmat.pipeline-config", "version": "2.0.0"},
        "input": {"kind": "direct-voxel", "path": "demo.voxels.json"},
        "mapping": {"name": "phase0-provisional-materials-v1"},
        "stages": {"validate_material": True, "validate_optical": True, "exports": []},
        "output": {"path": "bundle", "overwrite": False},
        "execution": {"random_seed": 0},
        "renderer": None,
    }


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "out"
    output_root.mkdir()
    work_root = tmp_path / "work"
    work_root.mkdir()
    return output_root, work_root


def test_run_generate_job_reuses_existing_valid_bundle(tmp_path: Path) -> None:
    output_root, work_root = _make_dirs(tmp_path)
    publish_name = publish_name_for(PRIMITIVE_ARRAY_METHOD, "demo", _CONFIG)
    dest = output_root / publish_name
    dest.mkdir()
    (dest / "run.json").write_text("{}", encoding="utf-8")
    (dest / "optical.zarr").mkdir()

    result = run_generate_job(
        method=PRIMITIVE_ARRAY_METHOD,
        config=_CONFIG,
        name="demo",
        output_root=output_root,
        work_root=work_root,
        seq=1,
    )
    assert result.reused is True
    assert result.publish_path == dest
    assert result.publish_name == publish_name


def test_run_generate_job_rejects_invalid_existing_target(tmp_path: Path) -> None:
    output_root, work_root = _make_dirs(tmp_path)
    publish_name = publish_name_for(PRIMITIVE_ARRAY_METHOD, "demo", _CONFIG)
    dest = output_root / publish_name
    dest.mkdir()  # no run.json / optical.zarr -> not a valid bundle

    with pytest.raises(PublishError) as excinfo:
        run_generate_job(
            method=PRIMITIVE_ARRAY_METHOD,
            config=_CONFIG,
            name="demo",
            output_root=output_root,
            work_root=work_root,
            seq=1,
        )
    assert excinfo.value.stage == "validate"
    # No work directory should have been created for a rejected collision.
    assert list(work_root.iterdir()) == []


def test_run_generate_job_rejects_invalid_name(tmp_path: Path) -> None:
    output_root, work_root = _make_dirs(tmp_path)
    with pytest.raises(PublishError) as excinfo:
        run_generate_job(
            method=PRIMITIVE_ARRAY_METHOD,
            config=_CONFIG,
            name="Not Valid",
            output_root=output_root,
            work_root=work_root,
            seq=1,
        )
    assert excinfo.value.stage == "validate"


def test_sweep_stale_work_dirs_removes_all_entries(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    (work_root / "001-demo").mkdir()
    (work_root / "001-demo" / "leftover.txt").write_text("x", encoding="utf-8")
    (work_root / "loose-file.txt").write_text("x", encoding="utf-8")

    sweep_stale_work_dirs(work_root)

    assert list(work_root.iterdir()) == []


def test_sweep_stale_work_dirs_tolerates_missing_root(tmp_path: Path) -> None:
    sweep_stale_work_dirs(tmp_path / "does-not-exist")
