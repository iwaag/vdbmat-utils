"""The one-shot generate pipeline: validate -> generate -> map -> verify -> publish.

Every stage after ``validate`` runs a pinned CLI as a subprocess
(``sys.executable -m vdbmat_utils.cli.main ...`` / ``sys.executable -m
vdbmat.cli.main ...``) — this module never imports ``vdbmat.pipeline`` or
any other vdbmat-utils-internal generation code, per the roadmap's package
boundary ("生成と optical mapping の実行は必ずサブプロセス CLI 経由とする").
It builds the ``vdbmat.pipeline-config`` JSON by hand as a plain dict for
the same reason.

The whole job runs inside a work directory ``W`` and is published to
``--output-root`` only after ``verify`` succeeds, via a single
``os.replace`` (atomic on the same filesystem). Any failure before
``publish`` leaves ``--output-root`` untouched; ``W`` is left in place for
inspection on failure, and removed on success.

Deviation from the roadmap sketch (recorded per plan Step "roadmap 記載の
`import` 段との差分"): the roadmap's five-stage sketch is
``validate -> generate -> import -> map -> publish``, but ADR-009 D1 means
canonical bundles are built by ``vdbmat run`` consuming the direct-voxel
manifest directly (it persists ``material.zarr`` itself in its own
``persist-material`` internal stage); a standalone ``import-voxels`` call
produces a zarr store no bundle ever references. So this module's stage
vocabulary is ``validate -> generate -> map -> verify -> publish``, folding
the sketch's ``import`` into ``map``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from designlab_registry import GeneratorMethod

from vdbmat_utils.core import GeneratorConfig, config_digest

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RUN_MANIFEST_NAME = "run.json"
_OPTICAL_ZARR_NAME = "optical.zarr"
_BUNDLE_DIRNAME = "bundle"
_BUILTIN_MAPPING_NAME = "phase0-provisional-materials-v1"

STAGES = ("validate", "generate", "map", "verify", "publish")


class PublishError(Exception):
    """The generate job failed at a named stage; nothing was published.

    ``message`` is either a validation message this module raised itself,
    or a subprocess's own stderr tail, passed through unreworded (roadmap:
    "サブプロセスのエラーを再解釈せずに表示する").
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"designlab generate failed at {stage}: {message}")


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """The outcome of one successful (published or reused) generate job."""

    publish_path: Path
    publish_name: str
    config_digest: str
    reused: bool


class _StageTimer:
    """Prints ``STAGE <transaction> <stage> <elapsed_s>`` lines as stages advance.

    Mirrors ``mitsuba_stage_core._StageTimer``: each ``advance`` call logs
    the just-finished stage's elapsed time before starting the next one,
    and invokes ``on_stage`` with the newly-started stage's name so a
    caller (GUI status line) can show live progress.
    """

    def __init__(self, transaction: str, on_stage: Callable[[str], None]) -> None:
        self.transaction = transaction
        self._on_stage = on_stage
        self._stage: str | None = None
        self._stage_started = time.perf_counter()
        self._total_started = self._stage_started

    def advance(self, stage: str) -> None:
        now = time.perf_counter()
        if self._stage is not None:
            elapsed = now - self._stage_started
            print(f"STAGE {self.transaction} {self._stage} {elapsed:.3f}")
        self._stage = stage
        self._stage_started = now
        self._on_stage(stage)

    def finish(self) -> float:
        now = time.perf_counter()
        if self._stage is not None:
            elapsed = now - self._stage_started
            print(f"STAGE {self.transaction} {self._stage} {elapsed:.3f}")
        return now - self._total_started


def validate_name(name: str) -> None:
    if not _NAME_PATTERN.match(name):
        raise PublishError(
            "validate", f"name must match {_NAME_PATTERN.pattern!r}, got {name!r}"
        )


def publish_name_for(
    method: GeneratorMethod, name: str, config: GeneratorConfig
) -> str:
    """Return ``<method_id>-<name>-<digest12>`` (roadmap naming convention)."""
    digest = config_digest(config)
    short_digest = digest.removeprefix("sha256:")[:12]
    return f"{method.method_id}-{name}-{short_digest}"


def _is_valid_bundle(path: Path) -> bool:
    has_run_manifest = (path / _RUN_MANIFEST_NAME).is_file()
    has_optical_zarr = (path / _OPTICAL_ZARR_NAME).exists()
    return has_run_manifest and has_optical_zarr


def check_roots(output_root: Path, work_root: Path) -> tuple[Path, Path]:
    """Resolve and validate ``--output-root``/``--work-root``.

    Both must already exist. ``work_root`` must not be, or be inside,
    ``output_root`` (the viewer's Input catalog walks dot-directories too,
    so a work directory left under the output root could be mistaken for a
    published bundle).
    """
    resolved_output = output_root.resolve()
    if not resolved_output.is_dir():
        raise PublishError(
            "validate", f"--output-root is not a directory: {output_root}"
        )
    resolved_work = work_root.resolve()
    if not resolved_work.is_dir():
        raise PublishError(
            "validate", f"--work-root is not a directory: {work_root}"
        )
    work_inside_output = (
        resolved_work == resolved_output
        or resolved_work.is_relative_to(resolved_output)
    )
    if work_inside_output:
        raise PublishError(
            "validate", f"--work-root must not be inside --output-root: {work_root}"
        )
    return resolved_output, resolved_work


def default_work_root(output_root: Path) -> Path:
    """The default ``--work-root``: an output-root sibling directory."""
    resolved = output_root.resolve()
    return resolved.parent / f"{resolved.name}.designlab-work"


def sweep_stale_work_dirs(work_root: Path) -> None:
    """Remove every entry directly under ``work_root`` (startup cleanup only).

    ``work_root`` is exclusively designlab's own scratch space (never
    shared with the viewer or anything else), so on process startup — when
    no job can possibly be running — every leftover entry is stale, the
    same assumption the viewer's own ``--work-dir`` sweep makes.
    """
    resolved = work_root.resolve()
    if not resolved.is_dir():
        return
    for child in resolved.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _run_subprocess(stage: str, argv: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr.strip() or result.stdout.strip()
        fallback = f"subprocess exited {result.returncode}: {argv}"
        raise PublishError(stage, tail or fallback)
    return result


def _write_run_config(work_dir: Path, name: str, voxels_filename: str) -> Path:
    document = {
        "schema": {"name": "vdbmat.pipeline-config", "version": "2.0.0"},
        "input": {"kind": "direct-voxel", "path": voxels_filename},
        "mapping": {"name": _BUILTIN_MAPPING_NAME},
        "stages": {"validate_material": True, "validate_optical": True, "exports": []},
        "output": {"path": _BUNDLE_DIRNAME, "overwrite": False},
        "execution": {"random_seed": 0},
        "renderer": None,
    }
    run_config_path = work_dir / _RUN_MANIFEST_NAME
    run_config_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return run_config_path


def run_generate_job(
    *,
    method: GeneratorMethod,
    config: GeneratorConfig,
    name: str,
    output_root: Path,
    work_root: Path,
    seq: int,
    on_stage: Callable[[str], None] = lambda stage: None,
) -> GenerateResult:
    """Run one full generate transaction; return its published/reused bundle.

    ``config`` is assumed already validated by construction (it came from
    ``method.form_to_config`` or ``method.config_cls.from_json``, both of
    which raise on invalid fields before this is ever called).
    """
    transaction = f"generate:{name}"
    timer = _StageTimer(transaction, on_stage)
    timer.advance("validate")

    validate_name(name)
    resolved_output_root, resolved_work_root = check_roots(output_root, work_root)
    publish_name = publish_name_for(method, name, config)
    dest = resolved_output_root / publish_name

    if dest.exists():
        if _is_valid_bundle(dest):
            timer.advance("reused")
            timer.finish()
            return GenerateResult(
                publish_path=dest,
                publish_name=publish_name,
                config_digest=config_digest(config),
                reused=True,
            )
        raise PublishError(
            "validate",
            f"publish target exists and is not a valid bundle: {dest}; "
            "delete it manually before retrying (no automatic overwrite)",
        )

    work_dir = resolved_work_root / f"{seq:03d}-{name}"
    work_dir.mkdir(parents=True)

    config_path = work_dir / f"{name}{method.config_suffix}"
    config_path.write_text(config.to_json(), encoding="utf-8")

    timer.advance("generate")
    _run_subprocess("generate", method.generator_argv(config_path, work_dir, name))
    voxels_manifest = work_dir / f"{name}.voxels.json"
    if not voxels_manifest.is_file():
        raise PublishError(
            "generate",
            f"generator did not produce expected manifest: {voxels_manifest}",
        )

    timer.advance("map")
    run_config_path = _write_run_config(work_dir, name, voxels_manifest.name)
    _run_subprocess(
        "map",
        [
            sys.executable,
            "-m",
            "vdbmat.cli.main",
            "run",
            str(run_config_path),
            "--json",
        ],
    )
    bundle_path = work_dir / _BUNDLE_DIRNAME
    if not _is_valid_bundle(bundle_path):
        raise PublishError("map", f"run did not produce a valid bundle: {bundle_path}")

    timer.advance("verify")
    _run_subprocess(
        "verify",
        [
            sys.executable,
            "-m",
            "vdbmat.cli.main",
            "validate",
            str(bundle_path),
            "--json",
        ],
    )

    timer.advance("publish")
    try:
        os.replace(bundle_path, dest)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise PublishError(
                "publish",
                f"cannot atomically publish across filesystems (EXDEV) from "
                f"{bundle_path} to {dest}; pass --work-root on the same "
                "filesystem as --output-root",
            ) from error
        raise PublishError("publish", str(error)) from error

    timer.finish()
    shutil.rmtree(work_dir, ignore_errors=True)

    return GenerateResult(
        publish_path=dest,
        publish_name=publish_name,
        config_digest=config_digest(config),
        reused=False,
    )


__all__ = [
    "STAGES",
    "GenerateResult",
    "PublishError",
    "check_roots",
    "default_work_root",
    "publish_name_for",
    "run_generate_job",
    "sweep_stale_work_dirs",
    "validate_name",
]
