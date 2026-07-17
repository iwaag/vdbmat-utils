"""Config catalog: scan / load / save under a server-local ``--config-root``.

Mirrors the containment and non-destructive conventions the viewer already
established for its own catalogs (``mitsuba_stage_inputs.py``,
``mitsuba_stage_presets.py``): paths are confined to an explicit root,
resolved-path escapes (including via symlink) are excluded rather than
raised during scanning, and a save never overwrites an existing file. No
browser upload path exists — every config is a server-local file the human
operator placed under ``--config-root`` themselves.

Load/save operate on a :class:`~vdbmat_utils.core.GeneratorConfig` and don't
know about any particular method's fields; the caller supplies the
registered ``GeneratorMethod`` to pick a suffix/config class.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from designlab_registry import GeneratorMethod, method_for_config_path

from vdbmat_utils.core import GeneratorConfig
from vdbmat_utils.core.errors import ConfigError

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class DesignlabConfigError(Exception):
    """A config catalog operation failed."""


@dataclass(frozen=True, slots=True)
class ConfigCandidate:
    """One catalog entry: a config file matched to a registered method."""

    method: GeneratorMethod
    root_relative: str
    path: Path


def _is_contained(root: Path, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or resolved.is_relative_to(root)


def resolve_config_root(cli_root: Path) -> Path:
    """Return the resolved, existence-checked config root."""
    resolved = cli_root.resolve()
    if not resolved.is_dir():
        raise DesignlabConfigError(f"--config-root is not a directory: {cli_root}")
    return resolved


def scan_config_catalog(root: Path) -> list[ConfigCandidate]:
    """Enumerate registered-method config files under ``root`` (recursive).

    A file whose resolved real path escapes ``root`` (a root-outer symlink)
    is silently excluded, matching the input catalog's containment rule. A
    file that does not match any registered method's ``config_suffix`` is
    skipped.
    """
    root = root.resolve()
    found: list[ConfigCandidate] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames.sort()
        for filename in sorted(filenames):
            candidate_path = current / filename
            if not _is_contained(root, candidate_path):
                continue
            method = method_for_config_path(candidate_path)
            if method is None:
                continue
            found.append(
                ConfigCandidate(
                    method=method,
                    root_relative=candidate_path.relative_to(root).as_posix(),
                    path=candidate_path,
                )
            )
    found.sort(key=lambda item: item.root_relative)
    return found


def resolve_config_path(root: Path, user_path: Path) -> Path:
    """Resolve a config path under ``root``, rejecting an escape or a miss."""
    root = root.resolve()
    candidate_path = user_path if user_path.is_absolute() else root / user_path
    if not candidate_path.is_file():
        raise DesignlabConfigError(f"config does not exist: {user_path}")
    resolved = candidate_path.resolve()
    if not _is_contained(root, resolved):
        raise DesignlabConfigError(
            f"config resolves outside --config-root: {user_path}"
        )
    return resolved


def load_config(path: Path, root: Path, method: GeneratorMethod) -> GeneratorConfig:
    """Load and parse a config file, containment-checked against ``root``.

    Parse errors (unknown fields, validation failures) come straight from
    ``method.config_cls.from_json`` and are not reworded here (roadmap:
    the GUI does not re-interpret error messages).
    """
    resolved = resolve_config_path(root, path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as error:
        raise DesignlabConfigError(f"cannot read config: {error}") from error
    try:
        return method.config_cls.from_json(text)
    except ConfigError as error:
        raise DesignlabConfigError(str(error)) from error


def save_config(
    config: GeneratorConfig, root: Path, method: GeneratorMethod, stem_name: str
) -> Path:
    """Write ``config`` as ``<root>/<stem_name><method.config_suffix>``.

    Refuses to overwrite an existing file (roadmap non-goal: destructive
    overwrite of an existing config) and refuses a ``stem_name`` that is not
    a bare, root-relative-safe token — no path separators, no leading dot.
    """
    root = root.resolve()
    if not _NAME_PATTERN.match(stem_name):
        raise DesignlabConfigError(
            f"config name must match {_NAME_PATTERN.pattern!r}, got {stem_name!r}"
        )
    target = root / f"{stem_name}{method.config_suffix}"
    if not _is_contained(root, target):
        raise DesignlabConfigError(
            f"config name resolves outside --config-root: {stem_name}"
        )
    if target.exists():
        raise DesignlabConfigError(
            f"refusing to overwrite existing config: {target}; choose a different name"
        )
    target.write_text(config.to_json(), encoding="utf-8")
    return target


__all__ = [
    "ConfigCandidate",
    "DesignlabConfigError",
    "load_config",
    "resolve_config_path",
    "resolve_config_root",
    "save_config",
    "scan_config_catalog",
]
