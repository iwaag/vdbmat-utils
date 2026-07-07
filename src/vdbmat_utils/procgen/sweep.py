"""Parameter sweeps for procedural formations."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vdbmat_utils.core.config import GeneratorConfig, config_digest
from vdbmat_utils.core.errors import ConfigError
from vdbmat_utils.procgen.models import FormationConfig, write_formation


@dataclasses.dataclass(frozen=True)
class SweepConfig(GeneratorConfig):
    base: dict[str, Any] = dataclasses.field(default_factory=dict)
    axes: tuple[dict[str, Any], ...] = ()
    max_runs: int = 64

    @classmethod
    def from_json(cls, text: str) -> SweepConfig:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"invalid sweep configuration JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise ConfigError("sweep configuration must be an object")
        unknown = sorted(set(payload) - {"seed", "base", "axes", "max_runs"})
        if unknown:
            raise ConfigError(
                f"unknown sweep configuration fields: {', '.join(unknown)}"
            )
        base = payload.get("base")
        if not isinstance(base, Mapping):
            raise ConfigError("base must be a FormationConfig object")
        axes = payload.get("axes", ())
        if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
            raise ConfigError("axes must be an array")
        parsed_axes: list[dict[str, Any]] = []
        for index, axis in enumerate(axes):
            if not isinstance(axis, Mapping):
                raise ConfigError(f"axes[{index}] must be an object")
            path = axis.get("path")
            values = axis.get("values")
            if not isinstance(path, str) or not path:
                raise ConfigError(f"axes[{index}].path must be a non-empty string")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ConfigError(f"axes[{index}].values must be an array")
            parsed_axes.append({"path": path, "values": list(values)})
        return cls(
            seed=int(payload.get("seed", 0)),
            base=dict(base),
            axes=tuple(parsed_axes),
            max_runs=int(payload.get("max_runs", 64)),
        )


@dataclasses.dataclass(frozen=True)
class SweepRun:
    index: int
    directory: Path
    manifest: Path
    stats_path: Path
    mapping_path: Path | None
    mapping_digest: str
    config_digest: str
    payload_digest: str
    overrides: dict[str, Any]
    stats: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class SweepResult:
    summary_path: Path
    runs: tuple[SweepRun, ...]


def run_sweep(config: SweepConfig, *, out: str | Path, name: str) -> SweepResult:
    combinations = _axis_combinations(config.axes)
    if len(combinations) > config.max_runs:
        raise ConfigError(
            f"sweep declares {len(combinations)} runs, "
            f"exceeding max_runs={config.max_runs}"
        )
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    runs: list[SweepRun] = []
    for index, overrides in enumerate(combinations):
        payload = _apply_overrides(config.base, overrides)
        if "seed" not in payload and config.seed != 0:
            payload["seed"] = config.seed
        formation = FormationConfig.from_json(_canonical_json(payload))
        run_name = f"{name}-{index:03d}"
        run_dir = root / run_name
        written = write_formation(formation, out=run_dir, name=run_name)
        stats_payload = json.loads(written.stats_path.read_text(encoding="utf-8"))
        payload_digest = hashlib.sha256(
            (run_dir / f"{run_name}.material_id.npy").read_bytes()
        ).hexdigest()
        runs.append(
            SweepRun(
                index=index,
                directory=run_dir,
                manifest=written.manifest,
                stats_path=written.stats_path,
                mapping_path=written.mapping_path,
                mapping_digest=written.mapping_digest,
                config_digest=config_digest(formation),
                payload_digest=f"sha256:{payload_digest}",
                overrides=overrides,
                stats=stats_payload,
            )
        )
    summary_path = root / "sweep_summary.json"
    summary_path.write_text(
        _canonical_json(_summary_dict(config, name, runs, root=root)) + "\n",
        encoding="utf-8",
    )
    return SweepResult(summary_path=summary_path, runs=tuple(runs))


def _axis_combinations(axes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    combinations: list[dict[str, Any]] = [{}]
    for axis in axes:
        path = str(axis["path"])
        values = axis["values"]
        combinations = [
            {**base, path: value}
            for base in combinations
            for value in values
        ]
    return combinations


def _apply_overrides(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    payload = json.loads(_canonical_json(base))
    if not isinstance(payload, dict):
        raise ConfigError("base must be an object")
    for path, value in overrides.items():
        _set_path(payload, path.split("."), value)
    return payload


def _set_path(container: Any, parts: list[str], value: Any) -> None:
    if not parts:
        raise ConfigError("override path must not be empty")
    head = parts[0]
    if len(parts) == 1:
        if isinstance(container, list):
            container[_list_index(head)] = value
        elif isinstance(container, dict):
            container[head] = value
        else:
            raise ConfigError(f"cannot set override path segment {head!r}")
        return
    if isinstance(container, list):
        child = container[_list_index(head)]
    elif isinstance(container, dict):
        if head not in container:
            raise ConfigError(f"override path references missing field {head!r}")
        child = container[head]
    else:
        raise ConfigError(f"cannot descend into override path segment {head!r}")
    _set_path(child, parts[1:], value)


def _list_index(raw: str) -> int:
    try:
        index = int(raw)
    except ValueError as error:
        raise ConfigError(
            f"list override segment must be an integer, got {raw!r}"
        ) from error
    if index < 0:
        raise ConfigError("list override segment must be non-negative")
    return index


def _summary_dict(
    config: SweepConfig, name: str, runs: Sequence[SweepRun], *, root: Path
) -> dict[str, Any]:
    return {
        "name": name,
        "config_digest": config_digest(config),
        "run_count": len(runs),
        "runs": [
            {
                "index": run.index,
                "directory": _relative(run.directory, root),
                "manifest": _relative(run.manifest, root),
                "stats_path": _relative(run.stats_path, root),
                "mapping_path": (
                    None
                    if run.mapping_path is None
                    else _relative(run.mapping_path, root)
                ),
                "mapping_digest": run.mapping_digest,
                "config_digest": run.config_digest,
                "payload_digest": run.payload_digest,
                "overrides": run.overrides,
                "stats": run.stats,
            }
            for run in runs
        ],
}


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = ["SweepConfig", "SweepResult", "SweepRun", "run_sweep"]
