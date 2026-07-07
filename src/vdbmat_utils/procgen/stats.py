"""Formation statistics and constraint checks (phase 3 step 2)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.core.errors import ConfigError
from vdbmat_utils.fields import squared_edt

from .connectivity import connected_components

ConstraintKind = Literal[
    "volume-fraction",
    "min-feature-size",
    "connected",
    "min-largest-component-fraction",
    "min-printable-thickness",
]


@dataclasses.dataclass(frozen=True)
class MaterialStats:
    material_id: int
    name: str
    count: int
    volume_fraction: float
    local_thickness_m: dict[str, float | None]
    component_count: int
    largest_component_fraction: float
    min_printable_thickness_m: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FormationStats:
    shape_zyx: tuple[int, int, int]
    voxel_size_xyz_m: tuple[float, float, float]
    total_voxels: int
    materials: tuple[MaterialStats, ...]

    def by_material_id(self, material_id: int) -> MaterialStats:
        for item in self.materials:
            if item.material_id == material_id:
                return item
        raise ConfigError(f"material_id {material_id} is not present in stats")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "voxel_size_xyz_m": list(self.voxel_size_xyz_m),
            "total_voxels": self.total_voxels,
            "materials": [item.to_json_dict() for item in self.materials],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_json_dict())


@dataclasses.dataclass(frozen=True)
class ConstraintResult:
    kind: ConstraintKind
    material_id: int
    passed: bool
    measured: float | int | str | None
    bound: float | int | str | tuple[float, float] | None
    message: str

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_stats(volume: MaterialLabelVolume) -> FormationStats:
    """Compute D7 statistics for a material-label volume."""
    labels = np.asarray(volume.material_id)
    total = int(labels.size)
    spacing_zyx = (
        float(volume.geometry.voxel_size_xyz_m[2]),
        float(volume.geometry.voxel_size_xyz_m[1]),
        float(volume.geometry.voxel_size_xyz_m[0]),
    )
    materials: list[MaterialStats] = []
    for material in volume.palette:
        mask = labels == material.material_id
        count = int(np.count_nonzero(mask))
        thickness = _local_thickness_values(mask, spacing_zyx)
        components = connected_components(mask)
        largest = int(components.sizes.max()) if components.count else 0
        fraction = (count / total) if total else 0.0
        materials.append(
            MaterialStats(
                material_id=material.material_id,
                name=material.name,
                count=count,
                volume_fraction=fraction,
                local_thickness_m=_percentiles(thickness),
                component_count=components.count,
                largest_component_fraction=(largest / count) if count else 0.0,
                min_printable_thickness_m=(
                    float(np.min(thickness)) if thickness.size else None
                ),
            )
        )
    return FormationStats(
        shape_zyx=(
            int(volume.geometry.shape_zyx[0]),
            int(volume.geometry.shape_zyx[1]),
            int(volume.geometry.shape_zyx[2]),
        ),
        voxel_size_xyz_m=(
            float(volume.geometry.voxel_size_xyz_m[0]),
            float(volume.geometry.voxel_size_xyz_m[1]),
            float(volume.geometry.voxel_size_xyz_m[2]),
        ),
        total_voxels=total,
        materials=tuple(materials),
    )


def evaluate_constraints(
    stats: FormationStats, constraints: Sequence[Mapping[str, Any]]
) -> tuple[ConstraintResult, ...]:
    """Evaluate declarative constraints against computed stats.

    Supported JSON forms:
    ``{"kind":"volume-fraction","material_id":1,"min":0.1,"max":0.4}``,
    ``{"kind":"min-feature-size","material_id":1,"threshold_m":0.002}``,
    ``{"kind":"connected","material_id":1,"mode":"single-component"}``,
    ``{"kind":"connected","material_id":1,"max_components":2}``,
    ``{"kind":"min-largest-component-fraction","material_id":1,"min":0.95}``,
    and ``{"kind":"min-printable-thickness","material_id":1,"threshold_m":...}``.
    """
    results: list[ConstraintResult] = []
    for index, raw in enumerate(constraints):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"constraints[{index}] must be an object")
        kind = raw.get("kind")
        material_id = _material_id(raw, index=index)
        item = stats.by_material_id(material_id)
        if kind == "volume-fraction":
            minimum = float(raw.get("min", 0.0))
            maximum = float(raw.get("max", 1.0))
            measured = item.volume_fraction
            passed = minimum <= measured <= maximum
            results.append(
                ConstraintResult(
                    kind="volume-fraction",
                    material_id=material_id,
                    passed=passed,
                    measured=measured,
                    bound=(minimum, maximum),
                    message=_message(
                        passed,
                        f"{measured:g} in [{minimum:g}, {maximum:g}]",
                    ),
                )
            )
        elif kind == "min-feature-size":
            threshold = float(raw["threshold_m"])
            p05_thickness = item.local_thickness_m["p05"]
            passed = p05_thickness is not None and p05_thickness >= threshold
            results.append(
                ConstraintResult(
                    kind="min-feature-size",
                    material_id=material_id,
                    passed=passed,
                    measured=p05_thickness,
                    bound=threshold,
                    message=_message(passed, f"p05 thickness >= {threshold:g} m"),
                )
            )
        elif kind == "connected":
            max_components = (
                1
                if raw.get("mode") == "single-component"
                else int(raw.get("max_components", raw.get("max-components", 1)))
            )
            measured = item.component_count
            passed = measured <= max_components
            results.append(
                ConstraintResult(
                    kind="connected",
                    material_id=material_id,
                    passed=passed,
                    measured=measured,
                    bound=max_components,
                    message=_message(
                        passed,
                        f"{measured} <= {max_components} components",
                    ),
                )
            )
        elif kind == "min-largest-component-fraction":
            minimum = float(raw["min"])
            measured = item.largest_component_fraction
            passed = measured >= minimum
            results.append(
                ConstraintResult(
                    kind="min-largest-component-fraction",
                    material_id=material_id,
                    passed=passed,
                    measured=measured,
                    bound=minimum,
                    message=_message(passed, f"{measured:g} >= {minimum:g}"),
                )
            )
        elif kind == "min-printable-thickness":
            threshold = float(raw["threshold_m"])
            min_thickness = item.min_printable_thickness_m
            passed = min_thickness is not None and min_thickness >= threshold
            results.append(
                ConstraintResult(
                    kind="min-printable-thickness",
                    material_id=material_id,
                    passed=passed,
                    measured=min_thickness,
                    bound=threshold,
                    message=_message(passed, f"min thickness >= {threshold:g} m"),
                )
            )
        else:
            raise ConfigError(f"constraints[{index}].kind is unsupported: {kind!r}")
    return tuple(results)


def stats_report_dict(
    stats: FormationStats, constraints: Sequence[ConstraintResult] = ()
) -> dict[str, Any]:
    return {
        "stats": stats.to_json_dict(),
        "constraints": [item.to_json_dict() for item in constraints],
    }


def write_stats_report(
    path: str | Path,
    stats: FormationStats,
    constraints: Sequence[ConstraintResult] = (),
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _canonical_json(stats_report_dict(stats, constraints)) + "\n",
        encoding="utf-8",
    )
    return destination


def _local_thickness_values(
    mask: npt.NDArray[np.bool_], spacing_zyx: tuple[float, float, float]
) -> npt.NDArray[np.float64]:
    if mask.dtype != np.bool_ or mask.ndim != 3:
        from . import ProcgenError

        raise ProcgenError(
            f"mask must be a 3-D bool array, got {mask.ndim}-D {mask.dtype}"
        )
    if not mask.any():
        return np.asarray([], dtype=np.float64)
    if mask.all():
        diagonal = float(
            np.sqrt(
                sum(
                    (cells * spacing) ** 2
                    for cells, spacing in zip(mask.shape, spacing_zyx, strict=True)
                )
            )
        )
        return np.full(int(mask.size), diagonal, dtype=np.float64)
    distances = np.sqrt(squared_edt(~mask, spacing_zyx))
    values: npt.NDArray[np.float64] = 2.0 * distances[mask]
    return values


def _percentiles(values: npt.NDArray[np.float64]) -> dict[str, float | None]:
    if not values.size:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _material_id(raw: Mapping[str, Any], *, index: int) -> int:
    value = raw.get("material_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"constraints[{index}].material_id must be an integer")
    return value


def _message(passed: bool, detail: str) -> str:
    return f"{'pass' if passed else 'fail'}: {detail}"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
