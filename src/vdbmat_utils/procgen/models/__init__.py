"""Composable procedural formation models (phase 3 step 3)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialDefinition, MaterialRole

from vdbmat_utils.core import build_material_label_volume
from vdbmat_utils.core.config import GeneratorConfig, config_digest
from vdbmat_utils.core.errors import ConfigError, PaletteError
from vdbmat_utils.core.provenance import build_provenance
from vdbmat_utils.fields import quantize_to_labels
from vdbmat_utils.io import write_asset
from vdbmat_utils.io.optical_mapping import (
    MappingEmission,
    build_mapping_config,
    maybe_write_mapping,
)
from vdbmat_utils.procgen.cells import worley
from vdbmat_utils.procgen.domain import (
    MAX_AXIS_CELLS,
    MAX_TOTAL_CELLS,
    FormationDomain,
)
from vdbmat_utils.procgen.hashing import hash_derive, hash_to_unit
from vdbmat_utils.procgen.morphology import open_mask
from vdbmat_utils.procgen.noise import fbm, fbm_at, ridged_fbm
from vdbmat_utils.procgen.stats import (
    ConstraintResult,
    FormationStats,
    compute_stats,
    evaluate_constraints,
    write_stats_report,
)

GENERATOR = "vdbmat-utils.procgen.formation"
GENERATOR_VERSION = "0.1.0"
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclasses.dataclass(frozen=True)
class FormationConfig(GeneratorConfig):
    shape_zyx: tuple[int, int, int] = (16, 16, 16)
    voxel_size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    palette: tuple[dict[str, Any], ...] = ()
    layers: tuple[dict[str, Any], ...] = ()
    constraints: tuple[dict[str, Any], ...] = ()
    mapping: dict[str, Any] | None = None
    local_to_world: tuple[tuple[float, ...], ...] | None = None
    max_axis_cells: int = MAX_AXIS_CELLS
    max_total_cells: int = MAX_TOTAL_CELLS

    @classmethod
    def from_json(cls, text: str) -> FormationConfig:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"invalid configuration JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise ConfigError("formation configuration must be an object")
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ConfigError(f"unknown configuration fields: {', '.join(unknown)}")
        return cls(
            seed=int(payload.get("seed", 0)),
            shape_zyx=_int3(payload.get("shape_zyx", (16, 16, 16)), "shape_zyx"),
            voxel_size_xyz_m=_float3(
                payload.get("voxel_size_xyz_m", (1.0, 1.0, 1.0)),
                "voxel_size_xyz_m",
            ),
            palette=tuple(_dicts(payload.get("palette", ()), "palette")),
            layers=tuple(_dicts(payload.get("layers", ()), "layers")),
            constraints=tuple(_dicts(payload.get("constraints", ()), "constraints")),
            mapping=_optional_dict(payload.get("mapping"), "mapping"),
            local_to_world=(
                None
                if payload.get("local_to_world") is None
                else tuple(
                    tuple(float(v) for v in row)
                    for row in payload["local_to_world"]
                )
            ),
            max_axis_cells=int(payload.get("max_axis_cells", MAX_AXIS_CELLS)),
            max_total_cells=int(payload.get("max_total_cells", MAX_TOTAL_CELLS)),
        )


@dataclasses.dataclass(frozen=True)
class FormationResult:
    volume: Any
    stats: FormationStats
    constraints: tuple[ConstraintResult, ...]
    mapping: MappingEmission


@dataclasses.dataclass(frozen=True)
class WrittenFormation:
    manifest: Path
    stats_path: Path
    mapping_path: Path | None
    mapping_digest: str
    constraints: tuple[ConstraintResult, ...]


def generate_formation(config: FormationConfig, *, name: str) -> FormationResult:
    domain = FormationDomain(
        shape_zyx=config.shape_zyx,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        local_to_world=config.local_to_world
        if config.local_to_world is not None
        else FormationDomain(config.shape_zyx, config.voxel_size_xyz_m).local_to_world,
        max_axis_cells=config.max_axis_cells,
        max_total_cells=config.max_total_cells,
    )
    palette = _palette(config.palette)
    palette_ids = {item.material_id for item in palette}
    labels = _paint_layers(domain, config.layers, palette_ids, seed=config.seed)
    mapping_config, mapping_required = build_mapping_config(
        palette=palette, mapping=config.mapping, name=name
    )
    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=(f"mapping-digest:{mapping_config.digest}",),
        notes=f"procedural formation {name}; mapping digest recorded in sources",
    )
    volume = build_material_label_volume(
        material_id=labels,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        palette=palette,
        provenance=provenance,
        local_to_world=config.local_to_world,
    )
    stats = compute_stats(volume)
    constraints = evaluate_constraints(stats, config.constraints)
    # The concrete mapping path is only known during writing; digest is filled
    # there and provenance is rebuilt to record it.
    return FormationResult(
        volume=volume,
        stats=stats,
        constraints=constraints,
        mapping=MappingEmission(
            path=None, digest=mapping_config.digest, required=mapping_required
        ),
    )


def write_formation(
    config: FormationConfig,
    *,
    out: str | Path,
    name: str,
) -> WrittenFormation:
    domain = FormationDomain(
        shape_zyx=config.shape_zyx,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        local_to_world=config.local_to_world
        if config.local_to_world is not None
        else FormationDomain(config.shape_zyx, config.voxel_size_xyz_m).local_to_world,
        max_axis_cells=config.max_axis_cells,
        max_total_cells=config.max_total_cells,
    )
    palette = _palette(config.palette)
    mapping = maybe_write_mapping(
        directory=out, name=name, palette=palette, mapping=config.mapping
    )
    labels = _paint_layers(
        domain,
        config.layers,
        {item.material_id for item in palette},
        seed=config.seed,
    )
    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=(f"mapping-digest:{mapping.digest}",),
        notes=(
            "procedural formation; constraints are measured, not repaired; "
            f"mapping_digest={mapping.digest}"
        ),
    )
    volume = build_material_label_volume(
        material_id=labels,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        palette=palette,
        provenance=provenance,
        local_to_world=config.local_to_world,
    )
    stats = compute_stats(volume)
    constraints = evaluate_constraints(stats, config.constraints)
    identity = _formation_identity(config)
    manifest = write_asset(volume, out, name, identity=identity)
    stats_path = write_stats_report(
        Path(out) / f"{name}.stats.json", stats, constraints
    )
    return WrittenFormation(
        manifest=manifest,
        stats_path=stats_path,
        mapping_path=mapping.path,
        mapping_digest=mapping.digest,
        constraints=constraints,
    )


def _paint_layers(
    domain: FormationDomain,
    layers: Sequence[Mapping[str, Any]],
    palette_ids: set[int],
    *,
    seed: int,
) -> npt.NDArray[np.uint16]:
    if not layers:
        raise ConfigError("layers must contain at least one host layer")
    labels: npt.NDArray[np.uint16] | None = None
    for index, layer in enumerate(layers):
        kind = layer.get("kind")
        if index == 0 and kind != "host":
            raise ConfigError("layers[0] must be kind 'host'")
        stream = 10_000 + index * 100
        if kind == "host":
            labels = _host(domain, layer, seed=seed, stream_id=stream)
        else:
            if labels is None:
                raise ConfigError("host layer did not produce labels")
            mask, values = _overlay(domain, layer, seed=seed, stream_id=stream)
            _require_palette(values[mask], palette_ids, layer_index=index)
            labels = labels.copy()
            labels[mask] = values[mask]
        _require_palette(labels, palette_ids, layer_index=index)
    assert labels is not None
    return labels


def _host(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> npt.NDArray[np.uint16]:
    if "material_id" in layer:
        material_id = _material_id(layer["material_id"], "layers[].material_id")
        return np.full(domain.shape_zyx, material_id, dtype=np.uint16)
    if "bin_edges" in layer and "material_ids" in layer:
        field = fbm(
            domain,
            frequency_per_m=float(layer.get("frequency_per_m", 1.0)),
            octaves=int(layer.get("octaves", 3)),
            stream_id=stream_id,
            seed=seed,
        )
        return quantize_to_labels(
            field,
            bin_edges=tuple(float(v) for v in layer["bin_edges"]),
            material_ids=tuple(
                _material_id(v, "layers[].material_ids")
                for v in layer["material_ids"]
            ),
        )
    raise ConfigError("host layer requires material_id or bin_edges/material_ids")


def _overlay(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint16]]:
    kind = layer.get("kind")
    if kind == "strata":
        return _strata(domain, layer, seed=seed, stream_id=stream_id)
    if kind == "veins":
        return _veins(domain, layer, seed=seed, stream_id=stream_id)
    if kind == "fractures":
        return _veins(domain, layer, seed=seed, stream_id=stream_id)
    if kind == "grains":
        return _grains(domain, layer, seed=seed, stream_id=stream_id)
    if kind == "pores":
        return _pores(domain, layer, seed=seed, stream_id=stream_id)
    raise ConfigError(f"unsupported layer kind {kind!r}")


def _strata(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint16]]:
    coordinate = _axis_coordinate(domain, str(layer.get("axis", "z")))
    thickness = float(layer["thickness_m"])
    if thickness <= 0:
        raise ConfigError("strata thickness_m must be positive")
    coordinate = coordinate + _warp_offset(
        domain, layer, seed=seed, stream_id=stream_id
    )
    material_ids = tuple(
        _material_id(v, "strata.material_ids") for v in layer["material_ids"]
    )
    bands = np.floor(coordinate / thickness).astype(np.int64)
    values = np.asarray(material_ids, dtype=np.uint16)[np.mod(bands, len(material_ids))]
    return (
        np.ones(domain.shape_zyx, dtype=np.bool_),
        np.broadcast_to(values, domain.shape_zyx).copy(),
    )


def _veins(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint16]]:
    coordinate = _axis_coordinate(domain, str(layer.get("axis", "x")))
    coordinate = coordinate + _warp_offset(
        domain, layer, seed=seed, stream_id=stream_id
    )
    width = float(layer["width_m"])
    offset = float(layer.get("offset_m", 0.0))
    spacing = layer.get("spacing_m")
    if spacing is None:
        distance = np.abs(coordinate - offset)
    else:
        period = float(spacing)
        centered = ((coordinate - offset + 0.5 * period) % period) - 0.5 * period
        distance = np.abs(centered)
    mask = np.broadcast_to(distance < (0.5 * width), domain.shape_zyx)
    material_id = _material_id(layer["material_id"], "veins.material_id")
    return mask.copy(), np.full(domain.shape_zyx, material_id, dtype=np.uint16)


def _grains(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint16]]:
    cells = worley(
        domain,
        cell_size_m=float(layer["cell_size_m"]),
        stream_id=stream_id,
        seed=seed,
    )
    weighted = layer["material_ids"]
    ids, weights = _weighted_ids(weighted)
    draws = hash_to_unit(hash_derive(cells.site_id, salt=17))
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
    cumulative = cumulative / cumulative[-1]
    picks = np.searchsorted(cumulative, draws, side="right")
    values = np.asarray(ids, dtype=np.uint16)[picks]
    if "boundary_material_id" in layer:
        boundary = cells.boundary().values < float(layer.get("boundary_width_m", 0.0))
        values = np.where(
            boundary,
            np.uint16(
                _material_id(
                    layer["boundary_material_id"],
                    "grains.boundary_material_id",
                )
            ),
            values,
        )
    mask = np.ones(domain.shape_zyx, dtype=np.bool_)
    if "threshold" in layer:
        field = fbm(
            domain,
            frequency_per_m=float(layer.get("frequency_per_m", 1.0)),
            octaves=int(layer.get("octaves", 3)),
            stream_id=stream_id + 50,
            seed=seed,
        )
        mask = field.values > float(layer["threshold"])
    return mask, values.astype(np.uint16, copy=False)


def _pores(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.uint16]]:
    field_kind = str(layer.get("field", "fbm"))
    if field_kind == "ridged":
        field = ridged_fbm(
            domain,
            frequency_per_m=float(layer.get("frequency_per_m", 1.0)),
            octaves=int(layer.get("octaves", 3)),
            stream_id=stream_id,
            seed=seed,
        )
    else:
        field = fbm(
            domain,
            frequency_per_m=float(layer.get("frequency_per_m", 1.0)),
            octaves=int(layer.get("octaves", 3)),
            stream_id=stream_id,
            seed=seed,
        )
    mask = field.values > float(layer["threshold"])
    radius = int(layer.get("open_radius_cells", 0))
    if radius:
        mask = open_mask(
            mask,
            radius_cells=radius,
            connectivity=int(layer.get("connectivity", 6)),
        )
    material_id = _material_id(layer["material_id"], "pores.material_id")
    return mask, np.full(domain.shape_zyx, material_id, dtype=np.uint16)


def _warp_offset(
    domain: FormationDomain, layer: Mapping[str, Any], *, seed: int, stream_id: int
) -> npt.NDArray[np.float64]:
    amplitude = float(layer.get("warp_amplitude_m", 0.0))
    if amplitude == 0.0:
        return np.zeros(domain.shape_zyx, dtype=np.float64)
    x, y, z = domain.coordinates_xyz_m()
    values = fbm_at(
        x,
        y,
        z,
        frequency_per_m=float(layer.get("warp_frequency_per_m", 1.0)),
        octaves=int(layer.get("warp_octaves", 3)),
        stream_id=stream_id + 25,
        seed=seed,
    )
    return np.broadcast_to(values * amplitude, domain.shape_zyx)


def _axis_coordinate(domain: FormationDomain, axis: str) -> npt.NDArray[np.float64]:
    if axis not in _AXES:
        raise ConfigError(f"axis must be one of x, y, z; got {axis!r}")
    x, y, z = domain.coordinates_xyz_m()
    return {"x": x, "y": y, "z": z}[axis]


def _palette(raw: Sequence[Mapping[str, Any]]) -> tuple[MaterialDefinition, ...]:
    if not raw:
        raise ConfigError("palette must not be empty")
    result: list[MaterialDefinition] = []
    for index, item in enumerate(raw):
        material_id = _material_id(
            item.get("material_id"), f"palette[{index}].material_id"
        )
        role = item.get("role", "background" if material_id == 0 else "material")
        result.append(
            MaterialDefinition(
                material_id=material_id,
                name=str(item["name"]),
                role=MaterialRole(role),
            )
        )
    return tuple(result)


def _require_palette(
    labels: npt.NDArray[np.uint16], palette_ids: set[int], *, layer_index: int
) -> None:
    missing = sorted(int(v) for v in np.unique(labels) if int(v) not in palette_ids)
    if missing:
        raise PaletteError(
            f"layers[{layer_index}] emits undeclared material ids {missing}"
        )


def _weighted_ids(raw: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigError("grains.material_ids must be an array")
    ids: list[int] = []
    weights: list[float] = []
    for item in raw:
        if isinstance(item, Mapping):
            ids.append(_material_id(item["material_id"], "grains.material_ids"))
            weights.append(float(item.get("weight", 1.0)))
        else:
            ids.append(_material_id(item, "grains.material_ids"))
            weights.append(1.0)
    if not ids or any(weight <= 0 for weight in weights):
        raise ConfigError("grains.material_ids must contain positive weights")
    return tuple(ids), tuple(weights)


def _formation_identity(config: FormationConfig) -> str:
    digest = hashlib.sha256(config_digest(config).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _material_id(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if not 0 <= value <= 65535:
        raise ConfigError(f"{field} must be in [0, 65535]")
    return value


def _int3(value: Any, field: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ConfigError(f"{field} must be an array of 3 integers")
    return (int(value[0]), int(value[1]), int(value[2]))


def _float3(value: Any, field: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ConfigError(f"{field} must be an array of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _dicts(value: Any, field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{field} must be an array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ConfigError(f"{field}[{index}] must be an object")
        result.append(dict(item))
    return tuple(result)


def _optional_dict(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be an object")
    return dict(value)


__all__ = [
    "FormationConfig",
    "FormationResult",
    "WrittenFormation",
    "generate_formation",
    "write_formation",
]
