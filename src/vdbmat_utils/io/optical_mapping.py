"""Optical-mapping emission for procedural formations."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vdbmat.core import MaterialDefinition, SchemaVersion
from vdbmat.optics import (
    MaterialOpticalProperties,
    OpticalMappingConfig,
    phase0_provisional_mapping,
)
from vdbmat.optics import (
    write_optical_mapping as write_vdbmat_optical_mapping,
)

from vdbmat_utils.core.errors import ConfigError


@dataclasses.dataclass(frozen=True)
class MappingEmission:
    path: Path | None
    digest: str
    required: bool


def built_in_material_names() -> frozenset[str]:
    return frozenset(item.name for item in phase0_provisional_mapping().materials)


def build_mapping_config(
    *,
    palette: Sequence[MaterialDefinition],
    mapping: Mapping[str, Any] | None,
    name: str,
) -> tuple[OpticalMappingConfig, bool]:
    """Build the mapping config for a formation palette.

    Non-built-in material names must be supplied in ``mapping["materials"]``.
    Built-ins are filled from vdbmat's public Phase 0 provisional mapping so
    the emitted document covers the whole palette and can be used directly by
    ``vdbmat convert``.
    """
    built_ins = {item.name: item for item in phase0_provisional_mapping().materials}
    required_names = {item.name for item in palette if item.name not in built_ins}
    required = bool(required_names)
    supplied = _supplied_materials(mapping)
    supplied_names = set(supplied)
    if required_names != supplied_names:
        missing = sorted(required_names - supplied_names)
        extra = sorted(supplied_names - required_names)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"extra {extra}")
        raise ConfigError(
            "mapping.materials must cover exactly non-built-in palette names"
            + (f" ({'; '.join(details)})" if details else "")
        )
    if not required and mapping is None:
        builtin_config = phase0_provisional_mapping()
        return builtin_config, False

    materials: list[MaterialOpticalProperties] = []
    for entry in palette:
        if entry.name in supplied:
            raw = supplied[entry.name]
            materials.append(
                MaterialOpticalProperties(
                    material_id=entry.material_id,
                    name=entry.name,
                    sigma_a_rgb_per_m=_triple(raw, "sigma_a_rgb_per_m"),
                    sigma_s_rgb_per_m=_triple(raw, "sigma_s_rgb_per_m"),
                    g=raw["g"],
                    ior=raw["ior"],
                )
            )
        else:
            builtin_material = built_ins[entry.name]
            materials.append(
                MaterialOpticalProperties(
                    material_id=entry.material_id,
                    name=entry.name,
                    sigma_a_rgb_per_m=builtin_material.sigma_a_rgb_per_m,
                    sigma_s_rgb_per_m=builtin_material.sigma_s_rgb_per_m,
                    g=builtin_material.g,
                    ior=builtin_material.ior,
                )
            )
    configuration_id = (
        str(mapping.get("configuration_id"))
        if mapping is not None and mapping.get("configuration_id") is not None
        else f"{name}-materials-v1"
    )
    calibration_status = (
        str(mapping.get("calibration_status"))
        if mapping is not None and mapping.get("calibration_status") is not None
        else "provisional-uncalibrated"
    )
    return (
        OpticalMappingConfig(
            configuration_id=configuration_id,
            version=SchemaVersion.parse("1.0.0"),
            materials=tuple(materials),
            calibration_status=calibration_status,  # type: ignore[arg-type]
        ),
        required,
    )


def maybe_write_mapping(
    *,
    directory: str | Path,
    name: str,
    palette: Sequence[MaterialDefinition],
    mapping: Mapping[str, Any] | None,
) -> MappingEmission:
    config, required = build_mapping_config(palette=palette, mapping=mapping, name=name)
    if not required:
        return MappingEmission(path=None, digest=config.digest, required=False)
    path = Path(directory) / f"{name}.optical-mapping.json"
    write_vdbmat_optical_mapping(path, config)
    return MappingEmission(path=path, digest=config.digest, required=True)


def _supplied_materials(
    mapping: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if mapping is None:
        return {}
    raw = mapping.get("materials", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigError("mapping.materials must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConfigError(f"mapping.materials[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"mapping.materials[{index}].name must be a string")
        if name in result:
            raise ConfigError(f"mapping.materials contains duplicate name {name!r}")
        result[name] = item
    return result


def _triple(raw: Mapping[str, Any], field: str) -> tuple[float, float, float]:
    value = raw[field]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{field} must be an array of 3 numbers")
    if len(value) != 3:
        raise ConfigError(f"{field} must contain 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))
