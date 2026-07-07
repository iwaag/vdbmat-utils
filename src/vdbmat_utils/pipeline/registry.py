"""Op-name → implementation and parameter-schema registry (plan D6).

Each registered operation declares which step keys name input volumes and
which name JSON parameters, plus an adapter that converts the JSON-shaped
parameters (lists, plain numbers) into the exact keyword types the ``ops``
functions take. Validation errors raised here are wrapped with the step
index by the engine.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.ops import (
    apply_mask,
    compose,
    crop,
    orient,
    pad,
    place,
    remap_materials,
    resample,
)
from vdbmat_utils.pipeline import PipelineError

OpApply = Callable[
    [Mapping[str, MaterialLabelVolume], Mapping[str, Any]], MaterialLabelVolume
]


@dataclasses.dataclass(frozen=True, slots=True)
class OpSpec:
    """One registered operation: input-reference keys, parameter names, and
    an adapter from (volumes, JSON params) to the result volume."""

    name: str
    volume_keys: tuple[str, ...]  # step keys referencing bound volume ids
    parameter_keys: frozenset[str]
    apply: OpApply


def _int_triplet(value: object, *, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise PipelineError(f"{name} must be a list of 3 integers")
    return (int(value[0]), int(value[1]), int(value[2]))


def _float_triplet(value: object, *, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise PipelineError(f"{name} must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _matrix4(value: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PipelineError(f"{name} must be a 4x4 matrix (list of 4 rows)")
    rows = []
    for row in value:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in row
            )
        ):
            raise PipelineError(f"{name} must be a 4x4 matrix of numbers")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _optional_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"{name} must be an integer")
    return int(value)


def _bool(value: object, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise PipelineError(f"{name} must be a boolean")
    return value


def _str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise PipelineError(f"{name} must be a string")
    return value


def _require(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise PipelineError(f"missing required parameter {key!r}")
    return params[key]


def _apply_crop(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    return crop(
        volumes["from"],
        min_zyx=_int_triplet(_require(params, "min_zyx"), name="min_zyx"),
        max_zyx=_int_triplet(_require(params, "max_zyx"), name="max_zyx"),
    )


def _apply_pad(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    return pad(
        volumes["from"],
        before_zyx=_int_triplet(_require(params, "before_zyx"), name="before_zyx"),
        after_zyx=_int_triplet(_require(params, "after_zyx"), name="after_zyx"),
        fill_material_id=_optional_int(
            params.get("fill_material_id"), name="fill_material_id"
        ),
    )


def _apply_resample(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    factor = params.get("factor_zyx")
    size = params.get("voxel_size_xyz_m")
    return resample(
        volumes["from"],
        factor_zyx=(
            None if factor is None else _float_triplet(factor, name="factor_zyx")
        ),
        voxel_size_xyz_m=(
            None if size is None else _float_triplet(size, name="voxel_size_xyz_m")
        ),
    )


def _apply_orient(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    raw_steps = _require(params, "steps")
    if not isinstance(raw_steps, (list, tuple)):
        raise PipelineError("steps must be a list of orient steps")
    steps = []
    for step in raw_steps:
        if not isinstance(step, (list, tuple)) or not all(
            isinstance(item, str) for item in step
        ):
            raise PipelineError("each orient step must be a list of strings")
        steps.append(tuple(step))
    return orient(
        volumes["from"],
        steps=tuple(steps),
        update_transform=_bool(
            params.get("update_transform"), name="update_transform", default=True
        ),
    )


def _apply_place(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    return place(
        volumes["from"],
        local_to_world=_matrix4(
            _require(params, "local_to_world"), name="local_to_world"
        ),
        compose_with_existing=_bool(
            params.get("compose_with_existing"),
            name="compose_with_existing",
            default=False,
        ),
    )


def _apply_apply_mask(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    return apply_mask(
        volumes["from"],
        volumes["mask"],
        mode=_str(params.get("mode", "keep"), name="mode"),
        fill_material_id=_optional_int(
            params.get("fill_material_id"), name="fill_material_id"
        ),
    )


def _apply_compose(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    return compose(
        volumes["base"],
        volumes["overlay"],
        mode=_str(_require(params, "mode"), name="mode"),
    )


def _apply_remap(
    volumes: Mapping[str, MaterialLabelVolume], params: Mapping[str, Any]
) -> MaterialLabelVolume:
    raw_mapping = _require(params, "mapping")
    if not isinstance(raw_mapping, Mapping):
        raise PipelineError("mapping must be an object of old_id -> new_id")
    mapping: dict[int, int] = {}
    for key, value in raw_mapping.items():
        try:
            old_id = int(key)
        except (TypeError, ValueError) as error:
            raise PipelineError(
                f"mapping keys must be integer ids, got {key!r}"
            ) from error
        if isinstance(value, bool) or not isinstance(value, int):
            raise PipelineError(f"mapping[{key}] must be an integer id")
        mapping[old_id] = int(value)
    raw_definitions = params.get("definitions")
    definitions: dict[int, dict[str, str]] | None = None
    if raw_definitions is not None:
        if not isinstance(raw_definitions, Mapping):
            raise PipelineError("definitions must be an object keyed by new id")
        definitions = {}
        for key, override in raw_definitions.items():
            try:
                new_id = int(key)
            except (TypeError, ValueError) as error:
                raise PipelineError(
                    f"definitions keys must be integer ids, got {key!r}"
                ) from error
            if not isinstance(override, Mapping) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in override.items()
            ):
                raise PipelineError(
                    f"definitions[{key}] must be an object of string fields"
                )
            definitions[new_id] = dict(override)
    return remap_materials(
        volumes["from"],
        mapping,
        definitions=definitions,
        prune_palette=_bool(
            params.get("prune_palette"), name="prune_palette", default=True
        ),
    )


REGISTRY: dict[str, OpSpec] = {
    spec.name: spec
    for spec in (
        OpSpec("crop", ("from",), frozenset({"min_zyx", "max_zyx"}), _apply_crop),
        OpSpec(
            "pad",
            ("from",),
            frozenset({"before_zyx", "after_zyx", "fill_material_id"}),
            _apply_pad,
        ),
        OpSpec(
            "resample",
            ("from",),
            frozenset({"factor_zyx", "voxel_size_xyz_m"}),
            _apply_resample,
        ),
        OpSpec(
            "orient", ("from",), frozenset({"steps", "update_transform"}), _apply_orient
        ),
        OpSpec(
            "place",
            ("from",),
            frozenset({"local_to_world", "compose_with_existing"}),
            _apply_place,
        ),
        OpSpec(
            "apply-mask",
            ("from", "mask"),
            frozenset({"mode", "fill_material_id"}),
            _apply_apply_mask,
        ),
        OpSpec("compose", ("base", "overlay"), frozenset({"mode"}), _apply_compose),
        OpSpec(
            "remap-materials",
            ("from",),
            frozenset({"mapping", "definitions", "prune_palette"}),
            _apply_remap,
        ),
    )
}
