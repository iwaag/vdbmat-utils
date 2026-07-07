"""Pipeline configuration, validation, and execution (plan D6).

Validation is complete before any array work: unknown ops, unknown or
missing parameters, unbound or rebound ids, and unused inputs all fail fast
with the step index. Execution loads inputs through the public ``vdbmat``
manifest reader, applies the registered ops, and stamps the result with
pipeline provenance (input manifest digests as ``sources``).
"""

import dataclasses
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vdbmat.core import MaterialLabelVolume
from vdbmat.io import read_material_label_manifest

from vdbmat_utils.core import (
    GeneratorConfig,
    VdbmatUtilsError,
    build_material_label_volume,
    build_provenance,
)
from vdbmat_utils.pipeline import PipelineError
from vdbmat_utils.pipeline.registry import REGISTRY, OpSpec

GENERATOR = "vdbmat-utils.pipeline"
GENERATOR_VERSION = "0.1.0"

_INPUT_KEYS = {"id", "manifest_path"}
_STEP_STRUCTURAL_KEYS = {"op", "as"}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PipelineConfig(GeneratorConfig):
    """Configuration for the apply-pipeline workflow.

    ``inputs`` binds asset ids to manifest paths (resolved relative to the
    config file's directory); ``steps`` is a flat SSA-style op list; each
    step reads bound ids (``"from"``, or ``"base"``/``"overlay"``/``"mask"``)
    and binds its result to a fresh id via ``"as"``; ``output`` names the id
    to write as ``{"ref": id}``. ``seed`` is inherited, unused, and reserved.
    """

    inputs: tuple[Mapping[str, object], ...]
    steps: tuple[Mapping[str, object], ...]
    output: Mapping[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedStep:
    """One validated step: its registry spec, volume references, parameters,
    and the id it binds."""

    index: int
    spec: OpSpec
    reads: Mapping[str, str]  # volume key -> bound id
    params: Mapping[str, Any]
    binds: str

    def describe(self) -> str:
        reads = ", ".join(f"{key}={ref}" for key, ref in self.reads.items())
        return f"step {self.index}: {self.spec.name}({reads}) -> {self.binds}"


def _input_ids(config: PipelineConfig) -> dict[str, str]:
    """Validate ``inputs`` and return id -> manifest path (as written)."""
    if not config.inputs:
        raise PipelineError("config.inputs: must be a non-empty array")
    inputs: dict[str, str] = {}
    for index, entry in enumerate(config.inputs):
        field = f"config.inputs[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != _INPUT_KEYS:
            raise PipelineError(
                f"{field}: must be an object with exactly the keys "
                f"{sorted(_INPUT_KEYS)}"
            )
        ref = entry["id"]
        path = entry["manifest_path"]
        if not isinstance(ref, str) or not ref:
            raise PipelineError(f"{field}.id: must be a non-empty string")
        if not isinstance(path, str) or not path:
            raise PipelineError(f"{field}.manifest_path: must be a non-empty string")
        if ref in inputs:
            raise PipelineError(f"{field}.id: duplicate id {ref!r}")
        inputs[ref] = path
    return inputs


def validate_pipeline(config: PipelineConfig) -> tuple[ResolvedStep, ...]:
    """Structurally validate ``config`` and return the resolved step plan.

    Fails fast — before any manifest is read or array touched — naming the
    offending step index.
    """
    inputs = _input_ids(config)
    if not config.steps:
        raise PipelineError("config.steps: must be a non-empty array")

    bound = set(inputs)
    referenced: set[str] = set()
    plan: list[ResolvedStep] = []
    for index, step in enumerate(config.steps):
        field = f"config.steps[{index}]"
        if not isinstance(step, Mapping):
            raise PipelineError(f"{field}: must be an object")
        op = step.get("op")
        if not isinstance(op, str):
            raise PipelineError(f"{field}.op: must be a string")
        spec = REGISTRY.get(op)
        if spec is None:
            raise PipelineError(
                f"{field}.op: unknown op {op!r}; known ops: "
                f"{', '.join(sorted(REGISTRY))}"
            )
        unknown = sorted(
            set(step) - _STEP_STRUCTURAL_KEYS - set(spec.volume_keys)
            - spec.parameter_keys
        )
        if unknown:
            raise PipelineError(
                f"{field}: unknown parameters for op {op!r}: {', '.join(unknown)}"
            )
        reads: dict[str, str] = {}
        for key in spec.volume_keys:
            ref = step.get(key)
            if not isinstance(ref, str) or not ref:
                raise PipelineError(
                    f"{field}.{key}: op {op!r} requires a volume id string"
                )
            if ref not in bound:
                raise PipelineError(
                    f"{field}.{key}: id {ref!r} is not bound by any input or "
                    "earlier step"
                )
            reads[key] = ref
            referenced.add(ref)
        binds = step.get("as")
        if not isinstance(binds, str) or not binds:
            raise PipelineError(f"{field}.as: must be a non-empty string id")
        if binds in bound:
            raise PipelineError(
                f"{field}.as: id {binds!r} is already bound; pipeline ids are "
                "single-assignment"
            )
        bound.add(binds)
        params = {
            key: value for key, value in step.items() if key in spec.parameter_keys
        }
        plan.append(
            ResolvedStep(
                index=index, spec=spec, reads=reads, params=params, binds=binds
            )
        )

    if not isinstance(config.output, Mapping) or set(config.output) != {"ref"}:
        raise PipelineError(
            'config.output: must be an object with exactly the key "ref"'
        )
    output_ref = config.output["ref"]
    if not isinstance(output_ref, str) or output_ref not in bound:
        raise PipelineError(
            f"config.output.ref: id {output_ref!r} is not bound by any input "
            "or step"
        )
    referenced.add(output_ref)
    unused = sorted(set(inputs) - referenced)
    if unused:
        raise PipelineError(
            f"config.inputs: input id(s) {', '.join(unused)} are never used"
        )
    return tuple(plan)


def run_pipeline(config: PipelineConfig, *, base_dir: Path) -> MaterialLabelVolume:
    """Validate and execute ``config``; return the output volume.

    Relative ``manifest_path`` entries resolve against ``base_dir`` (the
    config file's directory). The result carries pipeline provenance: the
    configuration digest and the SHA-256 of every input manifest file, in
    input order.
    """
    plan = validate_pipeline(config)
    inputs = _input_ids(config)

    volumes: dict[str, MaterialLabelVolume] = {}
    digests: list[str] = []
    for ref, path_text in inputs.items():
        path = Path(path_text)
        if not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            raise PipelineError(
                f"config.inputs: manifest for id {ref!r} not found at {path}"
            )
        digests.append(f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}")
        volumes[ref] = read_material_label_manifest(path)

    for step in plan:
        try:
            volumes[step.binds] = step.spec.apply(
                {key: volumes[ref] for key, ref in step.reads.items()}, step.params
            )
        except VdbmatUtilsError as error:
            raise PipelineError(
                f"step {step.index} ({step.spec.name}): {error}"
            ) from error

    output_ref = config.output["ref"]
    assert isinstance(output_ref, str)  # validated above
    result = volumes[output_ref]
    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=tuple(digests),
        notes="config-driven volume-operation pipeline",
    )
    return build_material_label_volume(
        material_id=result.material_id,
        voxel_size_xyz_m=result.geometry.voxel_size_xyz_m,
        palette=result.palette,
        provenance=provenance,
        local_to_world=result.geometry.local_to_world,
    )
