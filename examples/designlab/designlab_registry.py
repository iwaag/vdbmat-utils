"""Generator-method registry: the plug point for designlab's form GUI.

A :class:`GeneratorMethod` bundles the four things a method needs to be
driven end to end: its config schema (an existing ``vdbmat-utils``
``GeneratorConfig`` subclass — no GUI-only schema is invented), a form
builder/binder pair, and the CLI argv that actually generates its output.
``form_to_config``/``config_to_form`` operate on a *binding*: any object
whose attributes each expose a mutable ``.value`` (a real viser
``GuiInputHandle`` in the running app, or a plain test double in unit
tests), so this module has no dependency on viser at import time — only
``build_form`` touches viser, and only when called.

The registry itself is a module-level tuple; adding a method is "write one
``GeneratorMethod`` and append it here" (see ``docs/designlab.md``).
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from vdbmat_utils.core import GeneratorConfig
from vdbmat_utils.primitives import PrimitiveArrayConfig
from vdbmat_utils.primitives.types import BUILTIN_MATERIAL_IDS


class _HasValue(Protocol):
    value: Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorMethod:
    """One registered generation method.

    ``config_suffix`` doubles as the catalog discriminator: config files
    under ``--config-root`` are matched to a method by filename suffix
    (``*.primarray.json`` for primitive array).
    """

    method_id: str
    title: str
    config_suffix: str
    config_cls: type[GeneratorConfig]
    build_form: Callable[[Any], Any]
    form_to_config: Callable[[Any], GeneratorConfig]
    config_to_form: Callable[[Any, GeneratorConfig], None]
    generator_argv: Callable[[Path, Path, str], list[str]]


# --- primitive array -------------------------------------------------------


@dataclasses.dataclass(slots=True)
class PrimitiveArrayFormBinding:
    """Widget handles for one primitive-array form instance.

    Each field holds anything with a settable/readable ``.value`` — a viser
    ``GuiInputHandle`` when built by :func:`_build_primitive_array_form`, or
    a plain test double in browser-free unit tests.
    """

    voxel_size_x: _HasValue
    voxel_size_y: _HasValue
    voxel_size_z: _HasValue
    primitive: _HasValue
    counts_x: _HasValue
    counts_y: _HasValue
    counts_z: _HasValue
    primitive_size_m: _HasValue
    gap_m: _HasValue
    margin_m: _HasValue
    base_material_name: _HasValue
    inclusion_material_name: _HasValue
    max_axis_cells: _HasValue
    max_total_cells: _HasValue
    seed: _HasValue


def _build_primitive_array_form(gui: Any) -> PrimitiveArrayFormBinding:
    """Build the primitive-array form under ``gui`` (a viser GUI container).

    Imported lazily so this module (and the registry it defines) has no
    hard dependency on viser being installed; only the running app (which
    does depend on it, via the ``designlab`` dependency group) ever calls
    this.
    """
    material_names = sorted(BUILTIN_MATERIAL_IDS)
    defaults = PrimitiveArrayConfig(
        voxel_size_xyz_m=(0.0001, 0.0001, 0.0001),
        primitive="cube",
        counts_xyz=(2, 2, 2),
        primitive_size_m=0.0004,
        gap_m=0.0002,
        margin_m=0.0001,
    )

    vsize = defaults.voxel_size_xyz_m
    voxel_size_x = gui.add_number("voxel_size_x_m", initial_value=vsize[0])
    voxel_size_y = gui.add_number("voxel_size_y_m", initial_value=vsize[1])
    voxel_size_z = gui.add_number("voxel_size_z_m", initial_value=vsize[2])
    primitive = gui.add_dropdown(
        "primitive", options=("cube", "sphere"), initial_value=defaults.primitive
    )
    counts = defaults.counts_xyz
    counts_x = gui.add_number("counts_x", initial_value=counts[0], step=1)
    counts_y = gui.add_number("counts_y", initial_value=counts[1], step=1)
    counts_z = gui.add_number("counts_z", initial_value=counts[2], step=1)
    primitive_size_m = gui.add_number(
        "primitive_size_m", initial_value=defaults.primitive_size_m
    )
    gap_m = gui.add_number("gap_m", initial_value=defaults.gap_m)
    margin_m = gui.add_number("margin_m", initial_value=defaults.margin_m)
    base_material_name = gui.add_dropdown(
        "base_material_name",
        options=tuple(material_names),
        initial_value=defaults.base_material_name,
    )
    inclusion_material_name = gui.add_dropdown(
        "inclusion_material_name",
        options=tuple(material_names),
        initial_value=defaults.inclusion_material_name,
    )

    advanced = gui.add_folder("Advanced")
    with advanced:
        max_axis_cells = gui.add_number(
            "max_axis_cells", initial_value=defaults.max_axis_cells, step=1
        )
        max_total_cells = gui.add_number(
            "max_total_cells", initial_value=defaults.max_total_cells, step=1
        )
        seed = gui.add_number("seed", initial_value=defaults.seed, step=1)

    return PrimitiveArrayFormBinding(
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        primitive=primitive,
        counts_x=counts_x,
        counts_y=counts_y,
        counts_z=counts_z,
        primitive_size_m=primitive_size_m,
        gap_m=gap_m,
        margin_m=margin_m,
        base_material_name=base_material_name,
        inclusion_material_name=inclusion_material_name,
        max_axis_cells=max_axis_cells,
        max_total_cells=max_total_cells,
        seed=seed,
    )


def _primitive_array_form_to_config(
    binding: PrimitiveArrayFormBinding,
) -> PrimitiveArrayConfig:
    """Read the form into a config. Validation errors propagate untouched.

    ``PrimitiveArrayConfig.__post_init__`` already raises field-named
    ``PrimitiveArrayError``s; this function does not catch or reword them
    (roadmap: don't re-define CLI/config error messages in the GUI layer).
    """
    return PrimitiveArrayConfig(
        voxel_size_xyz_m=(
            binding.voxel_size_x.value,
            binding.voxel_size_y.value,
            binding.voxel_size_z.value,
        ),
        primitive=binding.primitive.value,
        counts_xyz=(
            int(binding.counts_x.value),
            int(binding.counts_y.value),
            int(binding.counts_z.value),
        ),
        primitive_size_m=binding.primitive_size_m.value,
        gap_m=binding.gap_m.value,
        margin_m=binding.margin_m.value,
        base_material_name=binding.base_material_name.value,
        inclusion_material_name=binding.inclusion_material_name.value,
        max_axis_cells=int(binding.max_axis_cells.value),
        max_total_cells=int(binding.max_total_cells.value),
        seed=int(binding.seed.value),
    )


def _primitive_array_config_to_form(
    binding: PrimitiveArrayFormBinding, config: GeneratorConfig
) -> None:
    """Push a loaded config's values into the form (used by config Load)."""
    assert isinstance(config, PrimitiveArrayConfig)
    binding.voxel_size_x.value = config.voxel_size_xyz_m[0]
    binding.voxel_size_y.value = config.voxel_size_xyz_m[1]
    binding.voxel_size_z.value = config.voxel_size_xyz_m[2]
    binding.primitive.value = config.primitive
    binding.counts_x.value = config.counts_xyz[0]
    binding.counts_y.value = config.counts_xyz[1]
    binding.counts_z.value = config.counts_xyz[2]
    binding.primitive_size_m.value = config.primitive_size_m
    binding.gap_m.value = config.gap_m
    binding.margin_m.value = config.margin_m
    binding.base_material_name.value = config.base_material_name
    binding.inclusion_material_name.value = config.inclusion_material_name
    binding.max_axis_cells.value = config.max_axis_cells
    binding.max_total_cells.value = config.max_total_cells
    binding.seed.value = config.seed


def _primitive_array_argv(config_path: Path, out_dir: Path, name: str) -> list[str]:
    """The subprocess argv that generates this method's output.

    ``sys.executable`` pins the current venv's interpreter (roadmap risk:
    "サブプロセスが別 venv で走る事故") rather than relying on a ``PATH``
    lookup of the ``vdbmat-utils`` console script.
    """
    return [
        sys.executable,
        "-m",
        "vdbmat_utils.cli.main",
        "generate-primitive-array",
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
        "--name",
        name,
    ]


PRIMITIVE_ARRAY_METHOD = GeneratorMethod(
    method_id="primitive-array",
    title="Primitive array",
    config_suffix=".primarray.json",
    config_cls=PrimitiveArrayConfig,
    build_form=_build_primitive_array_form,
    form_to_config=_primitive_array_form_to_config,
    config_to_form=_primitive_array_config_to_form,
    generator_argv=_primitive_array_argv,
)

#: All registered generation methods. Add a new method by appending its
#: ``GeneratorMethod`` here (see ``docs/designlab.md`` for the checklist).
REGISTRY: tuple[GeneratorMethod, ...] = (PRIMITIVE_ARRAY_METHOD,)


def method_by_id(method_id: str) -> GeneratorMethod:
    """Look up a registered method by ``method_id``."""
    for method in REGISTRY:
        if method.method_id == method_id:
            return method
    raise KeyError(f"unregistered generator method: {method_id!r}")


def method_for_config_path(path: Path) -> GeneratorMethod | None:
    """Return the registered method whose ``config_suffix`` matches ``path``.

    Returns ``None`` if no registered method's suffix matches, so callers
    building a catalog can silently skip unrelated files.
    """
    name = path.name
    for method in REGISTRY:
        if name.endswith(method.config_suffix):
            return method
    return None


__all__ = [
    "PRIMITIVE_ARRAY_METHOD",
    "REGISTRY",
    "GeneratorMethod",
    "PrimitiveArrayFormBinding",
    "method_by_id",
    "method_for_config_path",
]
