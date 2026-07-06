"""Deterministic synthetic fixture volumes.

These presets back the golden-fixture contract tests and the
``vdbmat-utils generate-fixture`` CLI command. Each preset exercises one
metadata risk called out by the roadmap (anisotropic voxels, non-zero origins,
rotations, multiple materials including names outside vdbmat's built-in
optical table) and uses an axis-asymmetric label pattern so z/y/x
transposition errors cannot cancel out.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from .core import GeneratorConfig, build_material_label_volume, build_provenance
from .core.errors import ConfigError
from .image import ImageStackConfig

GENERATOR_NAME = "vdbmat-utils-fixture"
GENERATOR_VERSION = "0.1.0"


@dataclasses.dataclass(frozen=True, slots=True)
class FixtureConfig(GeneratorConfig):
    preset: str = "anisotropic"


def _asymmetric_labels(
    shape_zyx: tuple[int, int, int], material_count: int
) -> npt.NDArray[np.uint16]:
    """Labels varying differently along each axis, so no transpose is a no-op."""
    nz, ny, nx = shape_zyx
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return ((z * 7 + y * 3 + x) % material_count).astype(np.uint16)


def _palette(names: tuple[str, ...]) -> tuple[MaterialDefinition, ...]:
    return tuple(
        MaterialDefinition(
            material_id=i,
            name=name,
            role=MaterialRole.BACKGROUND if i == 0 else MaterialRole.MATERIAL,
        )
        for i, name in enumerate(names)
    )


def _anisotropic(config: FixtureConfig) -> MaterialLabelVolume:
    """Anisotropic voxel size, identity transform, two materials."""
    return build_material_label_volume(
        material_id=_asymmetric_labels((3, 4, 5), 2),
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0004),
        palette=_palette(("void", "resin_clear")),
        provenance=build_provenance(
            generator=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            config=config,
        ),
    )


def _transformed(config: FixtureConfig) -> MaterialLabelVolume:
    """Non-zero origin plus a 90-degree rotation about the world z axis."""
    local_to_world = (
        (0.0, -1.0, 0.0, 0.01),
        (1.0, 0.0, 0.0, -0.02),
        (0.0, 0.0, 1.0, 0.005),
        (0.0, 0.0, 0.0, 1.0),
    )
    return build_material_label_volume(
        material_id=_asymmetric_labels((4, 3, 2), 2),
        voxel_size_xyz_m=(0.0002, 0.0002, 0.0002),
        palette=_palette(("void", "resin_white")),
        provenance=build_provenance(
            generator=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            config=config,
        ),
        local_to_world=local_to_world,
    )


def _multimaterial(config: FixtureConfig) -> MaterialLabelVolume:
    """Four materials; ``quartz_vein`` is outside vdbmat's built-in optical
    table, so downstream optical conversion of this fixture requires an
    external ``vdbmat.optical-mapping`` document (Phase 3 scope)."""
    return build_material_label_volume(
        material_id=_asymmetric_labels((5, 4, 3), 4),
        voxel_size_xyz_m=(0.0001, 0.0001, 0.0003),
        palette=_palette(("void", "resin_clear", "resin_white", "quartz_vein")),
        provenance=build_provenance(
            generator=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            config=config,
        ),
    )


_PRESETS: dict[str, Callable[[FixtureConfig], MaterialLabelVolume]] = {
    "anisotropic": _anisotropic,
    "transformed": _transformed,
    "multimaterial": _multimaterial,
}

FIXTURE_PRESETS = tuple(sorted(_PRESETS))


# Image-stack fixture: gray levels chosen so material ids 0..2 stay inside the
# pinned vdbmat builtin optical mapping, keeping `vdbmat convert` runnable.
_STACK_LEVELS: tuple[dict[str, object], ...] = (
    {"gray": 0, "material_id": 0, "name": "air", "role": "background"},
    {"gray": 100, "material_id": 1, "name": "transparent-resin", "role": "material"},
    {"gray": 255, "material_id": 2, "name": "white-resin", "role": "material"},
)
_STACK_GRAYS = (0, 100, 255)
_STACK_SHAPE_ZYX = (3, 4, 5)


def write_image_stack_fixture(directory: Path) -> tuple[Path, ImageStackConfig]:
    """Write a deterministic labeled PGM stack; return (slices_dir, config).

    Three materials (one background), axis-asymmetric pattern
    ``(7z + 3y + x) % 3`` — the same family as the volume presets, so no
    z/y/x transposition error can cancel out.
    """
    directory.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = _STACK_SHAPE_ZYX
    labels = _asymmetric_labels(_STACK_SHAPE_ZYX, len(_STACK_GRAYS))
    grays = np.asarray(_STACK_GRAYS, dtype=np.uint8)[labels]
    for z in range(nz):
        pixels = grays[z]
        header = f"P5\n{nx} {ny}\n255\n".encode("ascii")
        (directory / f"slice_{z:04d}.pgm").write_bytes(header + pixels.tobytes())
    config = ImageStackConfig(
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0003),
        levels=_STACK_LEVELS,
    )
    return directory, config


def build_fixture(preset: str, *, seed: int = 0) -> MaterialLabelVolume:
    """Build a named fixture volume deterministically."""
    if preset not in _PRESETS:
        known = ", ".join(FIXTURE_PRESETS)
        raise ConfigError(f"unknown fixture preset {preset!r}; expected one of {known}")
    return _PRESETS[preset](FixtureConfig(seed=seed, preset=preset))
