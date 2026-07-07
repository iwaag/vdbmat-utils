"""Deterministic synthetic fixture volumes.

These presets back the golden-fixture contract tests and the
``vdbmat-utils generate-fixture`` CLI command. Each preset exercises one
metadata risk called out by the roadmap (anisotropic voxels, non-zero origins,
rotations, multiple materials including names outside vdbmat's built-in
optical table) and uses an axis-asymmetric label pattern so z/y/x
transposition errors cannot cancel out.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from .core import GeneratorConfig, build_material_label_volume, build_provenance
from .core.errors import ConfigError
from .image import ImageStackConfig
from .io import write_asset
from .mesh import MeshVoxelizeConfig
from .morph import MorphStackConfig

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


# Mesh fixtures: watertight ASCII STL solids emitted in-code (no binary blobs
# in git). Both are prism extrusions of a CCW polygon; the L-bracket footprint
# is asymmetric in x/y and the extrusion height differs from both, so no axis
# transposition in the voxelizer can go unnoticed. Coordinates are millimetres
# (`source_unit="mm"` in the fixture config). Material ids stay inside the
# pinned vdbmat builtin optical mapping so `vdbmat convert` remains runnable.

_CUBE_SIZE_MM = 2.0
_L_BRACKET_POLYGON_MM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (3.0, 0.0),
    (3.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (0.0, 2.0),
)
# Fan from the reflex vertex (1, 1); every polygon edge appears in exactly one
# cap triangle, so extrusion yields a watertight mesh with no T-junctions.
_L_BRACKET_CAP_TRIANGLES = ((3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2))
_L_BRACKET_HEIGHT_MM = 1.0


def _extrude_polygon(
    polygon_xy: tuple[tuple[float, float], ...],
    cap_triangles: tuple[tuple[int, int, int], ...],
    height: float,
) -> npt.NDArray[np.float64]:
    """Extrude a CCW simple polygon along +z into outward-oriented triangles."""
    z0, z1 = 0.0, height
    triangles: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    for a, b, c in cap_triangles:
        (ax, ay), (bx, by), (cx, cy) = (
            polygon_xy[a],
            polygon_xy[b],
            polygon_xy[c],
        )
        # Bottom cap winds clockwise seen from +z (outward normal -z).
        triangles.append(((ax, ay, z0), (cx, cy, z0), (bx, by, z0)))
        triangles.append(((ax, ay, z1), (bx, by, z1), (cx, cy, z1)))
    count = len(polygon_xy)
    for i in range(count):
        (x0, y0) = polygon_xy[i]
        (x1, y1) = polygon_xy[(i + 1) % count]
        base_a = (x0, y0, z0)
        base_b = (x1, y1, z0)
        top_b = (x1, y1, z1)
        top_a = (x0, y0, z1)
        triangles.append((base_a, base_b, top_b))
        triangles.append((base_a, top_b, top_a))
    return np.asarray(triangles, dtype=np.float64)


def _stl_ascii(triangles: npt.NDArray[np.float64], name: str) -> bytes:
    lines = [f"solid {name}"]
    for triangle in triangles:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(
                f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return ("\n".join(lines) + "\n").encode("ascii")


def cube_stl_bytes() -> bytes:
    """A 2 mm axis-aligned cube at the origin, as ASCII STL bytes."""
    square = (
        (0.0, 0.0),
        (_CUBE_SIZE_MM, 0.0),
        (_CUBE_SIZE_MM, _CUBE_SIZE_MM),
        (0.0, _CUBE_SIZE_MM),
    )
    return _stl_ascii(
        _extrude_polygon(square, ((0, 1, 2), (0, 2, 3)), _CUBE_SIZE_MM), "cube"
    )


def l_bracket_stl_bytes() -> bytes:
    """An asymmetric 3x2x1 mm L-bracket prism, as ASCII STL bytes."""
    return _stl_ascii(
        _extrude_polygon(
            _L_BRACKET_POLYGON_MM, _L_BRACKET_CAP_TRIANGLES, _L_BRACKET_HEIGHT_MM
        ),
        "l_bracket",
    )


def write_mesh_fixture(directory: Path) -> tuple[Path, MeshVoxelizeConfig]:
    """Write the L-bracket STL fixture; return (mesh_path, config).

    0.5 mm voxels over the 3x2x1 mm bracket with the default one-cell padding
    give an 8x6x4 (x, y, z) grid — small enough for every test tier.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mesh_path = directory / "l_bracket.stl"
    mesh_path.write_bytes(l_bracket_stl_bytes())
    config = MeshVoxelizeConfig(
        source_unit="mm",
        voxel_size_xyz_m=(0.0005, 0.0005, 0.0005),
        material={
            "material_id": 1,
            "name": "transparent-resin",
            "role": "material",
        },
    )
    return mesh_path, config


# Morph fixture (Phase 2): a sparse key-slice set with a merge event. Gray
# levels and material names match the image-stack fixture so material ids
# stay inside the pinned vdbmat builtin optical mapping (`vdbmat convert`
# stays runnable). Slices are 10 (y) x 12 (x); keys at z = 0, 3, 7 with
# interpolated gaps between them.

_MORPH_KEY_INDICES = (0, 3, 7)
_MORPH_SHAPE_YX = (10, 12)


def _morph_key_pixels(z_index: int) -> npt.NDArray[np.uint8]:
    pixels = np.zeros(_MORPH_SHAPE_YX, dtype=np.uint8)
    if z_index == 0:
        pixels[1:5, 1:5] = 100  # two separated transparent-resin squares...
        pixels[1:5, 7:11] = 100
        pixels[6:9, 1:6] = 255  # ...and an asymmetric white-resin block
    elif z_index == 3:
        pixels[1:5, 1:6] = 100  # squares grown toward each other, still apart
        pixels[1:5, 7:11] = 100
        pixels[6:9, 2:7] = 255
    else:
        pixels[1:5, 1:11] = 100  # merged into one bar (topology change)
        pixels[6:9, 4:9] = 255
    return pixels


def write_morph_fixture(directory: Path) -> tuple[Path, MorphStackConfig]:
    """Write a deterministic sparse key-slice set; return (slices_dir, config).

    Three materials (one background), asymmetric shapes, one merge event
    (two squares → one bar between z=3 and z=7); interior gaps are
    interpolated by ``morph-stack``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ny, nx = _MORPH_SHAPE_YX
    for z_index in _MORPH_KEY_INDICES:
        pixels = _morph_key_pixels(z_index)
        header = f"P5\n{nx} {ny}\n255\n".encode("ascii")
        (directory / f"slice_{z_index:04d}.pgm").write_bytes(
            header + pixels.tobytes()
        )
    config = MorphStackConfig(
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0003),
        levels=_STACK_LEVELS,
    )
    return directory, config


def write_pipeline_fixture(directory: Path) -> Path:
    """Write a two-asset boolean-composition scenario; return the config path.

    Both assets share the Phase 1 fixture family's axis-asymmetric pattern
    and exact geometry; material names stay inside the pinned vdbmat builtin
    optical mapping (``air``/``transparent-resin``/``white-resin``) so
    ``vdbmat convert`` remains runnable on the composed output (the Phase 1
    volume presets use non-builtin names and cannot be reused here). The
    pipeline remaps the overlay's material onto a fresh id renamed
    ``white-resin`` and composes with ``union``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = build_material_label_volume(
        material_id=_asymmetric_labels((3, 4, 5), 2),
        voxel_size_xyz_m=(0.0001, 0.0002, 0.0004),
        palette=_palette(("air", "transparent-resin")),
        provenance=build_provenance(
            generator=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            config=FixtureConfig(preset="pipeline-base"),
        ),
    )
    write_asset(base, directory, "base")

    overlay_labels = np.zeros(base.geometry.shape_zyx, dtype=np.uint16)
    overlay_labels[1:3, 1:3, 1:4] = 1
    overlay = build_material_label_volume(
        material_id=overlay_labels,
        voxel_size_xyz_m=base.geometry.voxel_size_xyz_m,
        palette=_palette(("air", "transparent-resin")),
        provenance=build_provenance(
            generator=GENERATOR_NAME,
            generator_version=GENERATOR_VERSION,
            config=FixtureConfig(preset="pipeline-overlay"),
        ),
    )
    write_asset(overlay, directory, "overlay")

    payload = {
        "inputs": [
            {"id": "base", "manifest_path": "base.voxels.json"},
            {"id": "overlay", "manifest_path": "overlay.voxels.json"},
        ],
        "steps": [
            {
                "op": "remap-materials",
                "from": "overlay",
                "mapping": {"1": 2},
                "definitions": {"2": {"name": "white-resin"}},
                "as": "overlay_white",
            },
            {
                "op": "compose",
                "base": "base",
                "overlay": "overlay_white",
                "mode": "union",
                "as": "composed",
            },
        ],
        "output": {"ref": "composed"},
    }
    config_path = directory / "pipeline.json"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config_path


def build_fixture(preset: str, *, seed: int = 0) -> MaterialLabelVolume:
    """Build a named fixture volume deterministically."""
    if preset not in _PRESETS:
        known = ", ".join(FIXTURE_PRESETS)
        raise ConfigError(f"unknown fixture preset {preset!r}; expected one of {known}")
    return _PRESETS[preset](FixtureConfig(seed=seed, preset=preset))
