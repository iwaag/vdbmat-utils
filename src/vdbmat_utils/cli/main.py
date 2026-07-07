"""Entry point for the ``vdbmat-utils`` command.

Exit codes: 0 success, 1 validation or generation failure, 2 usage error
(argparse default).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Literal, cast

from vdbmat.core import MaterialLabelVolume, VolumeValidationError
from vdbmat.io import (
    VolumeIOError,
    VoxelManifestError,
    inspect_material_label_manifest,
    read_material_label_manifest,
)

from vdbmat_utils import __version__
from vdbmat_utils.core import VdbmatUtilsError, require_compatible_volume_schema
from vdbmat_utils.core.provenance import provenance_identity
from vdbmat_utils.fixtures import FIXTURE_PRESETS, build_fixture
from vdbmat_utils.image import (
    ImageStackConfig,
    convert_image_stack,
    stack_identity,
)
from vdbmat_utils.io import write_asset
from vdbmat_utils.mesh import (
    MeshVoxelizeConfig,
    load_mesh,
    voxelize_mesh,
)
from vdbmat_utils.mesh.voxelizer import SUPPORTED_MESH_UNITS
from vdbmat_utils.morph import MorphStackConfig, morph_stack
from vdbmat_utils.pipeline import PipelineConfig, run_pipeline, validate_pipeline
from vdbmat_utils.preview import material_counts, slice_ascii, slice_pgm

_EXPECTED_ERRORS = (
    VdbmatUtilsError,
    VoxelManifestError,
    VolumeIOError,
    VolumeValidationError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdbmat-utils",
        description="Inspect, validate, and generate vdbmat voxel assets.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect", help="show manifest metadata without loading the payload"
    )
    inspect_parser.add_argument("manifest", type=Path, metavar="MANIFEST")
    inspect_parser.add_argument("--json", action="store_true", dest="json_output")

    validate_parser = commands.add_parser(
        "validate", help="fully read and validate an asset against the contract"
    )
    validate_parser.add_argument("manifest", type=Path, metavar="MANIFEST")

    fixture_parser = commands.add_parser(
        "generate-fixture", help="write a deterministic synthetic fixture asset"
    )
    fixture_parser.add_argument(
        "preset", choices=FIXTURE_PRESETS, metavar="PRESET",
        help=f"one of: {', '.join(FIXTURE_PRESETS)}",
    )
    fixture_parser.add_argument(
        "-o", "--output", type=Path, required=True, metavar="DIR",
        help="output directory for the manifest and payload",
    )
    fixture_parser.add_argument("--seed", type=int, default=0)

    stack_parser = commands.add_parser(
        "convert-image-stack",
        help="stack labeled 2D slices into a vdbmat.voxels asset",
    )
    stack_parser.add_argument("slices_dir", type=Path, metavar="SLICES_DIR")
    stack_parser.add_argument(
        "--config", type=Path, required=True, metavar="CONFIG",
        help="ImageStackConfig JSON (voxel_size_xyz_m, levels, ...)",
    )
    stack_parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="output directory for the manifest and payload",
    )
    stack_parser.add_argument("--name", required=True, metavar="NAME")
    stack_parser.add_argument(
        "--voxel-size", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"), help="override config voxel_size_xyz_m",
    )
    stack_parser.add_argument(
        "--format", choices=("pgm", "png"), default=None, dest="image_format",
        help="override config slice format",
    )

    morph_parser = commands.add_parser(
        "morph-stack",
        help="interpolate sparse labeled key slices into a vdbmat.voxels asset",
    )
    morph_parser.add_argument("slices_dir", type=Path, metavar="SLICES_DIR")
    morph_parser.add_argument(
        "--config", type=Path, required=True, metavar="CONFIG",
        help="MorphStackConfig JSON (voxel_size_xyz_m, levels, z_count, ...)",
    )
    morph_parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="output directory for the manifest and payload",
    )
    morph_parser.add_argument("--name", required=True, metavar="NAME")
    morph_parser.add_argument(
        "--voxel-size", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"), help="override config voxel_size_xyz_m",
    )
    morph_parser.add_argument(
        "--z-count", type=int, default=None, dest="z_count",
        help="override config z_count (total output depth)",
    )

    pipeline_parser = commands.add_parser(
        "apply-pipeline",
        help="apply a configured op sequence to existing vdbmat.voxels assets",
    )
    pipeline_parser.add_argument(
        "--config", type=Path, required=True, metavar="PIPELINE",
        help="PipelineConfig JSON (inputs, steps, output)",
    )
    pipeline_parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="output directory for the manifest and payload",
    )
    pipeline_parser.add_argument("--name", required=True, metavar="NAME")
    pipeline_parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="validate and print the resolved step plan without running",
    )

    mesh_parser = commands.add_parser(
        "voxelize-mesh",
        help="voxelize one watertight STL into a vdbmat.voxels asset",
    )
    mesh_parser.add_argument("mesh", type=Path, metavar="MESH")
    mesh_parser.add_argument(
        "--config", type=Path, required=True, metavar="CONFIG",
        help="MeshVoxelizeConfig JSON (source_unit, voxel_size_xyz_m, material, ...)",
    )
    mesh_parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="output directory for the manifest and payload",
    )
    mesh_parser.add_argument("--name", required=True, metavar="NAME")
    mesh_parser.add_argument(
        "--source-unit", choices=SUPPORTED_MESH_UNITS, default=None,
        dest="source_unit", help="override config source_unit",
    )
    mesh_parser.add_argument(
        "--voxel-size", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"), help="override config voxel_size_xyz_m",
    )
    mesh_parser.add_argument(
        "--material-id", type=int, default=None, dest="material_id",
        help="override config material.material_id",
    )
    mesh_parser.add_argument(
        "--material-name", default=None, dest="material_name",
        help="override config material.name",
    )
    mesh_parser.add_argument(
        "--padding", type=int, default=None, dest="padding_cells",
        help="override config padding_cells",
    )

    counts_parser = commands.add_parser(
        "material-counts", help="print voxel counts per material id"
    )
    counts_parser.add_argument("manifest", type=Path, metavar="MANIFEST")
    counts_parser.add_argument("--json", action="store_true", dest="json_output")

    preview_parser = commands.add_parser(
        "preview-slices", help="render one slice as ASCII text or a PGM image"
    )
    preview_parser.add_argument("manifest", type=Path, metavar="MANIFEST")
    preview_parser.add_argument(
        "--axis", choices=("z", "y", "x"), default="z",
        help="axis perpendicular to the slice (default: z)",
    )
    preview_parser.add_argument(
        "--index", type=int, default=None,
        help="slice index along the axis (default: middle slice)",
    )
    preview_parser.add_argument(
        "--out", type=Path, default=None, metavar="FILE.pgm",
        help="write a grayscale PGM instead of printing ASCII to stdout",
    )

    return parser


def _cmd_inspect(manifest: Path, *, json_output: bool) -> int:
    inspection = inspect_material_label_manifest(manifest)
    fields = {
        "format_version": str(inspection.format_version),
        "shape_zyx": list(inspection.shape_zyx),
        "voxel_size_xyz_m": list(inspection.voxel_size_xyz_m),
        "material_ids": list(inspection.material_ids),
        "payload_path": inspection.payload_path,
        "payload_sha256": inspection.payload_sha256,
        "source_identity": inspection.source_identity,
    }
    if json_output:
        print(json.dumps(fields, indent=2))
    else:
        for key, value in fields.items():
            print(f"{key}: {value}")
    return 0


def _cmd_validate(manifest: Path) -> int:
    require_compatible_volume_schema()
    volume = read_material_label_manifest(manifest)
    print(
        f"OK: {manifest} is a valid vdbmat.voxels asset "
        f"(shape_zyx={volume.geometry.shape_zyx}, "
        f"materials={len(volume.palette)})"
    )
    return 0


def _print_material_counts(volume: MaterialLabelVolume) -> None:
    counts = material_counts(volume)
    for material in volume.palette:
        print(
            f"{material.material_id} ({material.name}, {material.role}): "
            f"{counts[material.material_id]}"
        )


def _cmd_material_counts(manifest: Path, *, json_output: bool) -> int:
    volume = read_material_label_manifest(manifest)
    if json_output:
        counts = material_counts(volume)
        print(json.dumps({str(k): v for k, v in counts.items()}, indent=2))
    else:
        _print_material_counts(volume)
    return 0


def _cmd_convert_image_stack(
    slices_dir: Path,
    config_path: Path,
    out: Path,
    name: str,
    voxel_size: list[float] | None,
    image_format: str | None,
) -> int:
    config = ImageStackConfig.from_json(config_path.read_text(encoding="utf-8"))
    if voxel_size is not None:
        config = dataclasses.replace(
            config,
            voxel_size_xyz_m=(voxel_size[0], voxel_size[1], voxel_size[2]),
        )
    if image_format is not None:
        config = dataclasses.replace(config, format=image_format)
    volume = convert_image_stack(slices_dir, config)
    manifest = write_asset(volume, out, name, identity=stack_identity(volume))
    print(f"wrote {manifest}")
    _print_material_counts(volume)
    return 0


def _cmd_morph_stack(
    slices_dir: Path,
    config_path: Path,
    out: Path,
    name: str,
    voxel_size: list[float] | None,
    z_count: int | None,
) -> int:
    config = MorphStackConfig.from_json(config_path.read_text(encoding="utf-8"))
    if voxel_size is not None:
        config = dataclasses.replace(
            config,
            voxel_size_xyz_m=(voxel_size[0], voxel_size[1], voxel_size[2]),
        )
    if z_count is not None:
        config = dataclasses.replace(config, z_count=z_count)
    volume = morph_stack(slices_dir, config)
    manifest = write_asset(
        volume, out, name, identity=provenance_identity(volume.provenance)
    )
    print(f"wrote {manifest}")
    _print_material_counts(volume)
    return 0


def _cmd_apply_pipeline(
    config_path: Path, out: Path, name: str, *, dry_run: bool
) -> int:
    config = PipelineConfig.from_json(config_path.read_text(encoding="utf-8"))
    if dry_run:
        for step in validate_pipeline(config):
            print(step.describe())
        print(f"output: {config.output['ref']}")
        return 0
    volume = run_pipeline(config, base_dir=config_path.resolve().parent)
    manifest = write_asset(
        volume, out, name, identity=provenance_identity(volume.provenance)
    )
    print(f"wrote {manifest}")
    _print_material_counts(volume)
    return 0


def _cmd_voxelize_mesh(
    mesh_path: Path,
    config_path: Path,
    out: Path,
    name: str,
    source_unit: str | None,
    voxel_size: list[float] | None,
    material_id: int | None,
    material_name: str | None,
    padding_cells: int | None,
) -> int:
    config = MeshVoxelizeConfig.from_json(config_path.read_text(encoding="utf-8"))
    if source_unit is not None:
        config = dataclasses.replace(config, source_unit=source_unit)
    if voxel_size is not None:
        config = dataclasses.replace(
            config,
            voxel_size_xyz_m=(voxel_size[0], voxel_size[1], voxel_size[2]),
        )
    if material_id is not None or material_name is not None:
        material = dict(config.material)
        if material_id is not None:
            material["material_id"] = material_id
        if material_name is not None:
            material["name"] = material_name
        config = dataclasses.replace(config, material=material)
    if padding_cells is not None:
        config = dataclasses.replace(config, padding_cells=padding_cells)

    mesh = load_mesh(mesh_path)
    result = voxelize_mesh(mesh, config)
    identity = (
        f"sha256:{mesh.source_sha256}" if mesh.source_sha256 is not None else None
    )
    manifest = write_asset(result.volume, out, name, identity=identity)
    print(f"wrote {manifest}")
    diagnostics = result.diagnostics
    print(
        f"shape_zyx: {diagnostics.shape_zyx} "
        f"(triangles={diagnostics.triangle_count}, "
        f"occupied={diagnostics.occupied_cells})"
    )
    _print_material_counts(result.volume)
    return 0


def _cmd_preview_slices(
    manifest: Path, axis: str, index: int | None, out: Path | None
) -> int:
    volume = read_material_label_manifest(manifest)
    plane_axis = cast(Literal["z", "y", "x"], axis)
    if index is None:
        extent = volume.geometry.shape_zyx[("z", "y", "x").index(plane_axis)]
        index = extent // 2
    if out is None:
        print(slice_ascii(volume, plane_axis, index))
    else:
        print(f"wrote {slice_pgm(volume, plane_axis, index, out)}")
    return 0


def _cmd_generate_fixture(preset: str, output: Path, seed: int) -> int:
    manifest_path = write_asset(build_fixture(preset, seed=seed), output, preset)
    print(f"wrote {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "inspect":
            return _cmd_inspect(arguments.manifest, json_output=arguments.json_output)
        if arguments.command == "validate":
            return _cmd_validate(arguments.manifest)
        if arguments.command == "generate-fixture":
            return _cmd_generate_fixture(
                arguments.preset, arguments.output, arguments.seed
            )
        if arguments.command == "convert-image-stack":
            return _cmd_convert_image_stack(
                arguments.slices_dir,
                arguments.config,
                arguments.out,
                arguments.name,
                arguments.voxel_size,
                arguments.image_format,
            )
        if arguments.command == "morph-stack":
            return _cmd_morph_stack(
                arguments.slices_dir,
                arguments.config,
                arguments.out,
                arguments.name,
                arguments.voxel_size,
                arguments.z_count,
            )
        if arguments.command == "apply-pipeline":
            return _cmd_apply_pipeline(
                arguments.config,
                arguments.out,
                arguments.name,
                dry_run=arguments.dry_run,
            )
        if arguments.command == "voxelize-mesh":
            return _cmd_voxelize_mesh(
                arguments.mesh,
                arguments.config,
                arguments.out,
                arguments.name,
                arguments.source_unit,
                arguments.voxel_size,
                arguments.material_id,
                arguments.material_name,
                arguments.padding_cells,
            )
        if arguments.command == "material-counts":
            return _cmd_material_counts(
                arguments.manifest, json_output=arguments.json_output
            )
        if arguments.command == "preview-slices":
            return _cmd_preview_slices(
                arguments.manifest, arguments.axis, arguments.index, arguments.out
            )
        raise AssertionError(f"unhandled command {arguments.command!r}")
    except _EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
