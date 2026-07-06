"""Entry point for the ``vdbmat-utils`` command.

Exit codes: 0 success, 1 validation or generation failure, 2 usage error
(argparse default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

from vdbmat.core import VolumeValidationError
from vdbmat.io import (
    VolumeIOError,
    VoxelManifestError,
    inspect_material_label_manifest,
    read_material_label_manifest,
)

from vdbmat_utils import __version__
from vdbmat_utils.core import VdbmatUtilsError, require_compatible_volume_schema
from vdbmat_utils.fixtures import FIXTURE_PRESETS, build_fixture
from vdbmat_utils.io import write_asset
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


def _cmd_material_counts(manifest: Path, *, json_output: bool) -> int:
    volume = read_material_label_manifest(manifest)
    counts = material_counts(volume)
    if json_output:
        print(json.dumps({str(k): v for k, v in counts.items()}, indent=2))
    else:
        for material in volume.palette:
            print(
                f"{material.material_id} ({material.name}, {material.role}): "
                f"{counts[material.material_id]}"
            )
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
