"""Entry point for the ``vdbmat-utils`` command.

Exit codes: 0 success, 1 validation or generation failure, 2 usage error
(argparse default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        raise AssertionError(f"unhandled command {arguments.command!r}")
    except _EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
