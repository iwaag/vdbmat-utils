"""Entry point for the ``vdbmat-utils`` command.

Phase 0 Step 6 adds ``inspect``, ``validate``, and ``generate-fixture``;
until then this stub only reports the package version.
"""

from __future__ import annotations

import argparse

from vdbmat_utils import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdbmat-utils",
        description="Inspect, validate, and generate vdbmat voxel assets.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
