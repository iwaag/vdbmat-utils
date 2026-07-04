"""Smoke tests for the package skeleton and the pinned vdbmat dependency."""

import vdbmat.core

from vdbmat_utils import __version__
from vdbmat_utils.cli.main import main


def test_version_string() -> None:
    assert __version__


def test_cli_runs_without_arguments() -> None:
    assert main([]) == 0


def test_pinned_vdbmat_exposes_public_contract_types() -> None:
    names = ("MaterialLabelVolume", "GridGeometry", "MaterialPalette", "Provenance")
    for name in names:
        assert hasattr(vdbmat.core, name)
