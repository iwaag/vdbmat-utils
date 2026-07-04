"""Integration smoke test: the pinned vdbmat CLI is invocable from this environment."""

import subprocess
import sys

import pytest


@pytest.mark.integration
def test_vdbmat_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vdbmat.cli.main", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "import-voxels" in result.stdout
