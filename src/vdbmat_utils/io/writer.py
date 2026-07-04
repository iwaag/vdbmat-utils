"""Asset writer delegating to the shared vdbmat manifest emitter.

Determinism contract: see ``docs/determinism.md``. This module adds no
timestamps, hostnames, or absolute paths of its own, and the underlying
``vdbmat`` writer records none, so repeated writes of the same volume are
byte-equal.
"""

from pathlib import Path

from vdbmat.core import MaterialLabelVolume
from vdbmat.io import write_material_label_manifest


def write_asset(
    volume: MaterialLabelVolume,
    directory: str | Path,
    name: str,
    *,
    identity: str | None = None,
) -> Path:
    """Write ``volume`` as ``<name>.voxels.json`` + ``<name>.material_id.npy``.

    Returns the manifest path. ``identity`` optionally records the
    source-data identity in the manifest's ``source`` block.
    """
    return write_material_label_manifest(directory, name, volume, identity=identity)
