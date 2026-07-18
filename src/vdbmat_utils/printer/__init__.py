"""GrabCAD Print voxel-printing PNG-slice exporter.

Converts a material-label voxel asset (``.voxels.json`` +
``.material_id.npy``) into an indexed-palette PNG slice stack plus a
sidecar manifest, on a printer-pitch grid derived from physical extent.
See ``docs/print-slices.md`` for the config shape and sampling rules.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class PrintSlicesError(VdbmatUtilsError):
    """A print-slices config, input, or derived grid violates the contract."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


from .types import PrintSlicesConfig  # noqa: E402

__all__ = [
    "PrintSlicesConfig",
    "PrintSlicesError",
]
