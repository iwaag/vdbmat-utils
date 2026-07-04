"""Exception hierarchy for vdbmat-utils.

``vdbmat.core.VolumeValidationError`` raised by the canonical types passes
through unwrapped; these exceptions cover failures that happen before a
canonical volume exists.
"""


class VdbmatUtilsError(Exception):
    """Base class for all vdbmat-utils errors."""


class ConfigError(VdbmatUtilsError):
    """A generator configuration is invalid or cannot be (de)serialized."""


class GeometryError(VdbmatUtilsError):
    """Grid shape, voxel size, or transform inputs are invalid."""


class PaletteError(VdbmatUtilsError):
    """Material palette inputs are invalid or inconsistent."""


class CompatibilityError(VdbmatUtilsError):
    """The pinned vdbmat contract version is outside the supported range."""
