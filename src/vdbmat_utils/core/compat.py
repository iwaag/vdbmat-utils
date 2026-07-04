"""Supported vdbmat contract-version range.

The effective pin is the ``vdbmat`` submodule commit recorded by the
``pj-voxel3dprint`` superproject (ADR 0001); this module asserts at runtime
that the pinned contract version is one this package knows how to target.
"""

from vdbmat.core import VOLUME_SCHEMA, SchemaIdentity

from .errors import CompatibilityError

SUPPORTED_VOLUME_SCHEMA_MAJOR = 1
"""Major version of the ``vdbmat.volume`` schema this package targets."""


def require_compatible_volume_schema(
    schema: SchemaIdentity = VOLUME_SCHEMA,
) -> SchemaIdentity:
    """Return ``schema`` if supported, otherwise raise ``CompatibilityError``."""
    if schema.name != "vdbmat.volume":
        raise CompatibilityError(
            f"unsupported schema name {schema.name!r}; expected 'vdbmat.volume'"
        )
    if schema.version.major != SUPPORTED_VOLUME_SCHEMA_MAJOR:
        raise CompatibilityError(
            f"vdbmat volume schema {schema.version} is unsupported; "
            f"this vdbmat-utils release targets major version "
            f"{SUPPORTED_VOLUME_SCHEMA_MAJOR}"
        )
    return schema
