"""Shared conventions for building vdbmat canonical volumes.

The canonical types themselves live in ``vdbmat.core``; this package only adds
the construction helpers and conventions (configuration serialization, seed
handling, provenance assembly, compatibility checks) shared by every generator.
"""

from .builder import build_material_label_volume
from .compat import SUPPORTED_VOLUME_SCHEMA_MAJOR, require_compatible_volume_schema
from .config import GeneratorConfig, config_digest, config_to_canonical_json
from .errors import (
    CompatibilityError,
    ConfigError,
    GeometryError,
    PaletteError,
    VdbmatUtilsError,
)
from .provenance import build_provenance, provenance_identity
from .seeds import rng_from_seed, spawn_rngs

__all__ = [
    "SUPPORTED_VOLUME_SCHEMA_MAJOR",
    "CompatibilityError",
    "ConfigError",
    "GeneratorConfig",
    "GeometryError",
    "PaletteError",
    "VdbmatUtilsError",
    "build_material_label_volume",
    "build_provenance",
    "config_digest",
    "config_to_canonical_json",
    "provenance_identity",
    "require_compatible_volume_schema",
    "rng_from_seed",
    "spawn_rngs",
]
