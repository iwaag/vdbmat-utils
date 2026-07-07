"""Provenance-assembly convention shared by all generators."""

import hashlib

from vdbmat.core import Provenance

from .config import GeneratorConfig, config_digest
from .errors import ConfigError


def build_provenance(
    *,
    generator: str,
    generator_version: str,
    config: GeneratorConfig | None = None,
    sources: tuple[str, ...] = (),
    notes: str | None = None,
) -> Provenance:
    """Return a ``vdbmat.core.Provenance`` following utils conventions.

    The configuration digest is always derived from the configuration's
    canonical JSON (which includes the seed), never hand-written.
    ``created_utc`` is deliberately left unset so that repeated runs produce
    identical provenance (see ``docs/determinism.md``).
    """
    return Provenance(
        generator=generator,
        generator_version=generator_version,
        configuration_digest=config_digest(config) if config is not None else None,
        sources=sources,
        notes=notes,
    )


def provenance_identity(provenance: Provenance) -> str:
    """Asset identity shared by source-driven generators (image stack, morph,
    pipeline): SHA-256 over the concatenated provenance ``sources`` (in
    order) plus the configuration digest."""
    if provenance.configuration_digest is None:
        raise ConfigError("provenance has no configuration digest")
    combined = hashlib.sha256()
    for source in provenance.sources:
        combined.update(source.encode("utf-8"))
    combined.update(provenance.configuration_digest.encode("utf-8"))
    return f"sha256:{combined.hexdigest()}"
