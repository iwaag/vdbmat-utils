"""Provenance-assembly convention shared by all generators."""

from vdbmat.core import Provenance

from .config import GeneratorConfig, config_digest


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
