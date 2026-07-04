"""Generator-configuration serialization and digest conventions.

Configurations are frozen dataclasses subclassing ``GeneratorConfig``. Their
canonical JSON form — sorted keys, no insignificant whitespace, NaN/Infinity
rejected, UTF-8 — is the input to the SHA-256 configuration digest recorded in
provenance, so two configurations are "the same run" exactly when their
canonical JSON is byte-equal.
"""

import dataclasses
import hashlib
import json
from typing import Any, Self

from .errors import ConfigError


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Base class for generator configurations.

    Subclasses must be frozen dataclasses whose field values are JSON-safe:
    ``str``, ``int``, ``float`` (finite), ``bool``, ``None``, or (possibly
    nested) lists/tuples/dicts of them. The seed is part of the configuration.
    """

    seed: int = 0

    def to_json(self) -> str:
        """Return the canonical JSON form of this configuration."""
        return config_to_canonical_json(self)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Reconstruct a configuration from its ``to_json`` output."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(f"invalid configuration JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ConfigError("configuration JSON must be an object")
        field_names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - field_names)
        if unknown:
            raise ConfigError(f"unknown configuration fields: {', '.join(unknown)}")
        try:
            return cls(**payload)
        except TypeError as error:
            raise ConfigError(str(error)) from error


def config_to_canonical_json(config: GeneratorConfig) -> str:
    """Serialize ``config`` to its canonical JSON form."""
    if not dataclasses.is_dataclass(config):
        raise ConfigError("configuration must be a dataclass instance")
    payload = _jsonable(dataclasses.asdict(config), path=type(config).__name__)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_digest(config: GeneratorConfig) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of ``config``.

    The format matches ``vdbmat.core.Provenance.configuration_digest``.
    """
    digest = hashlib.sha256(config.to_json().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _jsonable(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigError(f"{path}: non-finite float is not allowed")
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ConfigError(f"{path}: dict keys must be strings")
        return {
            key: _jsonable(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise ConfigError(f"{path}: unsupported value type {type(value).__name__}")
