"""Seed-handling convention shared by all generators.

Every generator takes a single non-negative integer seed. Random state is
always a ``numpy.random.Generator`` derived from that seed; independent
substreams come from ``spawn`` so adding a consumer never shifts the streams
of existing ones. Python's ``random`` module and NumPy's legacy global state
must not be used.
"""

import numpy as np

from .errors import ConfigError


def rng_from_seed(seed: int) -> np.random.Generator:
    """Return the root ``Generator`` for a generator run."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ConfigError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(seed)


def spawn_rngs(rng: np.random.Generator, count: int) -> list[np.random.Generator]:
    """Return ``count`` independent child generators of ``rng``."""
    if count < 0:
        raise ConfigError(f"count must be non-negative, got {count}")
    return rng.spawn(count)
