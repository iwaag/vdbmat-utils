# Determinism rules

Every vdbmat-utils generator must satisfy: **same inputs + same configuration (including seed) →
byte-equal output**, for both the `<name>.material_id.npy` payload and the `<name>.voxels.json`
manifest.

## What makes this hold

- **Seed handling** (`vdbmat_utils.core.seeds`): all randomness flows from one integer seed via
  `numpy.random.default_rng`; independent consumers use `spawn` substreams so adding a consumer
  never shifts existing streams. Python's `random` module and NumPy's legacy global state are
  forbidden.
- **Configuration identity** (`vdbmat_utils.core.config`): a configuration's canonical JSON
  (sorted keys, compact separators, finite floats only) defines run identity; its SHA-256 is the
  `configuration_digest` recorded in provenance. The seed is part of the configuration.
- **No wall-clock or environment data in output** (`vdbmat_utils.core.provenance`): `created_utc`
  is left unset; timestamps, hostnames, and absolute paths must not appear in manifests. Execution
  time, if worth recording, belongs in logs or an adjacent execution record, not the asset.
- **Payload checksum**: the manifest records the payload SHA-256 (written by `vdbmat`'s emitter),
  so byte-level regressions are detectable by comparing manifests alone.

## Verification

`tests/contract/test_determinism.py` writes the same volume twice into separate directories and
asserts both files are byte-equal. Every new generator must add the equivalent double-run test.

## Scope

Byte equality is required on the same platform and pinned dependency set (`uv.lock` + the `vdbmat`
submodule pin). Cross-platform or cross-NumPy-version byte equality is *not* promised; if a future
generator cannot achieve byte equality (e.g. parallel floating-point reduction), it must document
a scientific-equivalence rule and a tolerance-based test instead, per roadmap principle 4.
