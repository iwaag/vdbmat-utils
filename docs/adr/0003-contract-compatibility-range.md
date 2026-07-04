# ADR 0003: supported vdbmat contract-version range

Date: 2026-07-05
Status: accepted

## Context

The roadmap requires recording compatibility with an explicit supported range of vdbmat contract
versions and failing clearly on unsupported major versions. The pinned vdbmat (ADR 0001)
currently declares `vdbmat.volume` schema 1.0.0 and writes `vdbmat.voxels` manifests with
`format_version` 1.0.0.

## Decision

- This package targets **major version 1** of the `vdbmat.volume` schema
  (`vdbmat_utils.core.compat.SUPPORTED_VOLUME_SCHEMA_MAJOR = 1`).
- Minor/patch bumps within major 1 are accepted without a code change, per the schema's own
  `has_compatible_major` semantics.
- `require_compatible_volume_schema()` checks the pinned `vdbmat.core.VOLUME_SCHEMA` at runtime;
  it is invoked by the volume builder and the `validate` CLI command, so an incompatible
  submodule bump fails fast with `CompatibilityError` instead of producing drifted assets.
- The golden-fixture contract tests additionally pin the exact output bytes, so even a
  compatible-looking upstream change that alters emitted manifests surfaces as a reviewed test
  failure (see `tests/contract/test_golden_fixtures.py`).

## Consequences

- Bumping the vdbmat submodule to a schema-2.x version requires raising
  `SUPPORTED_VOLUME_SCHEMA_MAJOR` deliberately, together with whatever migration the new major
  demands; CI blocks accidental bumps.
- The supported range lives in exactly one constant; documentation references the constant
  rather than duplicating version numbers.
