# ADR 0012: metrics and optical-mapping emission

Date: 2026-07-07
Status: accepted

## Context

Procedural outputs need checkable structural claims and must run through vdbmat
optical conversion even when they use formation-specific material names.

## Decision

Statistics report exact volume fractions, 6-connected component counts,
largest-component fraction, and local thickness as `2 * EDT(mask)` sampled at
material voxels. Constraints are evaluated after generation and never repaired by
the generator.

When a palette name is outside vdbmat's built-in mapping, the formation config
must provide optical coefficients for exactly those names. vdbmat's public
optics API builds and writes the mapping document and computes the canonical
digest. The mapping digest is emitted by the CLI and recorded in the manifest
source notes.

## Consequences

Phase 3 makes scoped, measurable structural statements only. The local-thickness
metric is a conservative proxy; full sphere-fitting local thickness and
constraint-satisfying search are future work. Optical coefficient validity stays
outside vdbmat-utils; this package only packages user-supplied values into the
schema paired with the generated asset.
