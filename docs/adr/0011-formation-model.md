# ADR 0011: formation model

Date: 2026-07-07
Status: accepted

## Context

Phase 3 needs a small vocabulary for natural-material-like formations without
letting each model invent output semantics or label interpolation.

## Decision

A `FormationConfig` declares a palette, a dense domain, and ordered layers.
Generation starts with a `host` layer; every later layer computes a boolean
support mask and material values for that mask. Painting is last-writer-wins.
Every emitted id must exist in the palette.

Continuous work stays in `ScalarField` or boolean masks. Label conversion occurs
only through `quantize_to_labels` for scalar host fields or integer writes on
boolean supports. Warping displaces coordinates before primitive evaluation; it
does not resample an existing label or scalar array.

## Consequences

Layer order is the only precedence rule. The layer vocabulary covers host rock,
strata, veins, grains/crystals, pores, and fractures while preserving the same
vdbmat material-label contract as all earlier workflows. Size guards bound dense
NumPy memory use; sparse/chunked generation is deferred.
