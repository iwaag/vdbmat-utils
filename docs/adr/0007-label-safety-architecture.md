# ADR 0007: label-safety architecture (ScalarField split, quantize-only conversion, AST guard)

Date: 2026-07-07
Status: accepted

## Context

Phase 2 introduces morphing and volume operations. The classic failure mode
(roadmap: material ids "never accidentally interpolated as numeric
intensities") is a convenience float cast deep in a resample or morph path
producing plausible but wrong mixed labels — e.g. averaging label 1 and
label 3 into a phantom label 2.

## Decision

Label safety is structural, enforced three ways:

1. **Type split.** Discrete labels stay `MaterialLabelVolume` / `uint16`
   arrays. Continuous data lives in `fields.ScalarField` — a frozen
   dataclass (float64 z,y,x array + grid geometry, **no palette**), so a
   field has no material semantics until explicitly converted. All smooth
   math (the in-repo exact EDT, distance interpolation) operates on
   `ScalarField` data; signatures are mypy-checked.
2. **One conversion gate.** `fields/quantize.py::quantize_to_labels` is the
   *only* sanctioned scalar→label conversion (strictly increasing bin
   edges, deterministic edge rule, NaN rejected). Nothing else may
   manufacture label values from floats.
3. **AST guard test.** `tests/unit/test_label_safety.py` walks the sources
   of `ops/` and `morph/` and fails on interpolating calls (`interp`,
   `map_coordinates`, `zoom`, …), float casts, and `.mean()` on label-ish
   names — the same cheap lint-style approach as the import-isolation test.
   `fields/` is exempt by design: that is where continuous math belongs.

Additionally, label resampling is **nearest-neighbor only** in Phase 2
(`ops/resample.py` uses pure integer indexing), so interpolation is
impossible by construction on that path too.

## Consequences

- New label-touching code lands inside the guarded packages and inherits
  the checks automatically (the `morph` rows were declared before the
  package existed and activated when it appeared).
- The guard is heuristic, not a proof — reviews of `ops`/`morph` changes
  should still watch for label arithmetic the AST patterns miss.
- Smooth label rescaling (per-label SDF resampling) remains possible later
  *through* the sanctioned machinery: distances are fields, and the final
  argmin/quantize step is explicit (deferred; ADR 0009).
