# ADR 0008: per-label SDF morphing semantics

Date: 2026-07-07
Status: accepted

## Context

Phase 2 needs label-preserving interpolation between sparse labeled key
slices, with documented deterministic behavior for topology changes,
missing slices, label conflicts, and interpolation outside the source
domain. Full spec: `docs/morphing.md`; plan D4.

## Decision

- **In-repo exact EDT, no scipy.** The Felzenszwalb–Huttenlocher separable
  lower-envelope squared-distance transform (`fields/edt.py`) is
  implemented with NumPy only, keeping the base install at `numpy +
  vdbmat`. It is exact, deterministic (fixed loop order), O(n) per row,
  with anisotropic spacing as axis weights; correctness is pinned by
  brute-force O(n²) comparison tests. Pure-Python row loops are accepted at
  Phase 2 sizes; acceleration belongs to Phase 5.
- **Per-label signed distances, linear in z.** For each label present in
  the stack, 2-D signed distances (negative inside) on both key slices,
  interpolated linearly; output pixel = argmin over labels where the
  distance is strictly negative, else the configured background.
- **Tie rule:** lowest material id wins (implemented as ascending-id
  iteration with a strict `<` update).
- **Absence rule:** a label absent from a key slice has distance `+inf`
  there; IEEE arithmetic then removes it from every interior slice of that
  gap (immediate, not gradual — a deliberate, documented consequence). The
  `-inf`/`+inf` (label fills one slice, absent from the other) NaN corner
  resolves to `+inf`.
- **Edge policy:** interior gaps are the feature; outside the key range is
  extrapolation and errors by default (`edge_policy: "error"`), or repeats
  the nearest key verbatim (`"clamp"`). Never linear extrapolation.
- **Key slices verbatim:** emitted from the mapped input pixels, never
  re-derived from distances (byte-faithful, regression-tested).
- **Identity:** generator `vdbmat-utils.morph.stack` v0.1.0; sources =
  per-key-slice digests in z order; identity = SHA-256 over sources + config
  digest via `core.provenance.provenance_identity`, factored out of
  `image/stack.py` rather than copied.

## Consequences

- Topology changes (merge/split/appear/disappear) are emergent and
  golden-tested, not special-cased.
- Exactly-symmetric configurations cancel to distance 0 at cell centers at
  the midpoint and yield background (0 is not `< 0`); documented in
  `docs/morphing.md`.
- Memory stays O(slice area) per label (two float64 slices plus running
  argmin buffers); size guards (`max_axis_cells`, `max_total_cells`) bound
  the cost.
- Deferred: 3-D multi-axis morphing, non-linear morph timing,
  correspondence-based morphing.
