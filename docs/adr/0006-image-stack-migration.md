# ADR 0006: image-stack migration contract, PNG extension, and old-tool deletion

Date: 2026-07-06
Status: accepted

## Context

vdbmat shipped a reference image-stack generator (`tools/image_stack_generator/generate.py`,
its ADR-009 D2 example) with no public API or tests of its own. The roadmap moves image-stack
generation into vdbmat-utils "when its public API and compatibility tests are ready".
Backward compatibility is explicitly not required (user decision, 2026-07-06): delete
superseded tools outright rather than stubbing.

## Decision

- **Contract ported unchanged:** slices stack in ascending filename order as z = 0, 1, …;
  rows → +Y, columns → +X; every gray value present must be declared in `levels`; one shape
  and bit depth per stack; a gap in a numerically-named sequence is an error (interpolation
  is Phase 2).
- **Extensions:** PNG input (grayscale 8-bit) behind the `image` extra (`pillow>=10`) with a
  lazy import and an actionable install hint; optional rigid `local_to_world`; a real public
  API (`convert_image_stack(slices_dir, config) -> MaterialLabelVolume`) and
  `ImageStackConfig(GeneratorConfig)`. PGM keeps a zero-dependency reader so the base
  workflow needs no extra.
- **Fresh identity:** generator `vdbmat-utils.image.stack` v0.1.0 — no compatibility claim
  with the deleted tool's output. Provenance sources are per-slice digests in stack order;
  the asset identity hashes those digests plus the config digest.
- **Deletion, not deprecation:** with the contract tests green (phase1 report2),
  `vdbmat/tools/image_stack_generator/` and its test were deleted from vdbmat and the one
  forward-looking reference (`README_EXTEND.md`) now points at
  `vdbmat-utils convert-image-stack` (phase1 report5). No equivalence test against the old
  tool's output — the new generator's contract tests are the source of truth; the old
  behavior remains recoverable from vdbmat git history.

## Consequences

- vdbmat no longer contains any input generator; the generator-contract handoff
  (`vdbmat.voxels` manifest + payload) is exercised cross-repo by the `integration`-marked
  tests against the pinned vdbmat CLI.
- The deletion is a cross-repo change: the vdbmat commit and the superproject submodule
  bump follow ADR 0001's procedure, gated on this repo's contract suite.
- PNG stacks require the `image` extra; CI keeps a no-extras leg proving the PGM workflow
  and previews work in the minimal install.
