# ADR 0009: volume-op and pipeline contracts

Date: 2026-07-07
Status: accepted

## Context

Phase 2 ships composable volume operations and configuration-driven
pipelines (plan D5/D6). The contracts must be strict enough that failures
are early and actionable, and simple enough to diff and reason about. Full
specs: `docs/volume-ops.md`, `docs/pipelines.md`.

## Decision

- **Exact-geometry requirement for binary ops.** `compose` and `apply_mask`
  require identical shape, voxel size, and `local_to_world`; the error
  names the mismatching field and the reconciling ops
  (`crop`/`pad`/`resample`/`place`). No auto-alignment.
- **Union is last-writer-wins.** Overlay foreground (nonzero) beats base;
  the deterministic conflict rule where both are foreground. Intersect and
  subtract keep the base palette.
- **Palette-merge rules.** Shared ids merge only on identical
  name/role/external_id; conflicts are a `PaletteError` pointing at
  `remap-materials`, which exists precisely to fix ids before composing.
- **SSA-style pipeline config.** A flat list of steps with named
  single-assignment ids (`"from"`/`"base"`/`"overlay"`/`"mask"` read,
  `"as"` binds; `output: {"ref": id}` selects the result) — not a nested
  expression tree. Unknown ops/parameters, unbound/rebound ids, and unused
  inputs fail at validation time with the step index, before any array
  work. Relative input paths resolve against the config file's directory.
- **Provenance.** Pipeline sources = input-manifest SHA-256 digests in
  input order; the config digest covers the canonical config JSON as
  written (paths included); identity = shared `provenance_identity` recipe.
- **No conservation claim for resample** (nearest-neighbor aliasing);
  conservation identities for remap/mask/compose are contract-tested.

## Consequences

- Users combine assets by aligning geometry explicitly first; error
  messages teach the workflow.
- The registry (`pipeline/registry.py`) makes adding an op a local change:
  one spec (volume keys, parameter names, JSON adapter) plus docs.
- **Deferred deliberately** (scope control): generator steps inside
  pipelines (morph/image-stack as steps — pipelines start from existing
  `.voxels.json` assets), conditionals/loops/variables, caching, parallel
  execution (Phase 4/5), smooth SDF label rescaling, and general non-90°
  rigid resampling.
