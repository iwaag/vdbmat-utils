# ADR 0005: adoption of the recovered winding-number cell-centre voxelization

Date: 2026-07-06
Status: accepted

## Context

vdbmat once contained a complete mesh voxelization path — dense cell-centre classification by
signed winding number, topology validation, and an analytic test suite — designed in its
historical ADR-0006 and later deleted (last commit `8f55562`; see
`vdbmat/.local/memo_stltovoxel.md`). Phase 1 needs the same capability in vdbmat-utils.

## Options

1. **Port the recovered implementation wholesale (chosen).** The numerics encode debugged
   behavior: sub-voxel YZ jitter constants (`7.3e-5` / `3.1e-5`, deliberately unequal so
   samples leave 45° triangulation diagonals), the `1e-9` m closed-solid surface tolerance,
   the `1e-6` domain-snap epsilon absorbing float32 STL round-off, and the facing-mask
   epsilon. Rewriting would silently rediscover the bugs those constants fixed.
2. **Reimplement fresh** (e.g. flood fill, parity counting, or an SDF method) — cleaner-looking
   but discards a working, tested design for no Phase 1 benefit.

## Decision

Option 1. `vdbmat_utils.mesh.voxelizer` is the recovered algorithm with **no algorithmic
changes**; constants preserved verbatim, ported analytic tests (cube occupancy, mm/m grid
equivalence, closed-solid boundary rule, topology rejections) pin the behavior. Adaptations
are packaging only: output via the shared core builders, `MeshVoxelizeConfig` for
configuration, size guards promoted to config fields, errors re-rooted under
`VdbmatUtilsError`. Where the memo and the recovered code disagreed, the code won — notably
the jitter applies to the *winding-ray* evaluation while the surface test uses unjittered
centres (the memo stated the reverse; phase1 report3 records the delta). Explicit
`domain_min_m`/`domain_max_m` bounds are the one new feature: used verbatim, no padding added.

## Consequences

- Semantics are documented in `docs/voxelization.md`; the generator identity is
  `vdbmat-utils.mesh.voxelize` v0.1.0, bumped on any byte-level output change.
- The dense method is O(cells × triangles) and `_points_on_surface` is a known hotspot;
  Phase 1 deliberately does not optimize it (roadmap Principle 7 — benchmark first). The
  configurable `max_axis_cells` / `max_total_cells` guards bound the damage.
- Watertight-single-solid is a hard requirement with no `allow_open_mesh` escape hatch;
  mesh repair and multi-mesh composition are future phases.
