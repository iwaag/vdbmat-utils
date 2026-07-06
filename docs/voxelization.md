# Mesh voxelization

`vdbmat-utils voxelize-mesh` (API: `vdbmat_utils.mesh.voxelize_mesh`) converts one watertight
STL solid into a canonical material-label volume. The algorithm is the dense cell-centre
reference voxelizer recovered from vdbmat git history (commit `8f55562`; see ADR 0005) —
correctness and inspectability over speed.

## Input contract

- **Formats:** STL only, binary or ASCII, auto-detected (ADR 0004). A file starting with
  `solid` is still parsed as binary when its length matches the declared triangle count
  exactly.
- **Topology:** the mesh must be a watertight, consistently oriented, single connected solid.
  `inspect_topology` rejects open surfaces, non-manifold edges, inconsistent winding,
  degenerate faces, and multiple disconnected solids — always an error, no escape hatch
  (composition workflows are Phase 2).
- **Units:** STL is unitless, so `source_unit` (`"m"` or `"mm"`) is **required with no
  default**. The mm and m representations of the same shape produce identical grids
  (regression-tested).

## Semantics

- **Sampling rule:** dense cell-centre classification. Cell `(z, y, x)` gets the foreground
  material iff its centre — at `origin + (i + 0.5) * voxel_size` per axis — is inside or on
  the surface of the mesh.
- **Inside test:** signed winding number along a +X ray; inside iff `|winding| >= 0.5`.
  Triangles nearly parallel to the X axis are excluded by a facing mask; the surface test
  compensates.
- **Closed-solid tie rule:** centres *on* the surface count as inside (plane distance
  ≤ 1e-9 m plus barycentric containment).
- **Numerical jitter:** the winding ray's YZ sample point gets distinct sub-voxel offsets
  (`7.3e-5` / `3.1e-5` of a voxel) so centres cannot sit on an interior triangulation
  diagonal; the surface test uses the unjittered centres. These constants encode debugged
  behavior — do not change them casually (ADR 0005).
- **Axis order:** the array is `z, y, x` (canonical). The optional rigid `placement` (4×4,
  default identity) is composed with the derived origin as `placement @ translation(origin)`
  to form the manifest's `local_to_world`; the voxel-local array is unaffected by placement.

## Domain fitting

Auto-fit takes the mesh AABB, snaps each axis to whole cells with a 1e-6-cell epsilon (so
float32 STL round-off cannot add a spurious cell), then adds `padding_cells` (default 1) on
all sides. Alternatively give `domain_min_m`/`domain_max_m` (both or neither); explicit
bounds are used verbatim — **no padding is added**. Per-axis (`max_axis_cells`, default 128)
and total (`max_total_cells`, default 2 000 000) guards reject oversized grids with a
suggestion to coarsen the voxel size; the dense method is O(cells × triangles).

## Configuration

`MeshVoxelizeConfig` JSON, passed via `--config`; CLI flags (`--source-unit`,
`--voxel-size X Y Z`, `--material-id`, `--material-name`, `--padding`) override config
fields, and the *effective* config is what gets digested into provenance. `seed` is
inherited, unused in Phase 1, and reserved.

```json
{
  "source_unit": "mm",
  "voxel_size_xyz_m": [0.0005, 0.0005, 0.0005],
  "material": {"material_id": 1, "name": "transparent-resin", "role": "material"},
  "background": {"material_id": 0, "name": "air", "role": "background"},
  "padding_cells": 1
}
```

Exactly one foreground material per mesh; `background` defaults to air at id 0.
Multi-mesh composition is Phase 2.

## Worked example

The in-code L-bracket fixture (`vdbmat_utils.fixtures.l_bracket_stl_bytes`) is a 3×2×1 mm
L-prism. With the config above (0.5 mm voxels, default padding) the grid is 8×6×4 in x/y/z
— `shape_zyx = (4, 6, 8)` — and exactly 32 cells are occupied: bracket volume
4 mm³ ÷ (0.5 mm)³.

```bash
uv run vdbmat-utils voxelize-mesh l_bracket.stl --config config.json --out out/ --name bracket
uv run vdbmat-utils preview-slices out/bracket.voxels.json --axis z
```

```text
slice z=2  +x →  +y ↓
........
.111111.
.111111.
.11.....
.11.....
........
```

Provenance records generator `vdbmat-utils.mesh.voxelize` v0.1.0, the mesh file's SHA-256
(also the manifest's source identity), and the config digest; the version is bumped whenever
output for the same inputs changes byte-wise (`docs/determinism.md`).
