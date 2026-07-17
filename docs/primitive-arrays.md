# Primitive arrays

`vdbmat-utils generate-primitive-array` (API: `vdbmat_utils.primitives.generate_primitive_array`)
builds a canonical material-label volume of a transparent base block containing an
A × B × C grid of opaque cube or sphere inclusions, from one flat config with no input
files. It exists to give the designlab pipeline (`.devdocs/vision/designlab/roadmap.md`) a
minimal generator whose output is trivially countable by eye, so the input-generation
GUI's whole plumbing can be exercised before any method with real config complexity is
GUI-wired.

## Claims and non-claims

This is a light-transport **optical test pattern**, not a simulation of a 3D-print
process or of real material physics. It claims:

- deterministic, decomposable grid geometry (counts/size/gap/margin fully determine the
  shape — nothing is hand-fit), and
- a countable number of inclusions on any slice through the array.

It does **not** claim printability, structural feasibility, or physically calibrated
optical coefficients — those live in `vdbmat`'s optical-mapping layer, unaffected by this
generator.

## Configuration

`PrimitiveArrayConfig` JSON, passed via `--config`. No CLI field overrides exist (unlike
`voxelize-mesh`): the config file is the only input, which is what lets designlab's
GUI = CLI reproduction contract reduce to "hand the same config file to the CLI."

```json
{
  "voxel_size_xyz_m": [0.0001, 0.0001, 0.0001],
  "primitive": "cube",
  "counts_xyz": [3, 2, 1],
  "primitive_size_m": 0.0004,
  "gap_m": 0.0002,
  "margin_m": 0.0001,
  "base_material_name": "transparent-resin",
  "inclusion_material_name": "black-opaque-resin"
}
```

| Field | Meaning |
|---|---|
| `voxel_size_xyz_m` | grid resolution, meters, each component `> 0` |
| `primitive` | `"cube"` or `"sphere"` |
| `counts_xyz` | inclusion counts per axis, each `>= 1` |
| `primitive_size_m` | cube edge length / sphere diameter, `> 0` |
| `gap_m` | surface-to-surface spacing between adjacent inclusions along an axis, `>= 0`. **Required, no default** — "touching" and "one size apart" are equally plausible defaults, so the config must say which. |
| `margin_m` | spacing between the inclusion array's bounding box and the block's outer faces, `>= 0`. **Required, no default**, for the same reason as `gap_m`. |
| `base_material_name` | built-in material name for the block (default `transparent-resin`) |
| `inclusion_material_name` | built-in material name for the inclusions (default `black-opaque-resin`) |
| `max_axis_cells` / `max_total_cells` | size guards, defaults 256 / 8,000,000 |

`base_material_name`/`inclusion_material_name` accept only vdbmat's non-background
built-in names — `transparent-resin`, `white-resin`, `black-opaque-resin` — and must
differ from each other; `air` is rejected for both (it is the implicit, unused
background declaration, matching `nested_material_cube`). The name → id table is a
single constant in `vdbmat_utils.primitives.types.BUILTIN_MATERIAL_IDS`, pinned to the
ids in `vdbmat.optics.config.phase0_provisional_mapping()`; unit tests build a real
palette through `build_material_label_volume()` so drift from vdbmat's built-ins is
caught in-repo rather than downstream. `seed` is inherited from `GeneratorConfig` and
reserved: this generator uses no randomness, and the output payload does not depend on
it (only the config digest does, since `seed` is part of every config's canonical JSON).

No `local_to_world` field exists in Phase 1 — placement is always identity. Add it when
a concrete need appears.

## Grid derivation

Nothing about the grid shape is given directly; it is derived per axis from
`counts_xyz`, `primitive_size_m`, `gap_m`, and `margin_m`:

```
extent = 2 * margin_m + counts * primitive_size_m + (counts - 1) * gap_m
cells  = ceil(extent / voxel_size - 1e-6)
```

The `1e-6`-cell epsilon matches `voxelize-mesh`'s domain-snap constant: it absorbs
float round-off so an extent that is a whole number of cells up to floating-point noise
does not get a spurious padded cell. `max_axis_cells` / `max_total_cells` guard the
derived shape and, on overflow, name the offending axis and suggest coarsening the
voxel size — the same failure mode and message shape as `voxelize-mesh`.

## Sampling and classification

Cell-centre sampling, the same convention as the mesh voxelizer: cell `(k, j, i)`'s
centre is at `(i + 0.5, j + 0.5, k + 0.5) * voxel_size` (block-local origin at the
block's minimum corner). Primitive `(a, b, c)` (0-indexed per axis) is centered at
`margin + size/2 + index * (size + gap)` on each axis.

- **cube:** inside iff `max(|dx|, |dy|, |dz|) <= size/2` for its own primitive.
- **sphere:** inside iff `sqrt(dx² + dy² + dz²) <= size/2`.
- **boundary:** `<=` is inside (closed), matching the mesh voxelizer's closed-solid tie
  rule — including at `gap_m = 0`, where adjacent primitives touch.
- Every cell is either the inclusion material or the base material; no cell is `air`
  (the base fills the entire block, same as `nested_material_cube` — `air` is declared
  in the palette but never actually painted).

Because `gap_m >= 0`, a coordinate's per-axis distance to a primitive centre is
minimized independently on each axis, so the classification is separable and needs no
loop over the `counts_xyz` primitive grid, no ordering, and has no double-paint risk —
even when cubes touch exactly at `gap_m = 0`. The known limitation is cosmetic: at
coarse resolution a sphere renders with the same stepped/jagged boundary as any other
cell-centre-sampled curved surface (see `nested_material_cube`'s core in the top-level
`README_QUICK.md`); this is not a defect, and correctness is instead pinned by 3-axis
flip symmetry in the unit tests, not by visual smoothness.

## Worked example

```bash
uv run vdbmat-utils generate-primitive-array --config primarray.json --out out/ --name demo
uv run vdbmat-utils preview-slices out/demo.voxels.json --axis z
```

With the config shown above (3×2×1 cubes, 4-voxel edge, 2-voxel gap, 1-voxel margin),
the derived grid is 18×12×6 in x/y/z (`shape_zyx = (6, 12, 18)`), and every z-slice
through the inclusions shows a 3×2 array of `4 x 4`-cell squares:

```text
slice z=3  +x →  +y ↓
111111111111111111
133331133331133331
133331133331133331
133331133331133331
133331133331133331
111111111111111111
111111111111111111
133331133331133331
133331133331133331
133331133331133331
133331133331133331
111111111111111111
```

Provenance records generator `vdbmat-utils.primitives.array` v0.1.0 with no `sources`
(there is no input file); the config digest is the sole identity input. The version is
bumped whenever output for the same config changes byte-wise (`docs/determinism.md`).

## Downstream hand-off

The output uses the same built-in material names as the checked-in
`phase0-provisional-materials-v1` optical mapping, so the standard
`import-voxels` → `run` → `export`/viewer chain works with no new mapping file:

```bash
cd vdbmat
uv run vdbmat import-voxels ../vdbmat-utils/out/demo.voxels.json out/demo.zarr
uv run vdbmat run out/demo.run.json  # run config referencing out/demo.zarr and the
                                      # checked-in phase0-provisional-materials-v1 mapping
```

designlab (Phase 2 onward, `.devdocs/vision/designlab/roadmap.md`) automates this
hand-off; Phase 1 leaves it manual by design. `.devdocs/vision/designlab/p1/report5.md`
records one full manual walkthrough through `vdbmat run` and the existing viewer.
