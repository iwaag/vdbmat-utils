# Morphing sparse key slices (`morph-stack`)

`morph-stack` builds a full label volume from a *sparse* set of labeled 2D
key slices, interpolating the slices in between while preserving discrete
material ids. It is the Phase 2 companion to `convert-image-stack`, which
requires a dense, gap-free stack.

```bash
uv run vdbmat-utils morph-stack slices/ --config morph.json --out out/ --name part
# overrides: --voxel-size X Y Z, --z-count N (both re-digested into provenance)
```

## Input contract

- A directory of PGM slices (PNG with the `image` extra). Filenames must
  contain **exactly one numeric group**, which *is* the output z index:
  `slice_0000.pgm`, `slice_0008.pgm`, `slice_0020.pgm` → keys at z = 0, 8, 20.
  Gaps between key indices are the point of this generator, not an error.
- Duplicate z indices, filename order disagreeing with numeric order, or a
  stem with zero/multiple numeric groups are errors.
- All key slices share one `levels` table (exactly the image-stack contract:
  one gray value → one material id stack-wide). An undeclared gray value is
  an error naming the file, pixel, and value. Rows map to +Y, columns to +X.

## Configuration (`MorphStackConfig`)

| Field | Default | Meaning |
| --- | --- | --- |
| `voxel_size_xyz_m` | required | cell size in metres, x/y/z |
| `levels` | required | gray → (material_id, name, role) table |
| `background` | `0` | material id used where no label is inside |
| `z_count` | last key + 1 | total output depth |
| `edge_policy` | `"error"` | `"error"` or `"clamp"` (see below) |
| `local_to_world` | identity | optional rigid placement |
| `format` | `"pgm"` | `"pgm"` or `"png"` |
| `max_axis_cells` | `256` | size guard per axis |
| `max_total_cells` | `8_000_000` | size guard on the whole volume |

`seed` is inherited from `GeneratorConfig`, unused, and reserved.

## Algorithm (per-label SDF interpolation)

For each pair of consecutive key slices and each material id present
anywhere in the stack:

1. Compute the 2-D **signed** Euclidean distance field of the label's mask
   on both key slices (negative inside, positive outside, in metres; the
   exact in-repo Felzenszwalb–Huttenlocher transform — no scipy).
2. For an output slice at fraction `t` between keys `z0 < z < z1`,
   interpolate distances linearly: `d(z) = (1-t)·d(z0) + t·d(z1)`.
3. Each output pixel takes the label with the **minimum interpolated
   distance**, provided that distance is **strictly negative**; ties break
   to the **lowest material id**. If no label is inside, the pixel gets
   `background`.

Key slices themselves are emitted **verbatim** from the input images (never
re-derived), so key-slice pixels are byte-faithful to their sources.

### Topology changes are emergent

Merging, splitting, appearing, and disappearing shapes need no special
casing — they fall out of distance interpolation. Two squares growing toward
each other connect when their interpolated interiors first overlap:

```text
z=0 (key)      z=2 (interpolated)   z=4 (key)
.11...11.      .111..111.           .11111111.
.11...11.  →   .111..111.       →   .11111111.
```

### Absent labels vanish immediately

A label absent from a key slice has distance `+inf` everywhere on that
slice. IEEE arithmetic makes `(1-t)·d + t·(+inf) = +inf` for every interior
`t > 0`, so a label present on only one side of a gap exists **only on its
key slice** — it does not shrink gradually. (Shrinking happens when both
sides contain the label at different sizes.) The one indeterminate corner —
a label filling one key slice entirely (`-inf`) and absent from the other
(`+inf`) — resolves to `+inf`: with no boundary on either side there is
nothing to interpolate.

### Exactly-symmetric morphs are empty at the midpoint

Distances are sampled at cell centers, and "inside" means strictly `< 0`.
When two configurations mirror each other exactly, the midpoint distances
cancel to exactly `0` at every cell, which is *not* inside — the midpoint
slice is all background. This is the documented consequence of the argmin
rule, not a bug.

## Missing slices and the edge policy

Interior gaps between declared keys are always interpolated. Slices
**outside** the key range (the first key is not 0, or `z_count` extends past
the last key) would be extrapolation:

- `edge_policy: "error"` (default) — rejected with an actionable message.
- `edge_policy: "clamp"` — the nearest key slice is repeated verbatim.

There is **no linear extrapolation**, ever. `z_count` smaller than
`last key + 1` is always an error.

## Label conflicts

One `levels` table governs the whole stack; duplicate gray values, or two
levels mapping to one material id, are config errors (shared with
`convert-image-stack`). `background` must be one of the declared ids.

## Determinism, provenance, cost

- No RNG anywhere; double runs are byte-equal (contract-tested).
- Generator `vdbmat-utils.morph.stack` v0.1.0. Provenance `sources` are the
  per-key-slice SHA-256 digests in z order; the asset identity hashes those
  digests plus the config digest (shared recipe with the image stack).
- Cost is O(labels × slice area) time per key pair and O(slice area) memory
  per label (two float64 slices plus the running argmin buffers). The size
  guards bound the worst case; exceeding them suggests fewer output slices
  or downsampled input.

## Deferred (out of scope in Phase 2)

3-D multi-axis morphing, non-linear morph timing, and correspondence-based
morphing — see ADR 0008.
