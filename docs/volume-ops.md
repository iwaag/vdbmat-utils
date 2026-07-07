# Label-safe volume operations (`vdbmat_utils.ops`)

Every operation is a pure function `MaterialLabelVolume (+ params) →
MaterialLabelVolume`, re-validated through the canonical builder on output.
Operations move `uint16` labels **exactly** — nothing in `ops/` may cast a
label array to float or interpolate it (enforced by type separation, an AST
guard test, and nearest-neighbor-only resampling; see ADR 0007). All ops are
deterministic; there is no RNG anywhere in the package.

Available from Python (`from vdbmat_utils.ops import ...`) and as pipeline
steps (`docs/pipelines.md`).

| Op | Parameters | Semantics |
| --- | --- | --- |
| `crop` | `min_zyx`, `max_zyx` | half-open index box in canonical z,y,x; out-of-range is an error (no implicit clamping); `local_to_world` recomposed so surviving voxels keep world positions |
| `pad` | `before_zyx`, `after_zyx`, `fill_material_id` | grow by whole cells; fill defaults to background and must exist in the palette; transform recomposed likewise |
| `resample` | `factor_zyx` *or* `voxel_size_xyz_m` | nearest-neighbor only (see below) |
| `orient` | `steps`, `update_transform` | exact flips / 90° rotations; by default the inverse motion is composed into `local_to_world` (world geometry preserved); a net mirror is rejected as non-rigid unless `update_transform=False` (a genuine re-orientation) |
| `place` | `local_to_world`, `compose_with_existing` | metadata-only transform replacement/composition; never resamples |
| `apply_mask` | `mask`, `mode="keep"\|"clear"`, `fill_material_id` | mask is a second label volume (nonzero = selected) with **identical geometry**; keep fills outside the selection, clear fills inside |
| `compose` | `base`, `overlay`, `mode` | boolean composition, identical geometry required (below) |
| `remap_materials` | `mapping`, `definitions`, `prune_palette` | bulk id rewrite via lookup table; unmapped ids pass through; collapsing ids must agree on definition; `definitions` renames by *new* id; pruning drops entries labeling no voxel (background kept) |

## Geometry rules for binary ops

`compose` and `apply_mask` require **exact** geometry equality: shape, voxel
size, and `local_to_world` must all match. A mismatch is an error naming the
offending field and pointing at `crop`/`pad`/`resample`/`place` as the fix.
There is no auto-alignment in Phase 2.

## Boolean composition semantics

Foreground means "not background (id 0)".

- `union` — overlay foreground wins over base ("last writer wins"); palettes
  are merged (shared ids must have identical name/role/external_id;
  conflicts are an error pointing at `remap-materials`).
- `intersect` — keep base labels only where overlay is foreground; base
  palette kept.
- `subtract` — clear base labels where overlay is foreground; base palette
  kept.

## Resampling

Nearest-neighbor only: each output cell copies the label of the source cell
containing its center (`floor((i + 0.5) / factor)`, integer indexing), so
interpolation — and therefore label mixing — is impossible by construction.

- Exactly one of `factor_zyx` (output cells per source cell) and
  `voxel_size_xyz_m` (new cell size) must be given.
- Integer up/downsampling factors are exact repetition/decimation.
  Non-integer factors are allowed but **alias-prone** — thin features can
  vanish or double.
- The manifest voxel size updates; the local origin is unchanged, so
  `local_to_world` is untouched.

## Conservation claims

- `remap_materials` — total voxel count is preserved and per-id counts move
  exactly as mapped (contract-tested).
- `apply_mask` — `keep` and `clear` partition the base foreground exactly;
  `keep` with a label mask equals `compose(..., mode="intersect")`.
- `compose` — foreground counts follow the set identities
  (union = |base ∪ overlay|, intersect = |base ∩ overlay|,
  subtract = |base − overlay|), contract-tested.
- `resample` — **no conservation claim.** Counts scale only approximately
  with volume ratio, and non-integer factors alias.

## Future work (deferred, ADR 0009)

Smooth (per-label SDF) label rescaling — possible with the `fields`
machinery Phase 2 already ships — and general non-90° rigid resampling,
which needs an interpolation policy.
