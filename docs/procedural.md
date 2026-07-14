# Procedural Formations

`vdbmat-utils generate-formation` creates deterministic material-label volumes from
seeded procedural layers. The output is the normal vdbmat interchange pair:
`<name>.voxels.json` and `<name>.material_id.npy`; when any palette name is not a
vdbmat built-in, `<name>.optical-mapping.json` is emitted too.

```bash
uv run vdbmat-utils generate-formation \
  --config examples/formation_generation/marble-like.formation.json \
  --out out/marble --name marble-like --strict

uv run vdbmat-utils sweep-formation \
  --config examples/formation_generation/tiny-sweep.sweep.json \
  --out out/sweep --name tiny
```

## Config Shape

A formation config declares:

- `seed`: non-negative integer controlling hash-derived fields.
- `shape_zyx`: dense output grid shape.
- `voxel_size_xyz_m`: voxel size in x, y, z metres.
- `palette`: vdbmat material definitions (`material_id`, `name`, `role`).
- `layers`: ordered painter layers; later layers overwrite earlier labels.
- `constraints`: optional post-generation checks.
- `mapping`: optical coefficients for non-built-in material names.

All feature sizes are expressed in metres. Primitive evaluation uses voxel-centre
coordinates in the local frame; `local_to_world`, when supplied, is output metadata.

## Layers

`host` fills the domain with one material, or quantizes an fBm field using
`bin_edges` and `material_ids`.

`strata` paints repeating bands along `axis` with `thickness_m` and
`material_ids`. Optional fBm warp shifts the band coordinate.

`veins` paints sheet-like veins where a warped axis coordinate is within
`width_m / 2` of an offset or repeating spacing. This is the marble-like model.

`fractures` shares the vein implementation but is intended for thinner late
overlays.

`grains` uses Worley site ids. Each cell receives a material by stable hashing of
the site id and configured weights; optional `boundary_material_id` paints crystal
boundaries from the `f2 - f1` field. This is the granite-like model.

`pores` thresholds fBm or ridged fBm and may open the boolean mask before painting
the pore material, often `air`.

## Determinism

Lattice randomness is coordinate-hashed, not drawn from array-order RNG state.
The same `(ix, iy, iz, stream_id, seed)` always maps to the same hash, so cropping
or extending a domain does not reshuffle the interior. Noise gradients, Worley
jitter, and grain material picks all derive from this hash. Contract tests pin
payload digests, double-run byte equality, seed sensitivity, and axis-orientation
ASCII previews.

## Claims And Non-Claims

These models target visual and structural plausibility under the reported metrics.
They are not geological simulations, optical calibration, print-process simulation,
or predictions of mechanical/optical performance.

- Marble-like veins: claims controllable sheet width, spacing, warp, and volume
  fraction. Does not claim mineral genesis, stress history, or polished-stone
  appearance accuracy.
- Granite-like grains: claims stable cell-scale material domains and boundary
  labels. Does not claim crystallography, grain growth, or petrographic validity.
- Pores/fractures: claims explicit label masks with measurable thickness and
  connectivity. Does not claim crack propagation, porosity physics, or print
  reliability.

Use `formation-stats` and constraints for the quantitative statements the tool
actually supports.
