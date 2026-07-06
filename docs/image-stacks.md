# Image-stack conversion

`vdbmat-utils convert-image-stack` (API: `vdbmat_utils.image.convert_image_stack`) stacks a
directory of labeled 2D grayscale slices into a canonical material-label volume. The contract
is the migrated vdbmat reference generator's, extended with PNG input and an optional
transform (ADR 0006).

## Input contract

- **Formats:** PGM (P5 binary or P2 ASCII, 8-bit) with the zero-dependency built-in reader,
  or PNG (grayscale, 8-bit) via the `image` extra — `pip install 'vdbmat-utils[image]'`;
  without it PNG input fails with an actionable error. Non-grayscale and >8-bit PNGs are
  rejected explicitly.
- **Stacking:** slices are read in ascending filename order as z = 0, 1, …; image rows map
  to +Y and columns to +X. All slices must share one shape and bit depth.
- **Sequence gaps:** if every filename ends in a number, a missing index
  (`slice_0003.pgm` … `slice_0005.pgm`) is an error naming the gap — interpolation is
  Phase 2, not a silent skip.
- **Declared levels only:** every gray value present must be declared in `levels`; an
  undeclared value fails naming the value and the first offending file/pixel.

## Configuration

`ImageStackConfig` JSON, passed via `--config`; `--voxel-size X Y Z` and `--format pgm|png`
override config fields, and the effective config is digested into provenance. `seed` is
inherited, unused in Phase 1, and reserved.

```json
{
  "voxel_size_xyz_m": [0.0001, 0.0002, 0.0003],
  "levels": [
    {"gray": 0,   "material_id": 0, "name": "air",               "role": "background"},
    {"gray": 100, "material_id": 1, "name": "transparent-resin", "role": "material"},
    {"gray": 255, "material_id": 2, "name": "white-resin",       "role": "material"}
  ],
  "format": "pgm"
}
```

- Each `levels` entry maps one gray value (0–255) to one palette material; duplicate gray
  values or duplicate material ids are errors.
- Optional `local_to_world` (rigid 4×4) is recorded in the manifest; default identity.

## Example

```bash
uv run vdbmat-utils convert-image-stack slices/ --config config.json --out out/ --name stack
uv run vdbmat-utils validate out/stack.voxels.json
uv run vdbmat import-voxels out/stack.voxels.json out/stack.zarr
```

On success the CLI prints the manifest path and a per-material voxel-count summary.
Material counts are conserved: per-material pixel counts summed over the slices equal the
volume's voxel counts (contract-tested).

Provenance records generator `vdbmat-utils.image.stack` v0.1.0 (a fresh identity — no
compatibility claim with the deleted vdbmat tool), per-slice SHA-256 digests in stack order
as sources, and an asset identity hashed over those digests plus the config digest.
