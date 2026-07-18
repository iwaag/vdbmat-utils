# Print slices (GrabCAD PNG method)

`vdbmat-utils export-print-slices` (API: `vdbmat_utils.printer.export_print_slices`)
converts a material-label voxel asset (`.voxels.json` + `.material_id.npy`) into an
indexed-palette PNG slice stack plus a sidecar manifest, on a printer-pitch grid derived
from physical extent — the layout GrabCAD Print's voxel-printing **PNG method** expects
(Stratasys' *Guide to Voxel Printing*). It is the reverse direction of
`convert-image-stack` (image stack → voxels); see
`.devdocs/vision/printer_export/roadmap.md` for why the two live on opposite sides of the
same interchange contract and why this exporter is a `vdbmat-utils` concern rather than a
`vdbmat` renderer export (the input is material identity, not optical properties — no
`vdbmat run` mapping step is involved).

## Claims and non-claims

This phase claims:

- deterministic, physically-derived output grid and nearest-neighbour-only resampling
  (no interpolation, averaging, or majority voting anywhere),
- printer-constraint violations (color count, slice count, palette mismatch) are
  detected and rejected before anything is published,
- atomic publish: a failed or interrupted export never leaves a partial
  `<out>/<name>/`,
- a full round-trip contract against `convert-image-stack`: exporting a volume and
  reading the PNG stack back (with `levels` derived mechanically from the sidecar
  manifest, no hand-written mapping) reproduces the **exact same printer-grid
  material-id array** the exporter wrote — fixed by
  `tests/contract/test_print_slices_roundtrip.py` across the default axis mapping, every
  axis-swap/flip variant, an anisotropic-resampling case, and the exact-pitch identity
  case, plus double-run stack-identity stability — see "Round-trip verification" below,
  and
- `convert-image-stack` accepts this exporter's color-label PNG stack directly (the
  `rgb` levels extension, `docs/image-stacks.md`) as a first-class, independently-useful
  input path (e.g. for an externally color-painted label stack), not only as a
  round-trip test fixture.

It does **not** yet claim:

- that GrabCAD Print's Voxel Print Utility actually accepts the generated PNG stack and
  produces a GCVF — that confirmation is Phase 3's job
  (`.devdocs/vision/printer_export/roadmap.md`), and any interpretation this doc states
  as a *default* (axis mapping, background handling, naming) is this implementation's
  choice, not yet an externally verified fact; and
- halftoning/dithering for continuous material blends, named printer presets, or an
  automatic High Speed profile switch (pass `max_materials=3` by hand instead).

## Configuration

`PrintSlicesConfig` JSON, passed via `--config`. As with `generate-primitive-array`, no
CLI field overrides exist — the config file is the only input.

```json
{
  "dpi_x": 600.0,
  "dpi_y": 300.0,
  "layer_thickness_m": 2.7e-05,
  "max_materials": 6,
  "palette": {
    "1": [255, 0, 0],
    "3": [0, 255, 0]
  },
  "background_rgb": [0, 0, 0],
  "printer_x_axis": "x",
  "printer_y_axis": "y",
  "flip_x": false,
  "flip_y": false,
  "flip_z": false,
  "name_prefix": "slice_",
  "index_start": 0,
  "min_slices": 30,
  "max_total_pixels": 4000000000
}
```

| Field | Meaning |
|---|---|
| `dpi_x` / `dpi_y` | printer resolution across/along the fast axis, default 600 / 300 (GrabCAD's PNG-method values) |
| `layer_thickness_m` | slice pitch along the build (z) axis, meters, `> 0`. **Required, no default** — 14e-6 (High Quality) or 27e-6 (High Speed / High Mix) per GrabCAD's guide, must match the printer's own setting |
| `max_materials` | non-background colors allowed per PNG, `1..6` (pass `3` for High Speed) |
| `palette` | **required**, `material_id` (decimal string) → `[r, g, b]` (each `0..255`). Non-background materials only — the exporter rejects a config whose keys do not exactly match the input's non-background material-id set (either direction) |
| `background_rgb` | color for `material_id` 0, default black. Must not collide with any `palette` entry |
| `printer_x_axis` / `printer_y_axis` | which source axis (`"x"` or `"y"`) maps to printer X (fast, `dpi_x`) / Y (slow, `dpi_y`); must differ from each other |
| `flip_x` / `flip_y` / `flip_z` | reverse the corresponding printer-axis index order after sampling |
| `name_prefix` / `index_start` | slice filename prefix and starting index number |
| `min_slices` | derived slice count must be `>= min_slices` (default 30, GrabCAD's stated minimum); relax only for tests |
| `max_total_pixels` | guard on `n_slices * width * height`; exceeding it suggests coarsening the voxel size or cropping the input |

`seed` is inherited from `GeneratorConfig` and reserved: this exporter uses no
randomness, and pixel output does not depend on it (only the config digest does).

## Grid derivation and sampling (`printer/sampler.py`)

Physical pitch, never an integer-ratio resample (the printer pitch is irrational
relative to a typical source voxel size):

```
pitch_x = 0.0254 / dpi_x   (600 dpi -> ~42.33 um)
pitch_y = 0.0254 / dpi_y   (300 dpi -> ~84.67 um)
pitch_z = layer_thickness_m
```

Output cell counts are derived from the source's physical extent along the axis
assigned to each printer axis, with the same `ceil(extent / pitch - 1e-6)` epsilon
convention used elsewhere in this repo (`voxelize-mesh`, primitive arrays):

```
width  = ceil(extent(printer_x_axis) / pitch_x - 1e-6)
height = ceil(extent(printer_y_axis) / pitch_y - 1e-6)
n_slices = ceil(extent(z) / pitch_z - 1e-6)
```

Each output pixel is filled by **nearest-neighbour only**: its physical centre
`(i + 0.5) * pitch` maps to source index `clip(floor(center / src_voxel_size), 0,
src_cells - 1)`, computed once per axis as a 1-D index array and applied with `np.take`
(two per slice — never a Python loop over pixels). `flip_x`/`flip_y`/`flip_z` reverse
these index arrays after they are built; the centre-sampling rule itself never changes.
A boundary tie (a centre landing exactly on a source-cell boundary) resolves via
`floor`, consistently across every axis. When `printer_x_axis == "y"` (axes swapped from
the default), the per-slice extraction transposes accordingly.

## PNG encoding (`image/png.py`)

`write_indexed_png()` writes mode `"P"` (indexed-palette): palette index 0 is
`background_rgb`, indices 1..N are the `palette` config entries in ascending
`material_id` order. No antialiasing, resizing, or ICC handling ever touches the image,
so no intermediate color can appear structurally. Compression parameters are fixed for
a deterministic double-run, but the **encoded PNG bytes may still differ across Pillow
versions/builds** — the contract test's digest pin is on the *decoded pixel array*, not
the file bytes; only a same-environment double-run is required to be byte-identical.
After writing, each PNG is immediately decoded back and its observed palette-index set
is checked to be a subset of the declared set — a defence against a future regression
(e.g. an accidentally added resize step) reintroducing intermediate colors.

## Output layout and manifest

```
<out>/<name>/
├── slice_0000.png … slice_NNNN.png   (zero-padded, >= 4 digits)
└── <name>.printslices.json
```

The export is built in a sibling `<out>/.<name>.tmp-*` directory and published with one
atomic rename; `<out>/<name>/` must not already exist, and any failure during the build
removes the temporary directory and leaves `<out>` untouched.

`<name>.printslices.json` fields:

| Key | Contents |
|---|---|
| `format` / `format_version` | `"vdbmat.print-slices"` / `"1.0.0"` |
| `source` | the input manifest's filename, its own file sha256, and its declared payload sha256 |
| `config_digest` | `PrintSlicesConfig`'s canonical-JSON digest |
| `printer` | `dpi_x`/`dpi_y`/`layer_thickness_m` plus derived pitch in mm |
| `grid` | slice count, pixel dimensions, and physical size in mm on all three axes — the primary reference for catching an axis mix-up (a wrong axis mapping still "looks plausible" but is off by 2x in one dimension) |
| `palette` | `material_id` → `{name, role, rgb}`, names copied from the input manifest — this table is also the **GrabCAD GUI color→material assignment sheet**: after import, match each PNG color to the material name listed here when GrabCAD prompts for a material per detected color |
| `background_rgb` | echoed background color |
| `slices` | `name_prefix`, `index_start`, digit count, and each slice file's own sha256 |

No absolute paths, timestamps, or hostnames are recorded (`docs/determinism.md`).

## Worked example

```bash
uv run vdbmat-utils generate-primitive-array --config primarray.json --out input/ --name demo
uv run vdbmat-utils export-print-slices input/demo.voxels.json \
  --config printslices.json --out out/ --name demo
```

With a 3×2×1 cube primitive array (see `docs/primitive-arrays.md`'s worked example) and
a High Speed (27 µm, `max_materials: 3`) profile, a middle slice through the cubes
decodes to three countable inclusions across, matching `counts_xyz[0] = 3`; the 600 vs.
300 dpi anisotropy makes each square inclusion appear roughly twice as wide as it is
tall in the PNG (pixel aspect, not a distortion of the source data — the manifest's
`grid.physical_mm` records the true physical size). See
`.devdocs/vision/printer_export/p1/report6.md` for the recorded walkthrough and its
measured figures.

## Round-trip verification (`convert-image-stack`)

The exported PNG stack can be read back into a `MaterialLabelVolume` on the printer grid
via `convert-image-stack`'s `rgb` levels (`docs/image-stacks.md`). The `levels`
config needed to do this is derived mechanically from the sidecar manifest — no
hand-written gray/RGB↔material table — by
`vdbmat_utils.printer.roundtrip.image_stack_config_from_print_manifest()`:

- **`levels`**: one `rgb`/`material_id`/`name`/`role` entry per manifest `palette`
  entry, **including the background** (`material_id` 0). This is exactly the
  manifest's `palette` table reshaped into the `image-stacks.md` rgb-levels schema — a
  human reproducing it by hand would copy `palette[id].rgb`, the numeric `id` itself,
  `palette[id].name`, and `palette[id].role` into one `levels` entry per palette key.
- **`voxel_size_xyz_m`**: **recomputed** from the manifest's `printer.dpi_x`/
  `printer.dpi_y`/`printer.layer_thickness_m` via the same `0.0254 / dpi` formula the
  exporter's sampler uses — **not** read back from the manifest's `printer.pitch_*_mm`
  millimetre fields, since round-tripping through a mm conversion risks a float mismatch
  against the sampler's own pitch that this recomputation avoids by construction.
- **`format`**: always `"png"` (this exporter never writes PGM).
- The manifest's `format`/`format_version` are checked to be exactly
  `"vdbmat.print-slices"`/`"1.0.0"`; anything else is a explicit error rather than a
  best-effort read.

```bash
uv run vdbmat-utils export-print-slices input/demo.voxels.json \
  --config printslices.json --out out/ --name demo
# derive the read-back config from out/demo/demo.printslices.json (one-off script or
# vdbmat_utils.printer.roundtrip.image_stack_config_from_print_manifest in a shell),
# write it to roundtrip.json, then:
uv run vdbmat-utils convert-image-stack out/demo --config roundtrip.json \
  --out roundtrip/ --name demo
uv run vdbmat-utils preview-slices roundtrip/demo.voxels.json --axis z
uv run vdbmat-utils material-counts roundtrip/demo.voxels.json
```

The derived `levels` config currently has no CLI subcommand of its own (a library
function only, `printer/roundtrip.py`; CLI wiring is Phase 4 candidate if it turns out
to be needed beyond the contract test and this manual flow). `out/demo` itself is a
valid `convert-image-stack` input directory as-is — the sidecar
`demo.printslices.json` does not match the `*.png` glob, so it does not interfere.

`preview-slices --axis z` on the round-tripped `.voxels.json` reproduces the same
cross-section counts and X:Y pixel-aspect anisotropy described in the worked example
below, now confirmed against the *read-back* array rather than only the freshly-sampled
one — see `.devdocs/vision/printer_export/p2/report6.md` for a recorded run and
`material-counts` cross-check against the export summary's per-material pixel counts.

## Downstream hand-off

GrabCAD Print does not read the sidecar manifest — it is this repository's own
provenance record and the human-facing color→material assignment sheet described above.
Import the PNG stack via GrabCAD Print's voxel-printing PNG method, matching the
printer's own dpi/layer-thickness settings to this config's `dpi_x`/`dpi_y`/
`layer_thickness_m`. Phase 3 of `.devdocs/vision/printer_export/roadmap.md` covers
confirming GCVF generation against GrabCAD Voxel Print Utility itself; until that lands,
treat axis-mapping and background-handling defaults documented here as this
implementation's choice, not a verified GrabCAD-side fact.
