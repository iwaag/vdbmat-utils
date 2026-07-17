# vdbmat-utils

Generators and converters that produce material-labeled voxel inputs for
[vdbmat](https://github.com/iwaag/vdbmat). Every workflow in this repository emits the same
interchange contract — a `<name>.voxels.json` manifest plus a `<name>.material_id.npy` payload —
validated against the pinned `vdbmat` version.

## Status

Phases 0–3 complete: mesh voxelization (`voxelize-mesh`), image-stack conversion
(`convert-image-stack`), key-slice morphing (`morph-stack`), config-driven volume-op pipelines
(`apply-pipeline`), procedural formations (`generate-formation`, `formation-stats`,
`sweep-formation`), previews and diagnostics (`preview-slices`, `material-counts`), a
transparent-block + opaque cube/sphere inclusion-array generator
(`generate-primitive-array`, `docs/primitive-arrays.md`), plus the Phase 0
`inspect` / `validate` / `generate-fixture` CLI. All are deterministic and
contract-tested against the pinned `vdbmat`. `examples/designlab/` is a
browser GUI (`docs/designlab.md`) that fills in a generator config form
and runs generate → optical mapping → publish as one job; it currently
wires up `generate-primitive-array` only. Plans and reports live in
`.devdocs/vdbmat-utils/` of the parent `pj-voxel3dprint` repository; decisions in `docs/adr/`.

## Installation

This package depends on `vdbmat`, which is not published to PyPI. It is consumed as a local path
dependency on the sibling checkout, pinned by the `pj-voxel3dprint` superproject's submodule
commits (see `docs/adr/0001-vdbmat-dependency-pinning.md`).

```bash
git clone --recurse-submodules https://github.com/iwaag/pj-voxel3dprint.git
cd pj-voxel3dprint/vdbmat-utils
uv sync            # minimal install: numpy + vdbmat + dev tools
```

The minimal install covers STL voxelization, PGM image stacks, and all previews. The `image`
extra (`uv sync --extra image`) adds PNG slice input; `mesh` and `preview` are deliberately
empty, `vdb` is reserved for Phase 5.

## Usage

### Mesh workflow (`docs/voxelization.md`)

```bash
# config: source_unit (required), voxel_size_xyz_m, material block, ...
uv run vdbmat-utils voxelize-mesh part.stl --config mesh.json --out out/ --name part

# inspect the result visually (works without OpenVDB/matplotlib)
uv run vdbmat-utils preview-slices out/part.voxels.json --axis z
uv run vdbmat-utils material-counts out/part.voxels.json
```

Accepts one watertight, consistently oriented STL solid (binary or ASCII); invalid topology,
missing units, or oversized grids fail with a one-line diagnostic.

### Image-stack workflow (`docs/image-stacks.md`)

```bash
# slices/: PGM (or PNG with the image extra), stacked in filename order as z
uv run vdbmat-utils convert-image-stack slices/ --config stack.json --out out/ --name stack
```

Every gray value must be declared in the config's `levels`; gaps in numbered sequences and
shape mismatches are errors.

### Morph workflow (`docs/morphing.md`)

```bash
# slices/: sparse labeled key slices; the filename number IS the z index
# (slice_0000.pgm, slice_0008.pgm, ...); gaps are interpolated per-label
# through signed distance fields — material ids are never averaged.
uv run vdbmat-utils morph-stack slices/ --config morph.json --out out/ --name part
```

### Pipeline workflow (`docs/pipelines.md`)

```bash
# pipeline.json: inputs (existing .voxels.json assets) + a flat step list
# (crop, pad, resample, orient, place, apply-mask, compose, remap-materials)
uv run vdbmat-utils apply-pipeline --config pipeline.json --out out/ --name result
uv run vdbmat-utils apply-pipeline --config pipeline.json --out out/ --name result --dry-run
```

Volume-operation semantics (geometry rules, boolean modes, conservation claims):
`docs/volume-ops.md`.

### Procedural formation workflow (`docs/procedural.md`)

```bash
uv run vdbmat-utils generate-formation \
  --config examples/formation_generation/marble-like.formation.json \
  --out out/marble --name marble-like --strict
uv run vdbmat-utils formation-stats out/marble/marble-like.voxels.json
uv run vdbmat-utils sweep-formation \
  --config examples/formation_generation/tiny-sweep.sweep.json \
  --out out/sweep --name tiny
```

Formation configs compose host, strata, veins, grains, pores, and fractures as
discrete material labels. Non-built-in material names emit a companion
`vdbmat.optical-mapping` document; see `docs/optical-mappings.md`. Metrics and
constraint forms are in `docs/stats.md`.

### Primitive-array workflow (`docs/primitive-arrays.md`)

```bash
# config: voxel_size_xyz_m, primitive (cube|sphere), counts_xyz, primitive_size_m,
# gap_m, margin_m (both required, no default), base/inclusion material names
uv run vdbmat-utils generate-primitive-array --config primarray.json --out out/ --name demo
```

Builds a transparent base block containing an A × B × C grid of opaque cube or sphere
inclusions from one flat config with no input files; grid shape is always derived, never
given directly. An optical test pattern for designlab's pipeline, not a print-process
simulation.

### designlab: a browser GUI for building inputs (`docs/designlab.md`)

```bash
uv run --group designlab python examples/designlab/designlab_app.py -- \
  --config-root <CONFIG_ROOT> --output-root <OUTPUT_ROOT> --port 8081
# then open http://127.0.0.1:8081
```

A viser GUI that fills in a generator config form, saves/loads it, and runs
generate → optical mapping → publish as one background job, so a candidate
input model can be tried without hand-writing config JSON or chaining CLI
calls. Point `--output-root` at an existing `mitsuba_stage_viewer.py`
`--input-root` to see published bundles in its Input catalog. Phase 2
registers one method, `generate-primitive-array`; see `docs/designlab.md`
for the registry interface used to add the next one.

### Validation and hand-off

```bash
uv run vdbmat-utils validate out/part.voxels.json
uv run vdbmat import-voxels out/part.voxels.json out/part.zarr
uv run vdbmat convert out/part.zarr out/part-optical.zarr
```

Exit codes: 0 success, 1 validation/generation failure, 2 usage error.
Synthetic fixture presets (`generate-fixture`): `anisotropic`, `transformed`, `multimaterial`
(see `vdbmat_utils.fixtures`); all outputs are byte-deterministic per `docs/determinism.md`.

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Layout: `src/vdbmat_utils/` (package), `tests/{unit,contract,integration}/`, `docs/adr/`
(architecture decision records), `examples/`.

Dependency rule: modules depend toward `vdbmat_utils.core`; only `vdbmat`'s public API
(`vdbmat.core`, `vdbmat.io.voxel_manifest`, the `vdbmat` CLI) may be imported — never
underscore-prefixed internals.
