# vdbmat-utils

Generators and converters that produce material-labeled voxel inputs for
[vdbmat](https://github.com/iwaag/vdbmat). Every workflow in this repository emits the same
interchange contract — a `<name>.voxels.json` manifest plus a `<name>.material_id.npy` payload —
validated against the pinned `vdbmat` version.

## Status

Phase 0 (repository and contract foundation) complete: canonical construction helpers, a
deterministic asset writer, golden-fixture contract tests against the pinned `vdbmat`, and the
`inspect` / `validate` / `generate-fixture` CLI. Next: Phase 1 (mesh and image-stack conversion
workflows). Plans and reports live in `.devdocs/vdbmat-utils/` of the parent `pj-voxel3dprint`
repository; decisions in `docs/adr/`.

## Installation

This package depends on `vdbmat`, which is not published to PyPI. It is consumed as a local path
dependency on the sibling checkout, pinned by the `pj-voxel3dprint` superproject's submodule
commits (see `docs/adr/0001-vdbmat-dependency-pinning.md`).

```bash
git clone --recurse-submodules https://github.com/iwaag/pj-voxel3dprint.git
cd pj-voxel3dprint/vdbmat-utils
uv sync            # minimal install: numpy + vdbmat + dev tools
```

Optional extras (`mesh`, `image`, `vdb`, `preview`) are reserved for later phases and currently
empty.

## Usage

```bash
# write a deterministic synthetic asset
uv run vdbmat-utils generate-fixture multimaterial -o out/

# metadata-only view (add --json for machine-readable output)
uv run vdbmat-utils inspect out/multimaterial.voxels.json

# full contract validation (payload checksum, palette, transform, schema range)
uv run vdbmat-utils validate out/multimaterial.voxels.json

# hand off to vdbmat
uv run vdbmat import-voxels out/multimaterial.voxels.json out/multimaterial.zarr
```

Exit codes: 0 success, 1 validation/generation failure, 2 usage error.
Fixture presets: `anisotropic`, `transformed`, `multimaterial` (see
`vdbmat_utils.fixtures`); outputs are byte-deterministic per `docs/determinism.md`.

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
