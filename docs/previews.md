# Previews and diagnostics

`vdbmat_utils.preview` inspects generated assets without OpenVDB, matplotlib, or any other
optional dependency — NumPy only, available in the minimal install. Every conversion workflow
prints the material-count summary on success, and the same functions are exposed as CLI
commands for any existing asset.

## Material counts

```bash
uv run vdbmat-utils material-counts out/bracket.voxels.json          # human-readable
uv run vdbmat-utils material-counts out/bracket.voxels.json --json   # {"0": 160, "1": 32}
```

API: `material_counts(volume) -> dict[int, int]` — voxel count per material id, including
background. Useful for conservation checks (do the counts match the source data?) and for
spotting empty or runaway materials.

## Slice previews

```bash
# ASCII to stdout; --index defaults to the middle slice
uv run vdbmat-utils preview-slices out/bracket.voxels.json --axis z --index 2

# grayscale PGM instead
uv run vdbmat-utils preview-slices out/bracket.voxels.json --axis y --out slice.pgm
```

ASCII output prints one character per voxel — `.` for background-role materials, `0-9a-z…`
cycling by material id — under a legend line such as `slice z=2  +x →  +y ↓` that states the
in-plane axis directions, so a transposed or flipped volume is visible to the eye. This is
the project's primary axis-orientation diagnostic: the contract suites golden-test these
strings on asymmetric fixtures (`tests/contract/test_mesh_contract.py::test_orientation_goldens`).

`slice_pgm` writes the same slice as a PGM image, mapping each material id to a distinct
deterministic gray level, with no external libraries.

API: `slice_ascii(volume, axis, index) -> str`, `slice_pgm(volume, axis, index, path)`,
with `axis` one of `"z" | "y" | "x"`.
