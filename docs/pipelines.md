# Config-driven pipelines (`apply-pipeline`)

`apply-pipeline` loads one or more existing `.voxels.json` assets, applies a
configured sequence of label-safe operations, and writes a contract-valid
asset — no user Python required.

```bash
uv run vdbmat-utils apply-pipeline --config pipeline.json --out out/ --name result
uv run vdbmat-utils apply-pipeline --config pipeline.json --out out/ --name result --dry-run
```

`--dry-run` validates the whole configuration and prints the resolved step
plan without reading any payload or writing any output.

## Schema (`PipelineConfig`)

A pipeline is a flat, single-assignment (SSA-style) step list — no nesting,
no conditionals, no loops (deliberately; see ADR 0009):

```json
{
  "inputs": [
    {"id": "base",    "manifest_path": "base.voxels.json"},
    {"id": "overlay", "manifest_path": "overlay.voxels.json"}
  ],
  "steps": [
    {"op": "remap-materials", "from": "overlay",
     "mapping": {"1": 2}, "definitions": {"2": {"name": "white-resin"}},
     "as": "overlay_white"},
    {"op": "compose", "base": "base", "overlay": "overlay_white",
     "mode": "union", "as": "composed"}
  ],
  "output": {"ref": "composed"}
}
```

- `inputs` — binds ids to manifest paths. **Relative paths resolve against
  the config file's directory**; absolute paths are allowed.
- `steps` — each step names its op, reads bound ids (`"from"` for unary
  ops; `"base"`/`"overlay"` for `compose`; `"from"`/`"mask"` for
  `apply-mask`), takes op parameters inline, and binds its result to a
  fresh id with `"as"`.
- `output` — `{"ref": "<id>"}` names the volume to write.

Registered ops and their parameters: `crop` (`min_zyx`, `max_zyx`), `pad`
(`before_zyx`, `after_zyx`, `fill_material_id`), `resample` (`factor_zyx` or
`voxel_size_xyz_m`), `orient` (`steps`, `update_transform`), `place`
(`local_to_world`, `compose_with_existing`), `apply-mask` (`mode`,
`fill_material_id`), `compose` (`mode`), `remap-materials` (`mapping`,
`definitions`, `prune_palette`). Semantics: `docs/volume-ops.md`.

## Validation is fail-fast

All structural problems are reported **before any array work**, each naming
the offending step index: unknown op names, unknown or missing parameters,
references to unbound ids, rebinding an existing id, malformed
`inputs`/`output`, and inputs that no step (or the output) ever uses.
Runtime op failures are wrapped with `step N (op-name):` context.

## Determinism and provenance

Generator `vdbmat-utils.pipeline` v0.1.0. Provenance `sources` are the
SHA-256 digests of every input **manifest file** in input order, pinning the
actual referenced content; the configuration digest covers the canonical
JSON of the config *as written* (including the path strings), so provenance
is honest about what was referenced. The asset identity hashes sources plus
config digest (the shared generator recipe). Double runs are byte-equal
(contract-tested).

## Deferred (ADR 0009)

Generator steps inside pipelines (`morph-stack`/`convert-image-stack` as
steps — pipelines currently start from existing assets), conditionals,
variables, caching, and parallel execution.
