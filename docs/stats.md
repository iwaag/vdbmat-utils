# Formation Statistics

`vdbmat-utils formation-stats MANIFEST` computes the same metrics used after
`generate-formation`. It works on any material-label manifest, not only procedural
outputs.

```bash
uv run vdbmat-utils formation-stats out/marble-like.voxels.json
uv run vdbmat-utils formation-stats out/marble-like.voxels.json \
  --constraints examples/formation_generation/marble-like.formation.json \
  --out out/marble-like.stats.json
```

For each palette material the report includes:

- `count` and `volume_fraction`.
- `local_thickness_m`: min, p05, p50, p95, max.
- `component_count`: 6-connected components.
- `largest_component_fraction`.
- `min_printable_thickness_m`.

Local thickness is the Phase 3 proxy `2 * EDT(mask)` sampled at material voxels,
using anisotropic voxel spacing. It is a conservative inscribed-distance proxy,
not full sphere-fitting local thickness.

Supported constraint forms:

```json
{"kind":"volume-fraction","material_id":1,"min":0.1,"max":0.6}
{"kind":"min-feature-size","material_id":1,"threshold_m":0.001}
{"kind":"connected","material_id":1,"mode":"single-component"}
{"kind":"connected","material_id":1,"max_components":8}
{"kind":"min-largest-component-fraction","material_id":1,"min":0.95}
{"kind":"min-printable-thickness","material_id":1,"threshold_m":0.001}
```

Constraints are measured after generation. Phase 3 does not iterate, repair, or
search for a satisfying formation. `generate-formation --strict` exits with code
1 when any declared constraint fails.
