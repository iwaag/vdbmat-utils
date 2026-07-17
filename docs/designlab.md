# designlab

`designlab` is a browser GUI (via [viser](https://github.com/nerfstudio-project/viser))
that fills in a generator config form, saves/loads it, and runs the full
one-shot pipeline — generate → optical mapping → publish — to produce a
canonical run bundle the existing
[`mitsuba_stage_viewer.py`](../../vdbmat/examples/pipeline_run/demo/mitsuba_stage_viewer.py)
can pick up from its Input catalog. It exists to remove the "hand-write a
config JSON, then chain three CLI calls" friction of trying one input
model, without inventing a second config format: the file a config
dropdown saves/loads *is* the file the pinned CLI accepts.

Phase 2 registers exactly one generation method, **primitive array**
(`vdbmat-utils generate-primitive-array` — see
[`primitive-arrays.md`](primitive-arrays.md)), the minimal generator that
exists specifically to exercise this whole pipeline before a method with
real config complexity gets GUI-wired. See
`.devdocs/vision/designlab/roadmap.md` (in the parent `pj-voxel3dprint`
repository) for the phase sequence beyond this.

## Launching

From `vdbmat-utils`:

```bash
uv run --group designlab python examples/designlab/designlab_app.py -- \
  --config-root <CONFIG_ROOT> \
  --output-root <OUTPUT_ROOT> \
  [--work-root <WORK_ROOT>] \
  [--port 8081]
# then open http://127.0.0.1:8081
```

- `--config-root`: where the config catalog (dropdown + Refresh + Load/Save)
  scans and writes. Must already exist.
- `--output-root`: where published run bundles land. Point
  `mitsuba_stage_viewer.py --input-root` at the same directory to see them
  in its Input catalog.
- `--work-root`: scratch space for in-progress jobs. Defaults to an
  `--output-root` sibling, `<basename>.designlab-work`; created if missing.
  Must not be inside `--output-root` (the viewer's catalog walks
  dot-directories too, so a work directory left under the output root could
  be mistaken for a published bundle).
- The `designlab` dependency group pulls in `viser`; the console script
  runs in the same venv as `vdbmat-utils` and `vdbmat` (the latter is an
  editable path dependency), which is required — every generation/mapping
  stage runs as a subprocess of *this* interpreter
  (`sys.executable -m vdbmat_utils.cli.main ...` /
  `sys.executable -m vdbmat.cli.main ...`). A stray `vdbmat_utils`/`vdbmat`
  import failure at startup means you're not running inside that venv;
  the app exits with a message pointing at the `uv run --group designlab`
  form above.

## Form ↔ config

Primitive array's `PrimitiveArrayConfig` is fully flat, so every field gets
a widget — no JSON-text hybrid exists for this method (see
`primitive-arrays.md` for field semantics):

| Config field | Widget |
|---|---|
| `voxel_size_xyz_m` | 3 number inputs |
| `primitive` | dropdown (`cube` / `sphere`) |
| `counts_xyz` | 3 number inputs |
| `primitive_size_m` / `gap_m` / `margin_m` | number inputs |
| `base_material_name` / `inclusion_material_name` | dropdowns (`vdbmat_utils.primitives.types.BUILTIN_MATERIAL_IDS` keys) |
| `max_axis_cells` / `max_total_cells` / `seed` | number inputs, under an "Advanced" sub-folder |

**name** (the job's asset name, e.g. `demo`) is a separate text field, not a
config field — matching the existing CLI's `--name`, which is likewise
outside the config. It must match `[a-z0-9][a-z0-9-]*`.

A field-level validation failure (e.g. `primitive_size_m <= 0`) surfaces in
the status line as `PrimitiveArrayConfig`'s own error, unmodified — the GUI
never reinterprets or rewords a config or CLI error.

## Config save / load

- The catalog lists every `*.primarray.json` file under `--config-root`
  (recursive). No browser upload path exists; every config is a
  server-local file placed under `--config-root` directly.
- **Load** parses the selected file and pushes its values into the form.
  An unknown field or invalid value is shown exactly as
  `GeneratorConfig.from_json` raised it.
- **Save** writes the current form as canonical JSON
  (`GeneratorConfig.to_json()`) to `<config-root>/<name>.primarray.json`.
  It **refuses to overwrite an existing file** — pick a different name.
  There is no GUI-only persisted state; the config file is the only saved
  artifact.

## Generate: STAGE vocabulary

Clicking Generate runs one transaction through five named stages, printed
both in the status line and as `STAGE <transaction> <stage> <elapsed_s>`
lines on stdout (the same convention `mitsuba_stage_viewer.py` uses):

| Stage | What happens |
|---|---|
| `validate` | Name/root checks, publish-target collision check. The config itself was already validated by construction (`form_to_config`/`from_json` raise on invalid fields before this ever runs). |
| `generate` | `vdbmat-utils generate-primitive-array` subprocess, writing into the job's work directory. |
| `map` | A hand-built `vdbmat.pipeline-config` (`input.kind=direct-voxel`, `mapping.name=phase0-provisional-materials-v1`, `execution.random_seed=0`) is run via the `vdbmat run` subprocess. |
| `verify` | `vdbmat validate <bundle> --json` on the freshly built bundle. |
| `publish` | Atomic `os.replace` of the verified bundle into `--output-root`. |

This differs from the roadmap's five-stage sketch
(`validate → generate → import → map → publish`): per ADR-009 D1, `vdbmat
run` consumes the direct-voxel manifest directly and persists
`material.zarr` itself in its own internal stage, so a standalone
`import-voxels` call would only produce a zarr store no bundle ever
references. This module's `map` stage folds that step in; see
`.devdocs/vision/designlab/p2/plan.md` for the full rationale.

Any failure before `publish` leaves `--output-root` untouched — the status
line names the failing stage and shows the failing subprocess's own stderr
tail, unedited. The job's work directory is deleted on success and left in
place on failure, for inspection.

## Publish naming, collision, and reuse

A bundle publishes as `<output-root>/<method_id>-<name>-<digest12>`, e.g.
`primitive-array-demo-3fa9c2d81b04`, where `digest12` is the first 12 hex
characters of the config's SHA-256 digest. This keeps the name a pure,
deterministic function of what was generated, so it's safe to delete by
hand and two different configs (even under the same `name`) never collide.

If the publish target already exists:

- and it's a valid bundle (has `run.json` and `optical.zarr`) — generation
  is skipped and the existing bundle is reused (a deterministic generator
  plus the same mapping means identical scientific content; this is a
  cache hit, not a re-check).
- and it's *not* a valid bundle (partial/corrupt) — Generate fails with an
  explicit error naming the path. **There is no automatic deletion or
  overwrite of anything under `--output-root`.** Remove the offending
  directory by hand and retry.

A digest change from editing any field — including `seed`, which this
generator ignores scientifically but which is still part of every config's
canonical JSON — publishes to a *new* path rather than reusing or
overwriting the old one. Cleaning up accumulated digest variants is a
Phase 6 (generation management) concern; for now, deleting old publish
directories by hand is expected.

## Work-root cleanup

- On startup, every entry directly under `--work-root` is removed (it is
  designlab's own exclusive scratch space, so on process start nothing
  can legitimately still be running there).
- Generate jobs run one at a time on a single worker thread; a Generate
  click while a job is already running is rejected with a status message
  (no queue, no cancel — see the roadmap's risk notes on why cancel isn't
  built yet).

## GUI = CLI reproduction contract

A config `designlab` saved (or built from the form and used directly) and
name `N`, run through the plain CLI —

```bash
uv run vdbmat-utils generate-primitive-array --config C --out D --name N
```

— produces an `N.material_id.npy` whose SHA-256 matches the
`input_payload_sha256` recorded in the published bundle's `run.json`. This
is fixed by an integration test
(`tests/integration/test_designlab_pipeline.py::test_gui_saved_config_reproduces_bundle_payload_digest`),
the same "config file is portable, byte-identical input" idea as the
Mitsuba viewer's stage-preset/session replay contracts.

## Registry: adding a second method

A generation method is a `designlab_registry.GeneratorMethod`:

```python
GeneratorMethod(
    method_id="...",          # publish-name prefix
    title="...",              # dropdown display name
    config_suffix=".....json",  # catalog discriminator
    config_cls=SomeGeneratorConfig,
    build_form=build_form_fn,       # (gui) -> FormBinding
    form_to_config=form_to_config_fn,  # (FormBinding) -> SomeGeneratorConfig
    config_to_form=config_to_form_fn,  # (FormBinding, SomeGeneratorConfig) -> None
    generator_argv=generator_argv_fn,  # (config_path, out_dir, name) -> argv
)
```

`FormBinding` is any object whose attributes each expose a mutable
`.value` — `build_form` is the only function in the registry that touches
viser (imported lazily inside it), so `form_to_config`/`config_to_form` are
unit-testable with a plain test double instead of a real widget (see
`tests/unit/test_designlab_registry.py`).

To add the next method (`voxelize-mesh`, per the roadmap's Phase 3):

1. Write `_build_voxelize_mesh_form`, `_voxelize_mesh_form_to_config`,
   `_voxelize_mesh_config_to_form`, `_voxelize_mesh_argv` next to the
   primitive-array equivalents in `designlab_registry.py`, following
   `MeshVoxelizeConfig`'s existing fields.
2. Register the resulting `GeneratorMethod` by appending it to `REGISTRY`.
3. Extend `tests/unit/test_designlab_registry.py` with the same shape of
   coverage as the primitive-array cases (argv golden, form round-trip,
   error transparency).
4. `voxelize-mesh` takes an input file (an STL mesh) that primitive array
   doesn't — Phase 3's actual new work is a small, registry-independent
   asset catalog (`--asset-root`, same containment convention as the
   config catalog) that a form field can point at, not anything in
   `designlab_pipeline.py` or `designlab_jobs.py`, both of which are
   already generic across methods (`run_generate_job` never special-cases
   `PrimitiveArrayConfig`).
5. `designlab_app.py` currently builds the sole registered method's form
   unconditionally; a second entry needs its dropdown's `on_update` wired
   to rebuild the form folder for the newly selected method (not needed
   while `REGISTRY` has one entry).

No change to `designlab_pipeline.py`'s stage sequence, publish-naming
rule, or `designlab_jobs.py` is expected for a second flat-config method —
that is the point of having exercised the whole pipeline against a
deliberately trivial first method.
