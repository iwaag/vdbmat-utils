# ADR 0001: vdbmat dependency via local path + superproject submodule pin

Date: 2026-07-05
Status: accepted

## Context

`vdbmat-utils` must build against `vdbmat`, which is not published to PyPI (roadmap Phase 0).
The pin must be explicit, bumped deliberately, and used by contract-compatibility tests.
Both repositories are already submodules of the `pj-voxel3dprint` superproject, checked out as
siblings (`pj-voxel3dprint/vdbmat`, `pj-voxel3dprint/vdbmat-utils`).

## Options

1. **Local path dependency + submodule pin (chosen).**
   `[tool.uv.sources] vdbmat = { path = "../vdbmat", editable = true }`. The effective pin is the
   `vdbmat` submodule commit recorded in `pj-voxel3dprint`; bumping it is an explicit superproject
   commit. CI checks out the superproject with submodules so tests always run against the pinned
   commit.
2. **Git URL dependency with commit pin in `pyproject.toml`.**
   Works standalone but duplicates the pin already held by the superproject, so the two can drift;
   every bump needs coordinated edits in two places, and local development against uncommitted
   `vdbmat` changes requires overrides.

## Decision

Option 1. A single pin location (the superproject submodule gitlink), editable installs for
cross-repo development, and no network access needed at sync time.

## Consequences

- `vdbmat-utils` is developed inside the `pj-voxel3dprint` superproject; a standalone clone must
  place a `vdbmat` checkout at `../vdbmat`.
- Bumping `vdbmat` = updating the submodule commit in `pj-voxel3dprint` and re-running the
  contract test suite; a failing suite blocks the bump.
- If `vdbmat` is ever published or the repos decouple, revisit with a git-URL or index dependency.
