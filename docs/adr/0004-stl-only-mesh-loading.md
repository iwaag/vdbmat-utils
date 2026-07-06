# ADR 0004: STL-only, dependency-free mesh loading for Phase 1

Date: 2026-07-06
Status: accepted

## Context

Phase 1 needs a mesh input path. The roadmap calls for "a deliberately narrow set of mesh
formats first", and Principle 5 keeps the base install dependency-free beyond numpy + vdbmat.
A complete, debugged STL reader already existed in vdbmat and was deleted; it is recoverable
from git history (`git show 8f55562:src/vbdmat/io/mesh.py`).

## Options

1. **Port the recovered dependency-free STL reader (chosen).** Binary and ASCII STL,
   ~140 lines, zero third-party dependencies, already handles the classic "binary file that
   starts with `solid`" trap by trusting the exact binary length over the marker token.
2. **Adopt a mesh library (trimesh, meshio, numpy-stl).** Buys OBJ/PLY/glTF for free but
   drags a dependency tree into the `mesh` extra, imports someone else's parsing semantics
   (repair heuristics, silent winding fixes) under our topology contract, and re-derives
   behavior the recovered tests already pin.

## Decision

Option 1. `vdbmat_utils.mesh.loader` is the recovered reader, renamed into the utils error
hierarchy (`MeshReadError`). The `mesh` extra stays **empty** — the mesh workflow works in
the minimal install. OBJ/PLY/glTF are explicitly out of scope for Phase 1; a future format
addition is a new ADR, most likely as a real dependency behind the `mesh` extra rather than
more hand-rolled parsers.

## Consequences

- Loader behavior is fully owned and golden-testable; no upstream parser drift.
- Users with non-STL meshes must convert externally for now (any DCC tool exports STL).
- The binary-detection-by-exact-length rule is contract: a `solid`-prefixed binary file whose
  length matches its declared triangle count parses as binary, matching the old suite.
