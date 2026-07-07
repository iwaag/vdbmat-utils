# ADR 0010: procedural determinism

Date: 2026-07-07
Status: accepted

## Context

Phase 3 fields must remain reproducible under double runs, domain extension, and
future chunking. Sequential RNG draws would make values depend on traversal order
and domain bounds.

## Decision

Procedural lattice randomness uses `procgen.hashing.hash_lattice(ix, iy, iz,
stream_id, seed) -> uint64`, a fixed SplitMix64-style mix over explicit `uint64`
operations. Derived draws use `hash_derive` with fixed salts. Stream ids are
allocated per layer from `10_000 + layer_index * 100`, with local offsets for
octaves and roles.

Noise gradients, Worley jitter, site ids, and grain material picks derive from
coordinate hashes. Dense field math uses float64 NumPy expressions with fixed
ordering.

## Consequences

The same world lattice point has the same random value for a fixed seed and
stream, independent of evaluation order or domain bounds. Golden hash tests,
domain-extension tests, and Phase 3 payload goldens pin the behavior. Changing
hash constants is a breaking change to all procedural reference data.
