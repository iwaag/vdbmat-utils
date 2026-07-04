# ADR 0002: canonical spelling is `vdbmat` / `vdbmat-utils`

Date: 2026-07-05
Status: accepted

## Context

The roadmap's original decision log declared `vbdmat` canonical, but every artifact on disk —
GitHub repositories `iwaag/vdbmat` and `iwaag/vdbmat-utils`, the Python package name in
`vdbmat/pyproject.toml`, the CLI command, and the submodule paths — spells it `vdbmat`
(from OpenVDB + material). The `vbdmat` entry was itself a typo.

## Decision

`vdbmat` / `vdbmat-utils` everywhere: distribution name `vdbmat-utils`, import package
`vdbmat_utils`, CLI `vdbmat-utils`, devdocs paths `.devdocs/vdbmat-utils/`. Renaming the
repositories to match a typo would churn every existing reference for no benefit.

The roadmap decision log has been corrected accordingly (superseding entry dated 2026-07-05).
Deprecated spellings: `vbdmat`, `vdmat-utils`.
