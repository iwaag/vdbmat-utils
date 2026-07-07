# Optical Mappings For Formations

vdbmat has a small built-in material table (`air`, diagnostic axis materials,
black/white resin, transparent resin). A procedural formation usually names
domain materials such as `calcite-matrix` or `quartz-grain`; those names require
a companion `vdbmat.optical-mapping` document.

`generate-formation` compares the palette to vdbmat's public built-in table. If
any palette name is not built in, `mapping.materials` must cover exactly those
non-built-in names. Built-in entries are filled from vdbmat; user-supplied
coefficients are passed through verbatim.

```json
{
  "mapping": {
    "configuration_id": "phase3-marble-like-materials-v1",
    "calibration_status": "provisional-uncalibrated",
    "materials": [
      {
        "name": "calcite-matrix",
        "sigma_a_rgb_per_m": [15.0, 12.0, 10.0],
        "sigma_s_rgb_per_m": [180.0, 170.0, 160.0],
        "g": 0.0,
        "ior": 1.5
      }
    ]
  }
}
```

The emitted mapping is written through vdbmat's public optics API and its digest
is printed by the CLI and recorded in the manifest `source.notes`.

End-to-end handoff:

```bash
uv run vdbmat-utils generate-formation --config formation.json --out out --name rock
uv run vdbmat-utils validate out/rock.voxels.json
uv run vdbmat import-voxels out/rock.voxels.json out/rock.zarr
uv run vdbmat mapping-digest out/rock.optical-mapping.json
uv run vdbmat convert out/rock.zarr out/rock-optical.zarr \
  --mapping-file out/rock.optical-mapping.json
```
