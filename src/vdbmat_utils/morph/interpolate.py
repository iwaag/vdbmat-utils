"""Per-label SDF interpolation between key slices (plan D4).

For each pair of consecutive key slices and each material id present in the
stack, the 2-D signed distance fields of the label's masks are interpolated
linearly in z; each output pixel takes the most-inside label (argmin of
distance, ties to the lowest material id), falling back to the configured
background where no label is inside. Key slices are emitted verbatim.
Topology changes (merge/split/appear/disappear) are emergent. Memory stays
O(slice area): one running (min-distance, argmin) pair per output slice.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialLabelVolume

from vdbmat_utils.core import (
    GeneratorConfig,
    build_material_label_volume,
    build_provenance,
)
from vdbmat_utils.fields import signed_distance
from vdbmat_utils.morph import MorphError
from vdbmat_utils.morph.keyslices import KeySlices, load_key_slices

GENERATOR = "vdbmat-utils.morph.stack"
GENERATOR_VERSION = "0.1.0"

_FORMATS = ("pgm", "png")
_EDGE_POLICIES = ("error", "clamp")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MorphStackConfig(GeneratorConfig):
    """Configuration for the morph-stack workflow.

    ``levels`` follows the image-stack contract (one shared table for all key
    slices). ``z_count`` defaults to the last key index + 1; extending beyond
    the key range is governed by ``edge_policy`` ("error" rejects, "clamp"
    repeats the nearest key slice — never linear extrapolation). ``seed`` is
    inherited, unused, and reserved.
    """

    voxel_size_xyz_m: tuple[float, float, float]
    levels: tuple[Mapping[str, object], ...]
    background: int = 0
    z_count: int | None = None
    edge_policy: str = "error"
    local_to_world: tuple[tuple[float, ...], ...] | None = None
    format: str = "pgm"
    max_axis_cells: int = 256
    max_total_cells: int = 8_000_000


def _interpolate_gap(
    volume: npt.NDArray[np.uint16],
    low_layer: npt.NDArray[np.uint16],
    high_layer: npt.NDArray[np.uint16],
    z_low: int,
    z_high: int,
    material_ids: list[int],
    spacing: tuple[float, float],
) -> None:
    """Fill the open interval ``(z_low, z_high)`` of ``volume`` in place.

    Labels are visited in ascending id with a strict ``<`` update on a
    running (min-distance, label) pair per output slice, so ties go to the
    lowest material id and per-label memory stays two float64 slices (plan
    §7). Pixels where no label is inside (distance < 0) keep the background.
    """
    gap = range(z_low + 1, z_high)
    shape = low_layer.shape
    best = np.full((len(gap), *shape), np.inf, dtype=np.float64)
    for material_id in material_ids:
        low_mask = low_layer == material_id
        high_mask = high_layer == material_id
        if not low_mask.any() and not high_mask.any():
            continue
        low = signed_distance(low_mask, spacing)
        high = signed_distance(high_mask, spacing)
        for offset, z in enumerate(gap):
            t = (z - z_low) / (z_high - z_low)
            # IEEE: (1-t)*inf + t*finite stays +inf (absent label). The one
            # indeterminate case, -inf + +inf (label fills one key slice and
            # is absent from the other), yields NaN and resolves to +inf:
            # with no boundary on either side there is nothing to move.
            combined = (1.0 - t) * low + t * high
            combined = np.where(np.isnan(combined), np.inf, combined)
            inside = combined < np.minimum(best[offset], 0.0)
            best[offset] = np.where(inside, combined, best[offset])
            volume[z] = np.where(inside, np.uint16(material_id), volume[z])


def morph_stack(slices_dir: Path, config: MorphStackConfig) -> MaterialLabelVolume:
    """Interpolate a sparse labeled key-slice set into a full label volume."""
    if config.format not in _FORMATS:
        raise MorphError(
            f"config.format: unsupported format {config.format!r}; "
            f"expected one of {', '.join(_FORMATS)}"
        )
    if config.edge_policy not in _EDGE_POLICIES:
        raise MorphError(
            f"config.edge_policy: must be one of {', '.join(_EDGE_POLICIES)}, "
            f"got {config.edge_policy!r}"
        )

    key = load_key_slices(
        slices_dir, levels=tuple(config.levels), image_format=config.format
    )
    if config.background not in key.palette.material_ids:
        raise MorphError(
            f"config.background: material_id {config.background} is not "
            "declared in config.levels"
        )

    last_key = key.z_indices[-1]
    z_count = last_key + 1 if config.z_count is None else config.z_count
    if z_count < last_key + 1:
        raise MorphError(
            f"config.z_count: {z_count} is smaller than the last key-slice "
            f"index + 1 ({last_key + 1})"
        )
    if config.edge_policy == "error":
        if key.z_indices[0] != 0:
            raise MorphError(
                f"slices: first key slice is z={key.z_indices[0]}, not 0; "
                'slices before it would be extrapolation — use edge_policy '
                '"clamp" to repeat the nearest key slice'
            )
        if z_count > last_key + 1:
            raise MorphError(
                f"config.z_count: {z_count} extends beyond the last key slice "
                f"(z={last_key}); slices after it would be extrapolation — "
                'use edge_policy "clamp" to repeat the nearest key slice'
            )

    height, width = key.labels[0].shape
    for name, extent in (("z_count", z_count), ("rows", height), ("columns", width)):
        if extent > config.max_axis_cells:
            raise MorphError(
                f"size guard: {name} {extent} exceeds max_axis_cells "
                f"{config.max_axis_cells}; downsample the input or reduce "
                "the output depth"
            )
    if z_count * height * width > config.max_total_cells:
        raise MorphError(
            f"size guard: {z_count * height * width} output cells exceed "
            f"max_total_cells {config.max_total_cells}; downsample the input "
            "or reduce the output depth"
        )

    volume = np.full(
        (z_count, height, width), np.uint16(config.background), dtype=np.uint16
    )
    _fill_slices(volume, key, config)

    provenance = build_provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        config=config,
        sources=key.digests,
        notes="per-label SDF morph of labeled key slices; rows=+Y, columns=+X",
    )
    return build_material_label_volume(
        material_id=volume,
        voxel_size_xyz_m=config.voxel_size_xyz_m,
        palette=key.palette,
        provenance=provenance,
        local_to_world=config.local_to_world,
    )


def _fill_slices(
    volume: npt.NDArray[np.uint16], key: KeySlices, config: MorphStackConfig
) -> None:
    """Write key slices verbatim, clamped edges, and interpolated gaps."""
    for z_index, layer in zip(key.z_indices, key.labels, strict=True):
        volume[z_index] = layer

    # Edges (only reachable with edge_policy "clamp"): repeat nearest key.
    volume[: key.z_indices[0]] = key.labels[0]
    volume[key.z_indices[-1] + 1 :] = key.labels[-1]

    # In-plane spacing ordered like the array axes (y, x).
    spacing = (config.voxel_size_xyz_m[1], config.voxel_size_xyz_m[0])
    material_ids = sorted(key.palette.material_ids)
    for pair_index in range(len(key.z_indices) - 1):
        z_low, z_high = key.z_indices[pair_index], key.z_indices[pair_index + 1]
        if z_high - z_low < 2:
            continue
        _interpolate_gap(
            volume,
            key.labels[pair_index],
            key.labels[pair_index + 1],
            z_low,
            z_high,
            material_ids,
            spacing,
        )
