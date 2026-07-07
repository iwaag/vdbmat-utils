"""The only sanctioned scalar→label conversion (plan D3)."""

import numpy as np
import numpy.typing as npt

from . import FieldError, ScalarField


def quantize_to_labels(
    field: ScalarField,
    *,
    bin_edges: tuple[float, ...],
    material_ids: tuple[int, ...],
) -> npt.NDArray[np.uint16]:
    """Threshold a scalar field into discrete material ids.

    ``bin_edges`` must be strictly increasing; ``material_ids`` has one entry
    per bin, i.e. ``len(bin_edges) + 1``. A value ``v`` gets
    ``material_ids[i]`` where ``bin_edges[i-1] <= v < bin_edges[i]`` — a value
    exactly on an edge belongs to the higher bin (deterministic tie rule).
    NaN values are an error, never a guess.
    """
    if len(material_ids) != len(bin_edges) + 1:
        raise FieldError(
            f"need len(bin_edges) + 1 material ids: got {len(material_ids)} ids "
            f"for {len(bin_edges)} edges"
        )
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.size and not (np.diff(edges) > 0).all():
        raise FieldError("bin_edges must be strictly increasing")
    limit = int(np.iinfo(np.uint16).max)
    for material_id in material_ids:
        if not 0 <= int(material_id) <= limit:
            raise FieldError(
                f"material_id {material_id} is outside the uint16 range"
            )
    if np.isnan(field.values).any():
        raise FieldError("field contains NaN values; quantization refuses to guess")
    ids = np.asarray(material_ids, dtype=np.uint16)
    bins = np.searchsorted(edges, field.values, side="right")
    result: npt.NDArray[np.uint16] = ids[bins]
    return result
