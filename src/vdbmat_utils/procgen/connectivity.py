"""Connected-component labelling (plan D7).

In-repo union-find over the 6-neighbourhood: NumPy extracts the adjacency
edges, a Python union-find with path halving merges them, and components are
renumbered deterministically by first-encountered voxel in C order. Bounded
by the phase's size guards; phase 5 owns anything faster.
"""

import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class ComponentResult:
    """6-connected components of a boolean mask.

    ``component_ids`` holds 0 for background and 1..count for foreground
    components, numbered by first-encountered voxel in C order (deterministic).
    ``sizes[i]`` is the voxel count of component ``i + 1``.
    """

    component_ids: npt.NDArray[np.uint32]
    count: int
    sizes: npt.NDArray[np.int64]


def connected_components(mask: npt.NDArray[np.bool_]) -> ComponentResult:
    """Label the 6-connected components of a 3-D boolean mask."""
    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype != np.bool_:
        from . import ProcgenError

        raise ProcgenError(
            f"mask must be a 3-D bool array, got {array.ndim}-D {array.dtype}"
        )
    flat_index = np.full(array.shape, -1, dtype=np.int64)
    foreground = np.argwhere(array)
    flat_index[array] = np.arange(len(foreground), dtype=np.int64)

    parent = np.arange(len(foreground), dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[int(parent[node])]  # path halving
            node = int(parent[node])
        return node

    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        both = array[tuple(low)] & array[tuple(high)]
        first = flat_index[tuple(low)][both]
        second = flat_index[tuple(high)][both]
        for a, b in zip(first.tolist(), second.tolist(), strict=True):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

    component_ids = np.zeros(array.shape, dtype=np.uint32)
    sizes: list[int] = []
    if len(foreground):
        roots = np.array([find(node) for node in range(len(foreground))])
        # Renumber by first occurrence in C order: foreground is already in
        # C order, so np.unique's first-index ordering gives the rule.
        unique_roots, first_seen, inverse = np.unique(
            roots, return_index=True, return_inverse=True
        )
        order = np.argsort(first_seen, kind="stable")
        rank = np.empty(len(unique_roots), dtype=np.int64)
        rank[order] = np.arange(1, len(unique_roots) + 1, dtype=np.int64)
        numbered = rank[inverse]
        component_ids[array] = numbered
        counts = np.bincount(numbered, minlength=len(unique_roots) + 1)[1:]
        sizes = counts.tolist()
    return ComponentResult(
        component_ids=component_ids,
        count=len(sizes),
        sizes=np.asarray(sizes, dtype=np.int64),
    )
