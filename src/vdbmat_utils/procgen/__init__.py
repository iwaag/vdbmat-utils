"""Procedural natural-material generators (phase 3).

Seeded, deterministic primitives (lattice hashing, gradient noise) and the
formation models built on them. All continuous math lives in
``fields.ScalarField`` space; material labels appear only through the
sanctioned quantization or explicit integer assembly (plan D6).
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class ProcgenError(VdbmatUtilsError):
    """A procedural-generator input violates its contract."""


from .cells import WorleyCells, worley, worley_at  # noqa: E402
from .connectivity import ComponentResult, connected_components  # noqa: E402
from .domain import FormationDomain  # noqa: E402
from .hashing import hash_derive, hash_lattice, hash_to_unit  # noqa: E402
from .morphology import close_mask, dilate, erode, open_mask  # noqa: E402
from .noise import (  # noqa: E402
    fbm,
    fbm_at,
    gradient_noise,
    gradient_noise_at,
    ridged_fbm,
    ridged_fbm_at,
)
from .stats import compute_stats, evaluate_constraints  # noqa: E402
from .warp import warped_coordinates  # noqa: E402

__all__ = [
    "ComponentResult",
    "FormationDomain",
    "ProcgenError",
    "WorleyCells",
    "close_mask",
    "compute_stats",
    "connected_components",
    "dilate",
    "erode",
    "evaluate_constraints",
    "fbm",
    "fbm_at",
    "gradient_noise",
    "gradient_noise_at",
    "hash_derive",
    "hash_lattice",
    "hash_to_unit",
    "open_mask",
    "ridged_fbm",
    "ridged_fbm_at",
    "warped_coordinates",
    "worley",
    "worley_at",
]
