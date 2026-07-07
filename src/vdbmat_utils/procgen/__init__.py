"""Procedural natural-material generators (phase 3).

Seeded, deterministic primitives (lattice hashing, gradient noise) and the
formation models built on them. All continuous math lives in
``fields.ScalarField`` space; material labels appear only through the
sanctioned quantization or explicit integer assembly (plan D6).
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class ProcgenError(VdbmatUtilsError):
    """A procedural-generator input violates its contract."""


from .domain import FormationDomain  # noqa: E402
from .hashing import hash_lattice, hash_to_unit  # noqa: E402
from .noise import fbm, gradient_noise, ridged_fbm  # noqa: E402

__all__ = [
    "FormationDomain",
    "ProcgenError",
    "fbm",
    "gradient_noise",
    "hash_lattice",
    "hash_to_unit",
    "ridged_fbm",
]
