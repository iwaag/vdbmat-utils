"""Config-driven volume-operation pipelines (plan D6).

A pipeline is a flat SSA-style step list: each step reads named volume ids
(``"from"``, or ``"base"``/``"overlay"``/``"mask"`` for binary ops), applies
one registered operation, and binds its result to a fresh id (``"as"``). All
structural problems — unknown ops or parameters, unbound or rebound ids,
unused inputs — are reported at validation time with the step index, before
any array work.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class PipelineError(VdbmatUtilsError):
    """A pipeline configuration or its inputs violate the contract."""


from .engine import PipelineConfig, run_pipeline, validate_pipeline  # noqa: E402

__all__ = [
    "PipelineConfig",
    "PipelineError",
    "run_pipeline",
    "validate_pipeline",
]
