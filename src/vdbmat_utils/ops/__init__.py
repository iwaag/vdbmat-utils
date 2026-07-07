"""Label-safe volume operations (plan D5).

Every operation is a pure function ``MaterialLabelVolume (+ params) ->
MaterialLabelVolume`` on dense arrays, validated on output through
``build_material_label_volume``. Operations move ``uint16`` labels exactly —
no function in this package may cast a label array to float or interpolate
it; smoothness lives in ``vdbmat_utils.fields``. Provenance passes through
from the input volume (the ``base`` volume for binary operations); pipeline
runs assemble their own provenance in Phase 2 Step 3.
"""

from vdbmat_utils.core.errors import VdbmatUtilsError


class OpsError(VdbmatUtilsError):
    """A volume operation's inputs violate its contract."""


from .boolean import compose  # noqa: E402
from .crop_pad import crop, pad  # noqa: E402
from .mask import apply_mask  # noqa: E402
from .palette import merge_palettes  # noqa: E402
from .remap import remap_materials  # noqa: E402
from .resample import resample  # noqa: E402
from .transform import orient, place  # noqa: E402

__all__ = [
    "OpsError",
    "apply_mask",
    "compose",
    "crop",
    "merge_palettes",
    "orient",
    "pad",
    "place",
    "remap_materials",
    "resample",
]
