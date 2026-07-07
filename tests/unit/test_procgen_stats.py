import numpy as np
from vdbmat.core import MaterialDefinition, MaterialLabelVolume, MaterialRole

from vdbmat_utils.core import build_material_label_volume
from vdbmat_utils.core.provenance import build_provenance
from vdbmat_utils.procgen.stats import compute_stats, evaluate_constraints


def _volume(
    labels: np.ndarray, *, voxel_size: tuple[float, float, float]
) -> MaterialLabelVolume:
    return build_material_label_volume(
        material_id=labels.astype(np.uint16),
        voxel_size_xyz_m=voxel_size,
        palette=(
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(1, "stone", MaterialRole.MATERIAL),
        ),
        provenance=build_provenance(
            generator="test", generator_version="0.0.0", config=None
        ),
    )


def test_volume_fraction_components_and_constraints() -> None:
    labels = np.zeros((3, 4, 5), dtype=np.uint16)
    labels[0, 0, 0] = 1
    labels[2, 3, 4] = 1
    stats = compute_stats(_volume(labels, voxel_size=(1.0, 1.0, 1.0)))
    stone = stats.by_material_id(1)
    assert stone.count == 2
    assert stone.volume_fraction == 2 / 60
    assert stone.component_count == 2
    assert stone.largest_component_fraction == 0.5

    results = evaluate_constraints(
        stats,
        (
            {"kind": "connected", "material_id": 1, "max_components": 2},
            {
                "kind": "min-largest-component-fraction",
                "material_id": 1,
                "min": 0.6,
            },
        ),
    )
    assert [item.passed for item in results] == [True, False]


def test_local_thickness_uses_anisotropic_spacing() -> None:
    labels = np.zeros((5, 3, 3), dtype=np.uint16)
    labels[1:4, :, :] = 1
    stats = compute_stats(_volume(labels, voxel_size=(1.0, 1.0, 0.25)))
    stone = stats.by_material_id(1)
    # D7 proxy: 2 * EDT to the nearest non-mask voxel centre, in z/y/x spacing.
    assert stone.local_thickness_m["min"] == 0.5
    assert stone.local_thickness_m["max"] == 1.0


def test_empty_material_stats_are_reported() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.uint16)
    stats = compute_stats(_volume(labels, voxel_size=(1.0, 1.0, 1.0)))
    stone = stats.by_material_id(1)
    assert stone.count == 0
    assert stone.local_thickness_m["min"] is None
    assert stone.component_count == 0
