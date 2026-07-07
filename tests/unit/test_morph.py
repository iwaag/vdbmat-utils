"""Unit tests for the morph-stack workflow (plan D4, Step 2.3)."""

import itertools
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from vdbmat_utils.image import ImageStackConfig, ImageStackError, convert_image_stack
from vdbmat_utils.morph import MorphError, MorphStackConfig, morph_stack

LEVELS: tuple[Mapping[str, object], ...] = (
    {"gray": 0, "material_id": 0, "name": "air", "role": "background"},
    {"gray": 128, "material_id": 1, "name": "resin_a", "role": "material"},
    {"gray": 255, "material_id": 2, "name": "resin_b", "role": "material"},
)


def _config(**overrides: object) -> MorphStackConfig:
    fields: dict[str, object] = {
        "voxel_size_xyz_m": (0.001, 0.001, 0.001),
        "levels": LEVELS,
    }
    fields.update(overrides)
    return MorphStackConfig(**fields)  # type: ignore[arg-type]


def _write_pgm(path: Path, pixels: npt.NDArray[np.uint8]) -> Path:
    rows, cols = pixels.shape
    path.write_bytes(f"P5\n{cols} {rows}\n255\n".encode() + pixels.tobytes())
    return path


def _square(
    shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int, gray: int = 128
) -> npt.NDArray[np.uint8]:
    pixels = np.zeros(shape, dtype=np.uint8)
    pixels[y0:y1, x0:x1] = gray
    return pixels


def test_key_slices_reproduced_verbatim(tmp_path: Path) -> None:
    first = _square((6, 8), 1, 3, 1, 4)
    last = _square((6, 8), 2, 5, 3, 7, gray=255)
    _write_pgm(tmp_path / "slice_0000.pgm", first)
    _write_pgm(tmp_path / "slice_0004.pgm", last)
    volume = morph_stack(tmp_path, _config())
    assert volume.geometry.shape_zyx == (5, 6, 8)
    lookup = {0: 0, 128: 1, 255: 2}
    for key_z, pixels in ((0, first), (4, last)):
        expected = np.vectorize(lookup.__getitem__)(pixels)
        np.testing.assert_array_equal(volume.material_id[key_z], expected)


def test_midpoint_of_two_offset_squares(tmp_path: Path) -> None:
    # A 4-cell-wide square shifted 1 cell right between the keys: the
    # midpoint slice is the square at the intermediate position (cells whose
    # averaged signed distance is strictly negative).
    _write_pgm(tmp_path / "s0.pgm", _square((5, 7), 1, 4, 1, 5))
    _write_pgm(tmp_path / "s2.pgm", _square((5, 7), 1, 4, 2, 6))
    volume = morph_stack(tmp_path, _config())
    expected = np.zeros((5, 7), dtype=np.uint16)
    expected[1:4, 2:5] = 1
    np.testing.assert_array_equal(volume.material_id[1], expected)


def test_merge_event_two_squares_becoming_one(tmp_path: Path) -> None:
    # Two separated squares → one wide bar: interior slices grow toward each
    # other and eventually connect (an emergent topology change).
    two = np.zeros((5, 10), dtype=np.uint8)
    two[1:4, 1:3] = 128
    two[1:4, 7:9] = 128
    bar = _square((5, 10), 1, 4, 1, 9)
    _write_pgm(tmp_path / "k0.pgm", two)
    _write_pgm(tmp_path / "k4.pgm", bar)
    volume = morph_stack(tmp_path, _config())
    foreground_columns = [
        set(np.nonzero(volume.material_id[z][2])[0].tolist()) for z in range(5)
    ]
    # Monotone growth toward the bar, disconnected first, connected last.
    for z in range(4):
        assert foreground_columns[z] <= foreground_columns[z + 1]
    assert 4 not in foreground_columns[0] and 5 not in foreground_columns[0]
    assert foreground_columns[4] == set(range(1, 9))
    connected = [
        np.all(np.diff(sorted(columns)) == 1) for columns in foreground_columns
    ]
    assert connected[4] and not connected[0]


def test_disappearing_label_shrinks_monotonically(tmp_path: Path) -> None:
    # resin_b (255) present only on the first key slice: its foreground count
    # must be non-increasing toward the side where it is absent.
    first = _square((7, 7), 1, 6, 1, 6, gray=255)
    _write_pgm(tmp_path / "a0.pgm", first)
    _write_pgm(tmp_path / "a5.pgm", np.zeros((7, 7), dtype=np.uint8))
    volume = morph_stack(tmp_path, _config())
    counts = [int((volume.material_id[z] == 2).sum()) for z in range(6)]
    assert counts[0] == 25
    assert counts[5] == 0
    assert all(a >= b for a, b in itertools.pairwise(counts))
    # Plan D4 fixes IEEE inf arithmetic: an absent label's distance is +inf
    # on every strictly interior slice, so the label is gone from z=1 on.
    assert counts[1] == 0


def test_symmetric_swap_midpoint_is_background(tmp_path: Path) -> None:
    # Two labels swap halves between the key slices: at the midpoint every
    # pixel's interpolated distances cancel to exactly 0 for both labels, and
    # 0 is not inside (< 0), so the whole slice is background (plan D4).
    left_a = np.zeros((3, 8), dtype=np.uint8)
    left_a[:, 0:4] = 128
    left_a[:, 4:8] = 255
    right_a = np.zeros((3, 8), dtype=np.uint8)
    right_a[:, 0:4] = 255
    right_a[:, 4:8] = 128
    _write_pgm(tmp_path / "t0.pgm", left_a)
    _write_pgm(tmp_path / "t2.pgm", right_a)
    volume = morph_stack(tmp_path, _config())
    assert (volume.material_id[1] == 0).all()


def test_tie_goes_to_lowest_material_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exact negative ties are geometrically contrived with real EDTs, so pin
    # the rule directly: force every label's distance field to the same
    # negative constant and check that the lowest id claims every pixel.
    _write_pgm(tmp_path / "t0.pgm", _square((3, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "t2.pgm", _square((3, 4), 1, 3, 1, 3, gray=255))
    from vdbmat_utils.morph import interpolate

    # Calls arrive per label in ascending id, low key then high key:
    # background (0) stays outside, labels 1 and 2 tie at -1 everywhere.
    values = iter([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    monkeypatch.setattr(
        interpolate,
        "signed_distance",
        lambda mask, spacing: np.full(mask.shape, next(values), dtype=np.float64),
    )
    volume = morph_stack(tmp_path, _config())
    assert (volume.material_id[1] == 1).all()  # tie between 1 and 2 → lowest


def test_edge_policy_error_rejects_missing_leading_and_trailing(
    tmp_path: Path,
) -> None:
    _write_pgm(tmp_path / "s1.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "s3.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="first key slice is z=1"):
        morph_stack(tmp_path, _config())

    for path in tmp_path.glob("*.pgm"):
        path.unlink()
    _write_pgm(tmp_path / "s0.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "s2.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="extends beyond the last key slice"):
        morph_stack(tmp_path, _config(z_count=5))


def test_edge_policy_clamp_repeats_nearest_key(tmp_path: Path) -> None:
    first = _square((4, 4), 0, 2, 0, 2)
    last = _square((4, 4), 1, 3, 1, 3, gray=255)
    _write_pgm(tmp_path / "s1.pgm", first)
    _write_pgm(tmp_path / "s2.pgm", last)
    volume = morph_stack(tmp_path, _config(edge_policy="clamp", z_count=5))
    np.testing.assert_array_equal(volume.material_id[0], volume.material_id[1])
    np.testing.assert_array_equal(volume.material_id[3], volume.material_id[2])
    np.testing.assert_array_equal(volume.material_id[4], volume.material_id[2])
    assert (volume.material_id[1] == 1).sum() == 4
    assert (volume.material_id[2] == 2).sum() == 4


def test_z_count_smaller_than_last_key_is_error(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "s0.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "s4.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="z_count"):
        morph_stack(tmp_path, _config(z_count=3))


def test_undeclared_gray_names_file_and_value(tmp_path: Path) -> None:
    pixels = _square((4, 4), 0, 2, 0, 2)
    pixels[3, 3] = 77
    _write_pgm(tmp_path / "s0.pgm", pixels)
    _write_pgm(tmp_path / "s2.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match=r"s0\.pgm: gray value 77"):
        morph_stack(tmp_path, _config())


def test_gap_is_fine_where_convert_image_stack_errors(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "slice_0000.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "slice_0002.pgm", _square((4, 4), 1, 3, 1, 3))
    stack_config = ImageStackConfig(
        voxel_size_xyz_m=(0.001, 0.001, 0.001), levels=LEVELS
    )
    with pytest.raises(ImageStackError, match="missing index"):
        convert_image_stack(tmp_path, stack_config)
    assert morph_stack(tmp_path, _config()).geometry.shape_zyx == (3, 4, 4)


def test_duplicate_z_index_is_error(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "a_0001.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "b_0001.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="same z index 1"):
        morph_stack(tmp_path, _config(edge_policy="clamp"))


def test_multiple_numeric_groups_is_error(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "s1_v2.pgm", _square((4, 4), 0, 2, 0, 2))
    with pytest.raises(MorphError, match="exactly one numeric group"):
        morph_stack(tmp_path, _config())


def test_non_monotonic_filename_order_is_error(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "a10.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "b2.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="not monotonic"):
        morph_stack(tmp_path, _config(edge_policy="clamp"))


def test_size_guards(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "s0.pgm", _square((4, 4), 0, 2, 0, 2))
    _write_pgm(tmp_path / "s2.pgm", _square((4, 4), 1, 3, 1, 3))
    with pytest.raises(MorphError, match="max_axis_cells"):
        morph_stack(tmp_path, _config(max_axis_cells=2))
    with pytest.raises(MorphError, match="max_total_cells"):
        morph_stack(tmp_path, _config(max_total_cells=10))


def test_background_must_be_declared(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "s0.pgm", _square((4, 4), 0, 2, 0, 2))
    with pytest.raises(MorphError, match=r"config\.background"):
        morph_stack(tmp_path, _config(background=9))


def test_provenance_and_determinism(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "s0.pgm", _square((5, 5), 0, 3, 0, 3))
    _write_pgm(tmp_path / "s3.pgm", _square((5, 5), 2, 5, 2, 5, gray=255))
    first = morph_stack(tmp_path, _config())
    second = morph_stack(tmp_path, _config())
    np.testing.assert_array_equal(first.material_id, second.material_id)
    assert first.provenance == second.provenance
    assert first.provenance.generator == "vdbmat-utils.morph.stack"
    assert len(first.provenance.sources) == 2
    assert first.provenance.configuration_digest is not None
