"""Contract tests for Phase 3 procedural formations."""

import hashlib
import json
from pathlib import Path

from vdbmat.io import read_material_label_manifest

from vdbmat_utils.cli.main import main
from vdbmat_utils.preview import slice_ascii

ROOT = Path(__file__).resolve().parents[2]
MARBLE_CONFIG = ROOT / "examples/formation_generation/marble-like.formation.json"
GRANITE_CONFIG = ROOT / "examples/formation_generation/granite-like.formation.json"
SWEEP_CONFIG = ROOT / "examples/formation_generation/tiny-sweep.sweep.json"

PAYLOAD_GOLDENS = {
    "marble-like": "229bc9f7dae1da98f7164e7863abd4886e3df4f6a979eb92f3cf4cf5c4071032",
    "granite-like": "547dce46d60e6fd755a7a9cb2ea25b6f816794e3a442623a7818f1846a30aa6c",
}
STATS_GOLDENS = {
    "marble-like": "2d7c1812c8a4be67babebee85267dcfa52ffa32ceae279be4575aeb2b20e70d4",
    "granite-like": "845c66a6dd35f287bc95cf256c34254bece5d49a80c196e0fac00e8e34c28af5",
}
SWEEP_SUMMARY_SHA256 = (
    "a34f89cd7b4180e8aff0b0afba9981e66a73614c2e1f3ebcb0683d77897bacb2"
)

MARBLE_ASCII_Z = """\
slice z=9  +x →  +y ↓
11112211111111112211111111112
11112211111111112211111111112
11112211111111122211111111112
11112211111111112211111111112
11112211111111112211111111112
11112211111111112211111111112
11112211111111112221111111112
11112211111111113333311111112
33333333333333333333333331112
33312213333311112211113333333
11122211111111112211111111332
11122211111111112211111111122
11122211111111112211111111122
11112211111111112211111111122
11122211111111122211111111122
11122211111111112211111111112
11122211111111112221111111112
11122111111111122221111111122
11122211111111122211111111122
11122211111111122211111111122
11122211111111122211111111122
11122211111111122211111111122
11122211111111122211111111112"""

MARBLE_ASCII_Y = """\
slice y=11  +x →  +z ↓
11122221111111112211111111112
11122211111111122211111111112
11122211111111122111111111112
11122221111111122211111111112
11112221111111122211111111112
11112221111111122221111111122
11112211111111112211111111122
11112211111111112211111111122
11112211111111112211111111122
11122211111111112211111111122
11122211111111112211111111122
11112211111111112211111111122
11112211111111112211111111122
11112221111111112211111111112
11112221111111112211111111112
11111221111111112211111111112
11111221111111112211111111111
11112221111111112211111111112
11112221111111112211111111112"""

MARBLE_ASCII_X = """\
slice x=14  +y →  +z ↓
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111311111111111111
11111111311111111111111
11111111311111111111111
11111111311111111111111
11111111311111111111111
11111111311111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111331111111111111
11111111131111111111111"""


def _generate(tmp_path: Path, config: Path, name: str, out_name: str = "out") -> Path:
    out = tmp_path / out_name
    assert main(
        [
            "generate-formation",
            "--config",
            str(config),
            "--out",
            str(out),
            "--name",
            name,
        ]
    ) == 0
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_formation_double_run_is_byte_equal(tmp_path: Path) -> None:
    first = _generate(tmp_path, MARBLE_CONFIG, "marble-like", "a")
    second = _generate(tmp_path, MARBLE_CONFIG, "marble-like", "b")
    for filename in (
        "marble-like.voxels.json",
        "marble-like.material_id.npy",
        "marble-like.stats.json",
        "marble-like.optical-mapping.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_reference_payload_and_stats_goldens(tmp_path: Path) -> None:
    for name, config in (
        ("marble-like", MARBLE_CONFIG),
        ("granite-like", GRANITE_CONFIG),
    ):
        out = _generate(tmp_path, config, name, name)
        assert _sha256(out / f"{name}.material_id.npy") == PAYLOAD_GOLDENS[name]
        assert _sha256(out / f"{name}.stats.json") == STATS_GOLDENS[name]


def test_seed_change_changes_payload_digest(tmp_path: Path) -> None:
    changed_config = tmp_path / "marble-seed.json"
    payload = json.loads(MARBLE_CONFIG.read_text(encoding="utf-8"))
    payload["seed"] = payload["seed"] + 1
    changed_config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    original = _generate(tmp_path, MARBLE_CONFIG, "marble-like", "original")
    changed = _generate(tmp_path, changed_config, "marble-like", "changed")
    assert _sha256(original / "marble-like.material_id.npy") != _sha256(
        changed / "marble-like.material_id.npy"
    )


def test_orientation_ascii_goldens(tmp_path: Path) -> None:
    out = _generate(tmp_path, MARBLE_CONFIG, "marble-like")
    volume = read_material_label_manifest(out / "marble-like.voxels.json")
    assert volume.geometry.shape_zyx == (19, 23, 29)
    assert slice_ascii(volume, "z", 9) == MARBLE_ASCII_Z
    assert slice_ascii(volume, "y", 11) == MARBLE_ASCII_Y
    assert slice_ascii(volume, "x", 14) == MARBLE_ASCII_X


def test_sweep_formation_double_run_and_summary_golden(tmp_path: Path) -> None:
    for out_name in ("a", "b"):
        assert main(
            [
                "sweep-formation",
                "--config",
                str(SWEEP_CONFIG),
                "--out",
                str(tmp_path / out_name),
                "--name",
                "tiny",
            ]
        ) == 0
    assert (tmp_path / "a/sweep_summary.json").read_bytes() == (
        tmp_path / "b/sweep_summary.json"
    ).read_bytes()
    assert _sha256(tmp_path / "a/sweep_summary.json") == SWEEP_SUMMARY_SHA256
