"""Browser-free tests for the designlab config catalog."""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "designlab"
sys.path.insert(0, str(EXAMPLES_DIR))

from designlab_configs import (  # noqa: E402
    DesignlabConfigError,
    load_config,
    resolve_config_path,
    resolve_config_root,
    save_config,
    scan_config_catalog,
)
from designlab_registry import PRIMITIVE_ARRAY_METHOD  # noqa: E402

from vdbmat_utils.core import config_digest  # noqa: E402
from vdbmat_utils.primitives import PrimitiveArrayConfig  # noqa: E402

_CONFIG = PrimitiveArrayConfig(
    voxel_size_xyz_m=(0.0001, 0.0001, 0.0001),
    primitive="cube",
    counts_xyz=(2, 2, 2),
    primitive_size_m=0.0004,
    gap_m=0.0002,
    margin_m=0.0001,
)


def test_resolve_config_root_requires_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(DesignlabConfigError):
        resolve_config_root(missing)
    assert resolve_config_root(tmp_path) == tmp_path.resolve()


def test_scan_config_catalog_matches_registered_suffix_only(tmp_path: Path) -> None:
    (tmp_path / "demo.primarray.json").write_text(_CONFIG.to_json(), encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.primarray.json").write_text(_CONFIG.to_json(), encoding="utf-8")

    found = scan_config_catalog(tmp_path)
    assert [c.root_relative for c in found] == [
        "demo.primarray.json",
        "sub/nested.primarray.json",
    ]
    assert all(c.method is PRIMITIVE_ARRAY_METHOD for c in found)


def test_scan_config_catalog_excludes_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escaped.primarray.json").write_text(_CONFIG.to_json(), encoding="utf-8")
    link = root / "escaped.primarray.json"
    try:
        link.symlink_to(outside / "escaped.primarray.json")
    except OSError:
        pytest.skip("symlinks not supported in this environment")

    assert scan_config_catalog(root) == []


def test_resolve_config_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside_file = tmp_path / "outside.primarray.json"
    outside_file.write_text(_CONFIG.to_json(), encoding="utf-8")

    with pytest.raises(DesignlabConfigError):
        resolve_config_path(root, Path("../outside.primarray.json"))
    with pytest.raises(DesignlabConfigError):
        resolve_config_path(root, Path("missing.primarray.json"))


def test_save_then_load_round_trips_config_digest(tmp_path: Path) -> None:
    target = save_config(_CONFIG, tmp_path, PRIMITIVE_ARRAY_METHOD, "demo")
    assert target == tmp_path / "demo.primarray.json"

    loaded = load_config(target, tmp_path, PRIMITIVE_ARRAY_METHOD)
    assert config_digest(loaded) == config_digest(_CONFIG)

    resaved = save_config(loaded, tmp_path, PRIMITIVE_ARRAY_METHOD, "demo-2")
    reloaded = load_config(resaved, tmp_path, PRIMITIVE_ARRAY_METHOD)
    assert config_digest(reloaded) == config_digest(_CONFIG)


def test_save_config_refuses_overwrite(tmp_path: Path) -> None:
    save_config(_CONFIG, tmp_path, PRIMITIVE_ARRAY_METHOD, "demo")
    with pytest.raises(DesignlabConfigError):
        save_config(_CONFIG, tmp_path, PRIMITIVE_ARRAY_METHOD, "demo")


def test_save_config_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(DesignlabConfigError):
        save_config(_CONFIG, tmp_path, PRIMITIVE_ARRAY_METHOD, "../escape")
    with pytest.raises(DesignlabConfigError):
        save_config(_CONFIG, tmp_path, PRIMITIVE_ARRAY_METHOD, "Has Spaces")


def test_load_config_propagates_unknown_field_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.primarray.json"
    bad.write_text('{"not_a_field": 1}', encoding="utf-8")
    with pytest.raises(DesignlabConfigError, match="unknown configuration fields"):
        load_config(bad, tmp_path, PRIMITIVE_ARRAY_METHOD)
