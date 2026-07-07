"""Plan D3 guard: label arrays are never numerically interpolated.

Cheap AST heuristics over the label-handling packages (``ops`` and, once it
exists, ``morph``): forbid float casts, interpolating routines, and averaging
on label-ish names. ``fields`` is exempt — that is where continuous math is
supposed to live.
"""

import ast
from pathlib import Path

import vdbmat_utils

_PACKAGE_ROOT = Path(vdbmat_utils.__file__).parent
_GUARDED_PACKAGES = ("ops", "morph")
_FORBIDDEN_CALLS = {
    "interp",
    "interpn",
    "map_coordinates",
    "zoom",
    "gaussian_filter",
    "affine_transform",
}
_LABELISH = ("material_id", "label")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in _FORBIDDEN_CALLS:
            found.append(f"{path.name}:{node.lineno} calls {name}()")
        elif name == "astype" and any(
            "float" in ast.unparse(argument) for argument in node.args
        ):
            found.append(f"{path.name}:{node.lineno} casts to float")
        elif (
            name == "mean"
            and isinstance(node.func, ast.Attribute)
            and any(
                labelish in ast.unparse(node.func.value) for labelish in _LABELISH
            )
        ):
            found.append(f"{path.name}:{node.lineno} averages a label array")
    return found


def test_label_packages_never_interpolate_labels() -> None:
    violations: list[str] = []
    for package in _GUARDED_PACKAGES:
        directory = _PACKAGE_ROOT / package
        if not directory.exists():
            continue
        for source in sorted(directory.rglob("*.py")):
            violations.extend(_violations(source))
    assert not violations, "\n".join(violations)
