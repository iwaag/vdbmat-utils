"""Plans D1 (phase 1 and phase 2): subpackage import boundaries.

Cheap AST walk over the package sources. Each subpackage may import only its
allowlisted sibling subpackages (plus itself, external libraries, and
``vdbmat``); imports must flow toward ``core``.
"""

import ast
from pathlib import Path

import vdbmat_utils

_PACKAGE_ROOT = Path(vdbmat_utils.__file__).parent
# subpackage -> sibling subpackages it may import (itself is always allowed)
_ALLOWED_SIBLINGS: dict[str, set[str]] = {
    "core": set(),
    "io": {"core"},
    "preview": {"core"},
    "mesh": {"core", "io", "preview"},
    "image": {"core", "io", "preview"},
    "ops": {"core"},
    "fields": {"core"},
    "morph": {"core", "io", "ops", "fields", "image"},
    "pipeline": {"core", "io", "ops", "fields", "morph", "preview"},
}


def _imported_modules(path: Path, package: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                modules.add(f"vdbmat_utils.{package}.{node.module or ''}")
            elif node.module is not None:
                modules.add(node.module)
    return modules


def test_subpackage_imports_flow_toward_core() -> None:
    violations: list[str] = []
    for package, allowed in _ALLOWED_SIBLINGS.items():
        directory = _PACKAGE_ROOT / package
        if not directory.exists():
            continue
        permitted = allowed | {package}
        for source in sorted(directory.rglob("*.py")):
            for module in _imported_modules(source, package):
                if not module.startswith("vdbmat_utils."):
                    continue
                sibling = module.split(".")[1]
                if sibling not in permitted:
                    violations.append(
                        f"{source.relative_to(_PACKAGE_ROOT)} imports {module}"
                    )
    assert not violations, "\n".join(violations)
