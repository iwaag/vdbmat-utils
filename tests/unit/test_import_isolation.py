"""Plan D1: ``mesh`` and ``image`` must not import each other.

Cheap AST walk over the package sources; both may import ``core``, ``io``,
and ``preview``.
"""

import ast
from pathlib import Path

import vdbmat_utils

_PACKAGE_ROOT = Path(vdbmat_utils.__file__).parent
_FORBIDDEN = {
    "mesh": "vdbmat_utils.image",
    "image": "vdbmat_utils.mesh",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            prefix = "vdbmat_utils." if node.level > 0 else ""
            modules.add(f"{prefix}{node.module}")
    return modules


def test_mesh_and_image_do_not_import_each_other() -> None:
    for package, forbidden in _FORBIDDEN.items():
        for source in (_PACKAGE_ROOT / package).rglob("*.py"):
            for module in _imported_modules(source):
                assert not module.startswith(forbidden), (
                    f"{source.relative_to(_PACKAGE_ROOT)} imports {module}"
                )
