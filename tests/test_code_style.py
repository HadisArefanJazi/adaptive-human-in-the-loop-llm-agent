"""validate lowercase project naming and the allowed imports."""

import ast
from pathlib import Path as path_type


def test_project_bindings_and_imports_follow_code_style():
    root = path_type(__file__).resolve().parents[1]
    for folder in ("adaptive_hitl_agent", "scripts", "tests"):
        for path in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.append(node.id)
                elif isinstance(node, ast.arg):
                    names.append(node.arg)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.extend(alias.asname or alias.name for alias in node.names)
                    if isinstance(node, ast.Import):
                        assert all(alias.name not in {"re", "operator", "dataclasses"} for alias in node.names)
                    else:
                        assert node.module not in {"re", "operator", "dataclasses"}
                        assert not (
                            node.module == "collections"
                            and any(alias.name == "Counter" for alias in node.names)
                        )
                assert all(name == name.lower() for name in names), (path, names)
