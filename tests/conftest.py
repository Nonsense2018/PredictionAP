from __future__ import annotations

import importlib.util
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module(relative_path: str):
    """Load a project Python file as a module for direct function testing."""
    root = project_root()
    file_path = root / relative_path
    module_name = relative_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
