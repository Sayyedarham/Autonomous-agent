"""Filesystem confinement used by agent-facing tools."""
from __future__ import annotations
import os
from pathlib import Path

WORKSPACE_ROOT = (Path(__file__).resolve().parents[1] / "workspace").resolve()
MAX_FILE_BYTES = int(os.getenv("MAX_WORKSPACE_FILE_BYTES", "1048576"))

def safe_workspace_path(filename: str) -> Path:
    if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
        raise ValueError("filename must be a non-empty relative path within the workspace.")
    candidate = (WORKSPACE_ROOT / filename).resolve()
    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise ValueError("path escapes the workspace.")
    return candidate
