"""Bounded agent tools; file tools are confined to the project workspace."""
from __future__ import annotations
from pathlib import Path
from langchain.tools import tool
from langchain_tavily import TavilySearch

from agent.files import MAX_FILE_BYTES, safe_workspace_path

web_search = TavilySearch(max_results=3)

@tool
def write_file(filename: str, content: str) -> str:
    """Write UTF-8 text to a relative workspace filename, up to one configured MB."""
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return "Error: content exceeds workspace file-size limit."
    try:
        path = safe_workspace_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to workspace/{filename}"
    except (ValueError, OSError, UnicodeError) as exc:
        return f"Error: {exc}"

@tool
def read_file(filename: str) -> str:
    """Read a UTF-8 file from the relative workspace path, subject to size limits."""
    try:
        path = safe_workspace_path(filename)
        if not path.is_file():
            return f"Error: workspace/{filename} does not exist"
        if path.stat().st_size > MAX_FILE_BYTES:
            return "Error: file exceeds workspace file-size limit."
        return path.read_text(encoding="utf-8")
    except (ValueError, OSError, UnicodeError) as exc:
        return f"Error: {exc}"

tools = [web_search, write_file, read_file]
