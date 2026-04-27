import os
import httpx
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.tools import tool

load_dotenv()

# Tool 1: Web search
web_search = TavilySearch(max_results=3)

# Tool 2: Write a file into the workspace folder
@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the workspace folder."""
    os.makedirs("workspace", exist_ok=True)
    path = os.path.join("workspace", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to workspace/{filename}"

# Tool 3: Read a file from the workspace folder
@tool
def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = os.path.join("workspace", filename)
    if not os.path.exists(path):
        return f"Error: workspace/{filename} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# This is the list we'll import into graph.py
tools = [web_search, write_file, read_file]