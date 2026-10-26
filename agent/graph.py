"""LangGraph orchestration with explicit bounds and recoverable tool failures."""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from agent.tools import tools

load_dotenv()
MAX_GRAPH_ITERATIONS = int(os.getenv("MAX_GRAPH_ITERATIONS", "8"))
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "20"))

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    iterations: int
    terminal_error: str | None

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)
llm_with_tools = llm.bind_tools(tools)

def reason(state: AgentState):
    if state.get("iterations", 0) >= MAX_GRAPH_ITERATIONS:
        return {"messages": [], "terminal_error": f"Maximum graph iterations ({MAX_GRAPH_ITERATIONS}) reached."}
    try:
        return {"messages": [llm_with_tools.invoke(state["messages"])], "iterations": state.get("iterations", 0) + 1}
    except Exception as exc:
        return {"messages": [], "terminal_error": f"Model invocation failed: {type(exc).__name__}: {exc}"}

def run_tools(state: AgentState):
    tool_map = {item.name: item for item in tools}
    last_message = state["messages"][-1]
    results = []
    for call in getattr(last_message, "tool_calls", []):
        name, args = call.get("name"), call.get("args", {})
        if name not in tool_map:
            content = f"Tool error: unknown tool '{name}'."
        else:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    content = str(executor.submit(tool_map[name].invoke, args).result(timeout=TOOL_TIMEOUT_SECONDS))
            except TimeoutError:
                content = f"Tool error: '{name}' timed out after {TOOL_TIMEOUT_SECONDS} seconds."
            except Exception as exc:
                content = f"Tool error: {type(exc).__name__}: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
    return {"messages": results}

def should_continue(state: AgentState):
    if state.get("terminal_error"):
        return END
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

builder = StateGraph(AgentState)
builder.add_node("reason", reason)
builder.add_node("tools", run_tools)
builder.set_entry_point("reason")
builder.add_conditional_edges("reason", should_continue)
builder.add_edge("tools", "reason")
agent = builder.compile()
