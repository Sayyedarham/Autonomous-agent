import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from agent.tools import tools

load_dotenv()

# --- State definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- LLM setup ---
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- Node 1: Reasoning ---
def reason(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# --- Node 2: Tool execution ---
def run_tools(state: AgentState):
    tool_map = {t.name: t for t in tools}
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        print(f"\n>> Calling tool: {tool_name} with args: {tool_args}")
        result = tool_map[tool_name].invoke(tool_args)
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}

# --- Conditional edge: keep looping or stop ---
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# --- Build the graph ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("reason", reason)
graph_builder.add_node("tools", run_tools)

graph_builder.set_entry_point("reason")
graph_builder.add_conditional_edges("reason", should_continue)
graph_builder.add_edge("tools", "reason")

agent = graph_builder.compile()