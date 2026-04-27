from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import agent

goal = "Search for the top 3 open source LLMs in 2025 and save a summary to a file called llms.md"

print("Starting agent...\n")
result = agent.invoke({"messages": [HumanMessage(content=goal)]})
print("\n--- Final Answer ---")
print(result["messages"][-1].content)