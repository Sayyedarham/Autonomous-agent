"""Optional live-provider smoke test; it is deliberately not part of offline pytest."""

def main() -> None:
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from agent.graph import agent

    load_dotenv()
    goal = "Search for the top 3 open source LLMs in 2025 and save a summary to a file called llms.md"
    print("Starting agent...\n")
    result = agent.invoke({"messages": [HumanMessage(content=goal)], "iterations": 0})
    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
