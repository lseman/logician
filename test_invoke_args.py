import sys
import json
import os

sys.path.append('src')

from src.agent.core import Agent

def main():
    if os.path.exists("available_tools.json"):
        os.remove("available_tools.json")

    # Initialize the agent
    agent = Agent()
    print("Agent initialized")

    # Test invoke_skill executing tool directly
    try:
        res = agent.tools.call_tool("invoke_skill", skill="analysis", args='{"period": 12, "column": "value"}')
        print("Result of invoke_skill with args:")
        print(res)
    except Exception as e:
        print(f"invoke_skill failed: {e}")

    if os.path.exists("available_tools.json"):
        print("available_tools.json EXISTS.")
        with open("available_tools.json") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} tools.")
            # Print invoke_skill definition to verify args parameter
            invoke_tool = next((t for t in data if t["name"] == "invoke_skill"), None)
            if invoke_tool:
                has_args = any(p["name"] == "args" for p in invoke_tool.get("parameters", []))
                print(f"invoke_skill has args parameter: {has_args}")
    else:
        print("available_tools.json MISSING.")

if __name__ == "__main__":
    main()
