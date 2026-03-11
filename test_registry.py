import sys
sys.path.append('src')
from agent.tools.registry import create_registry
registry = create_registry()
print("Tools loaded:", len(registry.list_tools()))
if "web_search" in registry.get_tool_names():
    print("web_search is AVAILABLE")
else:
    print("web_search is MISSING")
