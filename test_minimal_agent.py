#!/usr/bin/env python3
"""
Minimal agent test - exactly as user requested.
"""

from src import create_agent, ToolParameter

print("Creating agent...")
agent = create_agent(
    embedding_model=None,
    config_overrides={"rag_enabled": False},
)
print("✅ Agent created.")

def ping_tool(text: str) -> str:
    return f"PONG: {text}"

agent.add_tool(
    name="ping",
    description="Simple echo ping.",
    function=ping_tool,
    parameters=[ToolParameter("text", "string", "Text to echo")],
)
print("✅ Tool added.")

print("\n" + "=" * 60)
print("Testing simple chat (no tools)...")
print("=" * 60)

response = agent.chat("Say hello (no tools).", fresh_session=True, verbose=True)

print("\n" + "=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response)
print("\n" + "=" * 60)

# Check if response is sane
if len(response) < 200 and ("hello" in response.lower() or "hi" in response.lower()):
    print("✅ Response looks GOOD")
else:
    print("❌ Response looks BAD (too long or irrelevant)")
    print(f"   Length: {len(response)} chars")
    print(f"   Contains 'hello': {'hello' in response.lower()}")
