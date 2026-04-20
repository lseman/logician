#!/usr/bin/env python3
"""
Diagnose LLM backend issues by testing the raw LLM without agent framework.
"""

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.backends import LlamaCppClient
from src.messages import Message, MessageRole


def test_raw_llm():
    """Test the LLM directly without any agent logic."""
    print("=" * 60)
    print("🔬 Testing Raw LLM Backend")
    print("=" * 60)

    # Create a minimal LLM client
    llm = LlamaCppClient(
        base_url="http://localhost:8080",
        timeout=30.0,
        use_chat_api=True,
        chat_template="chatml",
        stop=[],  # Empty list instead of None
        retry_attempts=1,
    )

    # Test 1: Simple prompt
    print("\n📝 Test 1: Simple Hello")
    print("-" * 60)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Say hello in one sentence."),
    ]

    try:
        response = llm.generate(
            messages,
            temperature=0.3,
            max_tokens=50,
            stream=False,
        )
        print(f"Response: {response}")
        print(f"Length: {len(response)} chars")

        # Check if response is relevant
        if "hello" in response.lower() or "hi" in response.lower():
            print("✅ Response is relevant")
        else:
            print("❌ Response is IRRELEVANT")
            print("   Expected: greeting")
            print(f"   Got: {response[:100]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    # Test 2: Math question
    print("\n📝 Test 2: Simple Math")
    print("-" * 60)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is 2+2? Answer in one sentence."),
    ]

    try:
        response = llm.generate(
            messages,
            temperature=0.1,
            max_tokens=30,
            stream=False,
        )
        print(f"Response: {response}")

        if "4" in response or "four" in response.lower():
            print("✅ Response is correct")
        else:
            print("❌ Response is WRONG or IRRELEVANT")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎯 Diagnosis Complete")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_raw_llm()

    if not success:
        print("\n⚠️  LLM Backend Issue Detected!")
        print("\nPossible causes:")
        print("1. Wrong model loaded (not instruction-tuned)")
        print("2. Model is too small (< 3B parameters)")
        print("3. Server configuration issue")
        print("4. Chat template mismatch")
        print("\nRecommendations:")
        print("- Check what model is running on localhost:8080")
        print("- Try a different chat template (llama2, mistral, etc.)")
        print("- Verify the model is instruction-following capable")
