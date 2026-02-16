#!/usr/bin/env python3
"""
Test different chat templates to find the right one.
"""

from src.backends import LlamaCppClient
from src.messages import Message, MessageRole

TEMPLATES = ["chatml", "llama2", "mistral", "alpaca", "vicuna", "zephyr"]

def test_template(template_name):
    """Test a specific chat template."""
    print(f"\n{'='*60}")
    print(f"Testing template: {template_name}")
    print('='*60)
    
    try:
        llm = LlamaCppClient(
            base_url="http://localhost:8080",
            timeout=10.0,
            use_chat_api=True,
            chat_template=template_name,
            stop=[],
            retry_attempts=1,
        )
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Say 'OK' and nothing else."),
        ]
        
        response = llm.generate(messages, temperature=0.1, max_tokens=10)
        print(f"Response: {response}")
        
        # Check if response is sane
        if len(response) < 50 and "ok" in response.lower():
            print(f"✅ Template '{template_name}' works!")
            return True
        else:
            print(f"❌ Template '{template_name}' failed (irrelevant response)")
            return False
            
    except Exception as e:
        print(f"❌ Template '{template_name}' error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Testing Chat Templates")
    print("This will help identify the correct template for your model.")
    
    working_templates = []
    for template in TEMPLATES:
        if test_template(template):
            working_templates.append(template)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    if working_templates:
        print(f"✅ Working templates: {', '.join(working_templates)}")
        print(f"\nUpdate your config to use: {working_templates[0]}")
    else:
        print("❌ NO templates worked!")
        print("\nThis means:")
        print("1. The model is not instruction-following capable")
        print("2. The model might be a base model (not chat/instruct)")
        print("3. You need to load a different model")
        print("\nRecommended models:")
        print("- Qwen2.5-7B-Instruct")
        print("- Mistral-7B-Instruct-v0.3")
        print("- Llama-3.2-3B-Instruct")
