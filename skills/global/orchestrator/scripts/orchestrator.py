"""Autonomous task orchestration and planning skill.

Uses SSR (Socratic Self-Refinement) to decompose complex requests into
the global todo list.
"""

from __future__ import annotations

import json
from typing import Any

def plan_complex_task(request: str) -> str:
    """Use when: The user gives a broad, multi-step goal (e.g., 'build a new backend feature').
    
    Triggers: decompose, orchestrate, complex task, multi-step, planning, architecture.
    Avoid when: Task is clearly single-step or already defined.
    
    Inputs:
      request (str): The full user request to decompose.
      
    Returns: JSON confirming task initialization and the rendered todo list.
    """
    from src.reasoners.ssr import SSRReasoner
    
    # We use a custom reasoner instance for planning
    reasoner = SSRReasoner(llm)
    
    blueprint_query = f"""
Decompose this request into a logical sequence of engineering tasks for an AI agent.
Request: {request}

Requirement:
- Each task must be actionable via tools (shell, file edit, search, etc.).
- Group related steps.
- Include verification steps.
- Limit to 5-10 high-level tasks.

Return ONLY a JSON list of objects: {{"title": "...", "note": "..."}}
"""
    
    trace = reasoner.solve(blueprint_query)
    
    try:
        # Reasoner output usually contains reasoning + answer
        # We look for the JSON array in the answer or reasoning
        text = trace.answer or trace.reasoning
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if match:
            tasks_data = json.loads(match.group(0))
        else:
            # Fallback parsing
            tasks_data = json.loads(text.strip("` \n"))
            
        if not isinstance(tasks_data, list):
            return json.dumps({"status": "error", "error": "Reasoner did not return a list of tasks"})
            
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Failed to parse plan: {e}", "raw": trace.answer})
    
    # Now push to the todo skill
    # Note: We rely on the todo skill being available in the same runtime
    # or we call it via the globals if registered.
    # Since skills are loaded into the same interpreter for a specific agent instance,
    # we can import the todo function if it was already defined or use the registry.
    
    try:
        from .m_20_todo import todo
        
        # Format for todo("set")
        todo_items = []
        for i, t in enumerate(tasks_data, 1):
            todo_items.append({
                "id": i,
                "title": t.get("title", f"Task {i}"),
                "status": "not-started",
                "note": t.get("note", "")
            })
            
        res = todo("set", items=todo_items)
        return res
        
    except ImportError:
        return json.dumps({
            "status": "warning", 
            "message": "Plan generated but 'todo' skill not found for auto-initialization",
            "plan": tasks_data
        })
    except Exception as e:
         return json.dumps({"status": "error", "error": f"Failed to initialize todo list: {e}"})

import re
__all__ = ["plan_complex_task"]


__tools__ = [plan_complex_task]
