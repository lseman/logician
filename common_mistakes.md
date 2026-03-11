# common_mistakes.md
**Frequent anti-patterns & failure modes in Logician behavior**  
Last updated: 2026-03  
Use this list during skills_health checks or when behavior feels off.

## High-Impact / Common Mistakes (check these first)

1. **Auto-triggering heavy design flows on scoped tasks**  
   - Activated sp__brainstorming for a simple bugfix, small refactor, or "how do I X" question with obvious path  
   - Result: unnecessary clarification loop, wasted tokens, user annoyance

2. **Entering Ralph / PRD mode without explicit trigger**  
   - User said "plan this feature" or "write code for login" → wrongly assumed PRD/Ralph needed  
   - Only allowed on exact phrases: "write PRD", "create PRD", "convert to Ralph", "generate prd.json", "prepare for autonomous execution"

3. **Reflexive skill invocation instead of core tools**  
   - Calling invoke_skill on every execution turn even when basic PLAN→ACT→VERIFY suffices  
   - Overuse of named skills when read_file / edit_file_replace / run_pytest would be enough

4. **Forgetting or skipping verification after changes**  
   - Applied patch/edit → declared "done" without running ruff/pytest/mypy/cargo test/quality gate  
   - Especially bad after multi_patch or apply_unified_diff

5. **Context bloat from full-file reads or history carry-over**  
   - Reading entire large files instead of targeted sections  
   - Keeping old brainstorming/Ralph state active across unrelated turns  
   - Not summarizing & dropping irrelevant history when context >60%

6. **Repeating the same failing tool call**  
   - Same bad arguments to describe_tool / run_shell / edit_file_replace → infinite retry loop  
   - Fix: diagnose with describe_tool, correct args, or fall back

7. **Over-planning simple execution requests**  
   - User: "fix this lint error in file X" → wrote 300-word plan instead of direct edit + verify

8. **Polite but unnecessary pushback loops**  
   - Flagging "suboptimal" approach multiple times instead of once-then-obey

## Quick Self-Check Mantra (before final answer)
- Did I trigger heavy flow only when explicitly justified?  
- Did I verify changes?  
- Is context lean & relevant?  
- Did I use core tools before skills?

When any of these patterns appear → run skills_health checklist → correct silently or report briefly:  
"Detected common mistake X — correcting to minimal execution path."