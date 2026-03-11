# skills_health.md
**Diagnostic checklist & quick health report for Logician's skill routing & invocation behavior**
Last updated: 2026-03

## Purpose
Run this checklist mentally (or output its sections) whenever:
- Skill routing feels off / unexpected
- A skill was invoked without clear user intent
- Repeated failed or looping tool/skill calls
- User reports "you're using the wrong workflow" or similar

Goal: Quickly detect misrouting, over-triggering, stale state, or context confusion.

## Step-by-Step Health Check (run in order)

1. **Currently Loaded / Active Skills**
   - Which skills or guidance cards are currently in context? (List names)
   - Were any auto-loaded without explicit user request?
   → If yes → note which and why (e.g. previous turn carried over)

2. **Last 3–5 Skill Invocations**
   - Skill name | User turn summary | Trigger condition | Outcome (success / fail / loop)
   - Example:
     sp__brainstorming | "how should I structure auth" | design intent matched → activated | waiting for approval
     sp__ralph | "fix this bug" | NO explicit PRD/Ralph phrase → should NOT have triggered | misfire

3. **Routing Decision for Current Turn**
   - Turn classification: social / informational / execution / design / prd
   - Did classification match user intent? (Y/N + reason)
   - Was invoke_skill called? If yes: exact trigger phrase or intent match
   - Was heavy flow (brainstorming / Ralph / multi-step skill) triggered? Justified?
   → Red flag if triggered on social/informational or minor scoped execution

4. **Context Contamination Check**
   - Previous turn(s) still influencing routing? (e.g. Ralph mode carried over)
   - Any dangling state from earlier skills? (e.g. brainstorming waiting for approval)
   - Context window pressure high? (>60% → summarize & drop irrelevant history)

5. **Common Misrouting Patterns (self-check)**
   - [ ] Activated brainstorming on scoped bugfix / refactor
   - [ ] Entered Ralph loop without "PRD", "prd.json", "Ralph format" phrase
   - [ ] Invoked named skill reflexively instead of core tools
   - [ ] Kept old skill state active across unrelated turns
   - [ ] Over-invoked planning/brainstorming on clear execution requests
   - [ ] Failed to fall back to PLAN→ACT→VERIFY when skill routing failed

6. **Recovery Actions (pick the least invasive that fits)**
   - Clear stale skill state → respond without invoking previous mode
   - Re-classify current turn → reply using only core workflow
   - Explicitly say: "Detected possible routing error — resetting to core execution mode"
   - Run describe_tool on any failing tool → correct args
   - If still confused → ask user one focused clarification question
   - Worst case: "Routing health check failed — please restate goal or disable skills temporarily"

## Quick One-Liner Report Template (use when diagnosing aloud)
"skills_health: loaded=[list], last_invoc=[skill], classification=[type], red_flags=[count], action=taking=[recovery step]"

Use this file as a mental or outputted reference — do not load it into every context.
Only reference / quote relevant sections when behavior is drifting.
