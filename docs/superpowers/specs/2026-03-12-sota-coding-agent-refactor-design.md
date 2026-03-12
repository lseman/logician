# SOTA Coding Agent Refactor — Design Spec

**Date:** 2026-03-12
**Status:** Approved
**Author:** Laio Oriel Seman

---

## Overview

Refactor the Logician agent into a SOTA coding agent modeled after Claude Code and Codex CLI. The refactor is split into two sequential phases:

- **Phase A** — Deep architectural refactor: decompose the monolithic 5,256-line `core.py` into focused, testable components
- **Phase B** — Tool/skill surface: expand CC-parity tools, lazy-load domain groups, slim the prompt pipeline

The agent's backends, config dataclasses, database layer, memory, and test harness are **not changed** in this refactor. Backwards compatibility with existing callers is maintained via a thin `Agent` facade in `core.py`.

---

## Motivation

### Current Problems

1. `core.py` is 5,256 lines; the main loop alone is ~2,000 lines
2. ~50 local variables carry loop state — no single source of truth
3. ~15 guardrail checks are scattered `if` blocks inside the loop
4. System prompt is assembled by ad-hoc multi-injection — everything rendered every turn
5. Tool availability is conflated with skill routing — no stable "always-on" tools
6. No formal `LLMBackend` protocol — backends are duck-typed

### Goals

- Main loop ≤ 100 lines
- All loop state in one dataclass (`TurnState`)
- All guardrails in one pluggable engine (`GuardrailEngine`)
- System prompt assembled by a composable pipeline (`PromptBuilder`)
- Parallel tool dispatch as first-class behaviour (`ToolDispatcher`)
- 9 always-on core tools; domain tools lazy-activated per query
- Full test coverage for each new component in isolation

---

## Shared Data Types

These types are used across all components and must be defined first (in `src/agent/types.py`).

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None   # set when role == "tool"

@dataclass
class ToolCall:
    tool_name: str
    args: dict[str, Any]
    call_id: str   # unique per call; matches tool result back to call

@dataclass
class TurnResult:
    state: TurnState
    messages: list[Message]

    @property
    def final_response(self) -> str | None:
        return self.state.final_response

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.state.tool_calls
```

---

## Turn Classification

`classify_turn()` lives in `src/agent/classify.py`. It takes the last user message and returns a `TurnClassification`:

```python
@dataclass
class TurnClassification:
    intent: Literal["social", "informational", "execution", "design"]
    domain_groups: set[str]   # e.g. {"timeseries", "academic"}

# Keyword tables (order = priority; first match wins)
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("social",        ["hello", "hi", "thanks", "thank you", "bye"]),
    ("design",        ["design", "architect", "how should i structure", "propose", "trade-off"]),
    ("informational", ["explain", "what is", "how does", "describe", "why"]),
    ("execution",     []),   # default
]

_DOMAIN_PATTERNS: dict[str, list[str]] = {
    "timeseries": ["reservoir", "forecast", "ons", "time series", "hydroelectric", "energy data"],
    "academic":   ["paper", "citation", "s2", "ieee", "openalex", "literature", "review"],
    "rag":        ["ingest", "retrieve", "embed", "knowledge base", "vector store"],
    "svg":        ["diagram", "svg", "visual", "chart"],
}

def classify_turn(content: str) -> TurnClassification:
    intent = _match_intent(content.lower())
    domain_groups = _match_domains(content.lower())
    return TurnClassification(intent=intent, domain_groups=domain_groups)
```

`classify_turn()` is deterministic (no LLM calls). The intent match uses simple substring search; domain group detection is additive (multiple groups can activate in one turn).

---

## Architecture

### Core loop model

ReAct-style tight loop with external guardrails — the same model used by Claude Code and Codex CLI:

```
while iteration < max_iterations:
    system_prompt = PromptBuilder.build(state, config)
    response      = LLM.generate(messages, system_prompt)
    tool_calls    = parse_tool_calls(response)
    guard_result  = GuardrailEngine.run(state, response, tool_calls)

    if guard_result.hard_stop → set final_response, break
    if guard_result.nudge     → append nudge message, continue
    if not tool_calls         → set final_response, break

    results = ToolDispatcher.dispatch(tool_calls, state)   # parallel where safe
    messages.extend(format_tool_results(results))
    state.consecutive_tool_count += len(tool_calls)
    iteration += 1
```

No explicit phases (inspect/edit/verify). The LLM decides when to stop calling tools; guardrails enforce correctness externally.

---

## Phase A — File Structure

```
src/
├── agent/
│   ├── types.py         # Shared: Message, ToolCall, TurnResult
│   ├── classify.py      # TurnClassification, classify_turn()
│   ├── state.py         # TurnState dataclass
│   ├── guardrails.py    # GuardrailEngine + 6 Guard classes
│   ├── prompt.py        # PromptBuilder + PromptComponent chain
│   ├── dispatcher.py    # ToolDispatcher (parallel batch execution)
│   ├── loop.py          # AgentLoop (~100 lines)
│   └── core.py          # Thin Agent facade (backwards-compatible)
├── backends/
│   ├── base.py          # LLMBackend Protocol (new)
│   ├── llama_cpp.py     # Unchanged
│   └── vllm.py          # Unchanged
└── tools/
    ├── core/
    │   ├── files.py     # read_file, write_file, edit_file, apply_edit_block
    │   ├── shell.py     # bash
    │   ├── search.py    # glob, grep
    │   └── tasks.py     # todo, think
    └── registry/        # Existing mixins (unchanged except routing slim-down)
```

Skills directory layout is **unchanged**. Domain skill groups become lazy-loaded tool groups.

---

## Component Specs

### TurnState (`src/agent/state.py`)

Single dataclass passed through every component. Replaces all local loop variables.

```python
@dataclass
class TurnState:
    turn_id: str

    # Loop control
    iteration: int = 0
    consecutive_tool_count: int = 0   # incremented by dispatcher; reset when LLM produces no tools

    # Tool tracking
    # tool_calls: append-only ordered log; preserves call sequence for VerificationGuard
    tool_calls: list[ToolCall] = field(default_factory=list)
    # seen_signatures: maps "tool_name:sha256(args)" → call count for threshold-based guards
    seen_signatures: dict[str, int] = field(default_factory=dict)
    files_written: list[str] = field(default_factory=list)
    domain_groups_activated: set[str] = field(default_factory=set)

    # Guardrail state
    guardrail_nudges: dict[str, int] = field(default_factory=dict)   # guard_name → nudge count

    # Turn classification
    classified_as: Literal["social", "informational", "execution", "design"] = "execution"

    # Output
    final_response: str | None = None
    trace: list[dict] = field(default_factory=list)
```

**Invariants:**
- `tool_calls` is append-only and preserves chronological order across the full turn
- `seen_signatures` maps signature strings to occurrence counts (not a set); used by `DuplicateToolGuard`
- `consecutive_tool_count` is incremented by `ToolDispatcher.dispatch()` (by count of calls dispatched) and reset to 0 in `AgentLoop` whenever `tool_calls` parsed from a response is empty
- `files_written` accumulates path strings whenever `write_file`, `edit_file`, or `apply_edit_block` succeeds; never cleared mid-turn
- `guardrail_nudges` tracks how many times each guard has nudged; used for escalation to hard_stop

---

### GuardrailEngine (`src/agent/guardrails.py`)

```python
class Guard(Protocol):
    name: str
    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult: ...

@dataclass
class GuardrailResult:
    passed: bool
    nudge: str | None = None   # Appended as user message to steer LLM
    hard_stop: bool = False    # Breaks the loop immediately

class GuardrailEngine:
    def __init__(self, guards: list[Guard]) -> None: ...

    def run(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        """Run ALL guards regardless of order. Aggregate result:
        - If any guard returns hard_stop=True → return hard_stop (ignore nudges)
        - Else if any guard returns passed=False → return the first nudge (by guard list order)
        - Else → return passed=True
        """
        results = [g.check(state, response, tool_calls) for g in self.guards]
        hard_stops = [r for r in results if r.hard_stop]
        if hard_stops:
            return hard_stops[0]
        failures = [r for r in results if not r.passed]
        if failures:
            return failures[0]
        return GuardrailResult(passed=True)
```

**Aggregation rule:** All guards always run. `hard_stop` takes priority over nudge; among equal priority, list order determines which guard's message is returned. This prevents a cheap guard from masking a hard_stop guard that runs later.

**Built-in guards:**

| Guard | Condition | Action |
|---|---|---|
| `DuplicateToolGuard` | `seen_signatures[sig] >= 2` | nudge; hard stop when `>= 3` |
| `ConsecutiveToolGuard` | `state.consecutive_tool_count > config.max_consecutive_tool_calls` | nudge to produce answer |
| `ToolClaimGuard` | Response text asserts tool ran; `tool_calls` is empty | nudge to use actual tool |
| `VerificationGuard` | `files_written` non-empty; no verification tool in `tool_calls` after last write index | nudge to verify |
| `StallGuard` | `guardrail_nudges["stall"] >= 2` and same no-tool response | hard stop |
| `InspectionGuard` | Disabled by default; enabled via `config.enable_inspection_guard` |  |

**`VerificationGuard` implementation note:** "after last write index" means the guard checks `state.tool_calls[last_write_index+1:]` for any tool name matching `config.verification_tool_patterns`. `last_write_index` is the position of the most recent write tool in `state.tool_calls`. This is implementable because `tool_calls` is an ordered list.

**`InspectionGuard` implementation note:** This guard requires an LLM call to detect contradictions — it is disabled by default (`config.enable_inspection_guard = False`). When enabled, the guard receives the LLM backend via constructor injection. It calls a cheap one-shot LLM prompt: "Does this response contradict the content of the files listed below? Answer YES or NO." The guard's constructor signature is `InspectionGuard(llm: LLMBackend, config: Config)`. Because this guard is opt-in and its complexity is self-contained, it does not conflict with the `Guard` protocol — the protocol does not restrict constructor args.

Guards are passed as a list at agent init — custom guards can be added without modifying core code.

---

### PromptBuilder (`src/agent/prompt.py`)

```python
class PromptComponent(Protocol):
    def render(self, state: TurnState, config: Config) -> str | None: ...

class PromptBuilder:
    def __init__(self, components: list[PromptComponent]) -> None: ...

    def build(self, state: TurnState, config: Config) -> str:
        parts = [
            rendered
            for comp in self.components
            if (rendered := comp.render(state, config))
        ]
        return "\n\n".join(parts)
```

**Default component chain (in order):**

1. **`IdentityComponent`** — SOUL.md content; static, cached at startup
2. **`CoreToolSchemasComponent`** — schemas for the 9 always-on tools
3. **`DomainToolsComponent`** — adds domain group schemas when `state.domain_groups_activated` is non-empty; returns `None` otherwise
4. **`SkillPlaybookComponent`** — injects slim guidance for the top-1 BM25-routed skill; returns `None` when `config.enable_skill_routing` is false or no skill scores above threshold
5. **`TurnContextComponent`** — files written this turn, iteration, verification status; returns `None` when all values are default (iteration=0, files_written=[])

Each component returns `None` when it has nothing to add — no empty sections in the final prompt.

---

### ToolDispatcher (`src/agent/dispatcher.py`)

Executes tool calls, batching parallel-safe calls into `asyncio.gather`.

```python
@dataclass
class DispatchResult:
    tool_name: str
    call_id: str
    output: str
    error: str | None = None
    duration_ms: int = 0

# Tools classified as read-only (parallel-safe). Hardcoded; not configurable.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file", "glob", "grep", "think",
})
# All other tools (including bash) are treated as write tools and run serially.
# bash is conservative-default serial because it may have side effects.

class ToolDispatcher:
    def __init__(self, registry: ToolRegistry) -> None: ...

    async def dispatch(
        self,
        calls: list[ToolCall],
        state: TurnState,
    ) -> list[DispatchResult]:
        """
        Split calls into batches: [parallel_reads..., serial_writes...].
        Execute parallel batch with asyncio.gather, then serial tools one-by-one.
        After each successful write tool, append path to state.files_written.
        After all tools, increment state.consecutive_tool_count by len(calls).
        Update state.tool_calls and state.seen_signatures.
        """
        ...
```

**Parallelism rule:** `_READ_ONLY_TOOLS` is hardcoded (not config-driven) to avoid touching `config.py` (a non-goal). `bash` is always serial by default. If a future caller needs parallel bash, that is a separate spec.

**State updates performed by dispatcher:**
- `state.tool_calls.extend(calls)` — preserves order
- For each call: `state.seen_signatures[sig] = state.seen_signatures.get(sig, 0) + 1`
- For each successful write tool: `state.files_written.append(path_from_args)`
- `state.consecutive_tool_count += len(calls)`

---

### AgentLoop (`src/agent/loop.py`)

```python
class AgentLoop:
    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        guardrails: GuardrailEngine,
        prompt_builder: PromptBuilder,
        dispatcher: ToolDispatcher,
        config: Config,
    ) -> None: ...

    async def run(self, messages: list[Message]) -> TurnResult:
        classification = classify_turn(messages[-1].content)
        state = TurnState(
            turn_id=str(uuid4()),
            classified_as=classification.intent,
            domain_groups_activated=classification.domain_groups,
        )

        if classification.intent in ("social", "informational"):
            return await self._fast_path(messages, state)

        while state.iteration < self.config.max_iterations:
            system = self.prompt_builder.build(state, self.config)
            response = await self.llm.generate(messages, system_prompt=system)

            tool_calls = parse_tool_calls(response)

            # Reset consecutive count when LLM produces no tool calls
            if not tool_calls:
                state.consecutive_tool_count = 0

            guard_result = self.guardrails.run(state, response, tool_calls)
            if guard_result.hard_stop:
                state.final_response = response
                break
            if guard_result.nudge:
                messages.append(Message(role="user", content=guard_result.nudge))
                state.guardrail_nudges[guard_result.guard_name] = (
                    state.guardrail_nudges.get(guard_result.guard_name, 0) + 1
                )
                state.iteration += 1
                continue

            if not tool_calls:
                state.final_response = response
                break

            results = await self.dispatcher.dispatch(tool_calls, state)
            messages.extend(format_tool_results(results))
            state.iteration += 1

        return TurnResult(state=state, messages=messages)
```

**Note:** `GuardrailResult` gains a `guard_name: str` field so `AgentLoop` can update `state.guardrail_nudges` correctly.

**`Agent` facade in `core.py`** remains the public interface. It constructs `AgentLoop` with defaults and delegates `run()`. All existing callers, tests, and the Rust bridge are unaffected.

---

### LLMBackend Protocol (`src/backends/base.py`)

Formalises the interface both backends already implement:

```python
class LLMBackend(Protocol):
    async def generate(
        self,
        messages: list[Message],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        grammar: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...
```

Both `LlamaCppClient` and `VLLMClient` will be annotated to implement this protocol. No logic changes to either backend.

---

## Phase B — Tool Surface

### Always-on core tools (9 tools, always in schema)

Located in `src/tools/core/`. Promoted or new:

| Tool | Module | Status | Signature |
|---|---|---|---|
| `read_file` | `files.py` | Promote from skill | `(path, start_line?, end_line?)` |
| `write_file` | `files.py` | Promote from skill | `(path, content)` |
| `edit_file` | `files.py` | Promote from skill | `(path, old_string, new_string)` |
| `apply_edit_block` | `files.py` | Keep, promote | `(path, blocks)` |
| `bash` | `shell.py` | Promote from shell skill | `(command, timeout?)` |
| `glob` | `search.py` | New | `(pattern, path?)` |
| `grep` | `search.py` | New | `(pattern, path?, glob?, output_mode?)` |
| `todo` | `tasks.py` | New | `(todos)` |
| `think` | `tasks.py` | Keep, promote | `(thought)` |

### Lazy domain tool groups

Activated when `classify_turn()` detects trigger keywords. Domain tools are loaded into the registry and added to `DomainToolsComponent` for that turn only.

| Group | Trigger keywords | Source |
|---|---|---|
| `timeseries` | reservoir, forecast, ONS, time series, hydroelectric | `skills/timeseries/` |
| `academic` | paper, citation, S2, IEEE, OpenAlex, review | `skills/academic/` |
| `rag` | ingest, retrieve, embed, knowledge base | `skills/rag/` |
| `svg` | diagram, SVG, visual, chart | `skills/svg/` |

`classify_turn()` returns `domain_groups` as part of `TurnClassification`; `AgentLoop` sets `state.domain_groups_activated` before the loop starts. `DomainToolsComponent.render()` reads this field to decide what to inject.

### Skill routing changes

- BM25 routing retained; dense embeddings remain opt-in (`AGENT_SKILL_DENSE_ENABLED`)
- Routing injects *guidance text only* (slim `SkillPlaybookComponent`), never controls tool availability
- Top-1 skill selected (down from top-3) to keep prompt lean
- Domain group activation replaces the existing "domain skill" routing paths

---

## Testing Strategy

Each new component is tested in isolation before integration:

| Component | Test file | Key scenarios |
|---|---|---|
| `types.py` | `test/test_agent_types.py` | ToolCall construction, TurnResult properties |
| `classify_turn()` | `test/test_classify.py` | Each intent class, multi-domain detection, default to execution |
| `TurnState` | `test/test_turn_state.py` | seen_signatures count, files_written accumulation, consecutive reset |
| `GuardrailEngine` | `test/test_guardrails.py` | All-pass, first nudge wins, hard_stop overrides nudge, each guard fires independently |
| `PromptBuilder` | `test/test_prompt_builder.py` | None-returning components skipped, domain component conditional, component order |
| `ToolDispatcher` | `test/test_dispatcher.py` | Parallel read batch, serial write batch, state updates correct |
| `AgentLoop` | `test/test_agent_loop.py` | Full turn with FakeLLM; guardrail nudge path; hard_stop path; fast path; consecutive reset |
| Core tools | `test/test_core_tools.py` | glob patterns, grep output modes, bash timeout, edit_file exact match |

Existing tests in `test/test_agent_runtime_behavior.py` and `test/test_tool_registry_enhancements.py` must continue to pass via the `Agent` facade.

---

## Implementation Order

### Phase A
1. `src/agent/types.py` — Message, ToolCall, TurnResult (no deps)
2. `src/agent/classify.py` — TurnClassification, classify_turn() (no deps)
3. `src/agent/state.py` — TurnState (depends on types.py)
4. `src/backends/base.py` — LLMBackend Protocol (depends on types.py)
5. `src/agent/guardrails.py` — GuardrailEngine + 6 guards (depends on state.py, types.py)
6. `src/agent/prompt.py` — PromptBuilder + 5 components (depends on state.py)
7. `src/agent/dispatcher.py` — ToolDispatcher (depends on state.py, types.py)
8. `src/agent/loop.py` — AgentLoop (depends on all above)
9. `src/agent/core.py` — slim Agent facade delegating to AgentLoop
10. Tests for each component (in order matching above)
11. Verify all existing tests pass

### Phase B
1. `src/tools/core/files.py` — read_file, write_file, edit_file, apply_edit_block
2. `src/tools/core/shell.py` — bash
3. `src/tools/core/search.py` — glob, grep
4. `src/tools/core/tasks.py` — todo, think
5. Domain group lazy activation in `DomainToolsComponent` (classify_turn() already done in Phase A step 2)
6. Slim `SkillPlaybookComponent` (top-1, BM25 only)
7. Tests for core tools and domain activation

---

## Non-Goals

- No changes to `src/backends/llama_cpp.py` or `src/backends/vllm.py` internals
- No changes to `src/config.py` dataclasses (config fields remain identical; `_READ_ONLY_TOOLS` is hardcoded in dispatcher, not config-driven)
- No changes to `src/db/`, `src/memory.py`, `src/reasoners/`, `src/eoh/`, `src/mcp/`
- No changes to the Rust TUI or `logician_bridge.py`
- No changes to skills directory layout or SKILL.md format
- No removal of existing tool implementations in skill scripts (they remain; core tools are promoted copies)
