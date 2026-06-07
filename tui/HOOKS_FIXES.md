# Hooks System Fixes

## Bug #1: `withDrainHook` shadows user hooks (Critical)

**File:** `src/agent-core/harness.ts`
**Lines:** 432–488
**Severity:** Critical — user hooks never run

### Problem

`withDrainHook` intercepts `prepareNextTurn`, `afterToolCall`, `getSteeringMessages`, `getFollowUpMessages` and replaces them entirely. User hooks are stored in local vars (`userPrepare`, `userAfterToolCall`) then dropped:

```typescript
// Current (broken):
return {
    ...config,
    hooks: {
        ...userHooks,                    // spreads keys, but next line overwrites them
        getSteeringMessages,             // shadows userHooks.getSteeringMessages
        getFollowUpMessages,             // shadows userHooks.getFollowUpMessages
        prepareNextTurn,                 // shadows userHooks.prepareNextTurn
        afterToolCall,                   // shadows userHooks.afterToolCall
    },
};
```

The `...userHooks` spread is useless — the keys following it have identical names and win. User `prepareNextTurn` runs zero times. Same for `afterToolCall`.

### Fix

Use `HookBus` to compose queue-drain handlers with user hooks, same pattern `composeHooks` uses for built-in + user hooks:

```typescript
import { HookBus } from "./hook-bus.ts";

private withDrainHook(config: AgentConfig): AgentConfig {
    const userHooks: AgentLoopHooks = config.hooks || {};

    // Build a bus and register user hooks first, then queue-drain handlers.
    const bus = new HookBus();
    bus.register(userHooks, { source: "user" });

    // Queue-drain handlers register second — they see user hooks' output.
    bus.register(
        {
            getSteeringMessages: async (ctx) => {
                let out: string[];
                if (this.steeringQueueMode === "all") {
                    out = this.steeringQueue.splice(0);
                } else {
                    const first = this.steeringQueue[0];
                    out = first ? [first] : [];
                    if (first) this.steeringQueue.shift();
                }
                if (!out.length) return undefined;
                return out.map((text) => createUserMessage(text));
            },
            getFollowUpMessages: async (ctx) => {
                let out: string[];
                if (this.followUpQueueMode === "all") {
                    out = this.followUpQueue.splice(0);
                } else {
                    const first = this.followUpQueue[0];
                    out = first ? [first] : [];
                    if (first) this.followUpQueue.shift();
                }
                if (!out.length) return undefined;
                return out.map((text) => createUserMessage(text));
            },
            getNextTurnMessages: () => {
                const pending = this.nextTurnQueue.splice(0);
                if (!pending.length) return undefined;
                return pending.map((text) => createUserMessage(text));
            },
        },
        { source: "harness-queues" }
    );

    return { ...config, hooks: bus.toHooks() };
}
```

This eliminates the `userPrepare`, `userAfterToolCall` local vars, removes the shadowing, and composes deterministically via the bus reducers.

---

## Bug #2: `getNextTurnMessages` typed but never wired (High)

**File:** `src/agent-core/types.ts:248–254`
**Severity:** High — API surface exists but is dead code

### Problem

`AgentLoopHooks.getNextTurnMessages` is typed. `AgentHarness.nextTurnQueue` and `AgentHarness.nextTurn()` exist. But `getNextTurnMessages` never calls into the queue system. The loop uses `drainNextTurnMessages()` directly, bypassing the hook bus entirely.

### Fix

Remove raw `nextTurnQueue` from the harness. Route through `getNextTurnMessages` via `HookBus` (see fix above). This gives plugins the same access point:

```typescript
// In harness constructor: remove nextTurnQueue field entirely.
// nextTurn() method: push to a message list, not a raw string array.
nextTurn(text: string): void {
    this.nextTurnMessages.push(text);
}

// getNextTurnMessages handler in HookBus (see fix above):
getNextTurnMessages: () => {
    const pending = this.nextTurnMessages.splice(0);
    if (!pending.length) return undefined;
    return pending.map((text) => ({ role: "user" as const, content: text }));
}
```

Update all callers of `harness.nextTurnQueue` in `agent-bridge.ts`:

```typescript
// agent-bridge.ts:454
nextTurn(message: string): void {
    this.harness?.nextTurn(message);  // unchanged, now routed via hooks
}

// agent-bridge.ts:475, 517, 529, 532
// Replace direct nextTurnQueue access with:
const nextTurnMessages = this.harness?.getMessagesFromHooks?.("nextTurn") || [];
```

---

## Bug #3: `Notification` event is dead code (Low)

**File:** `src/agent-core/plugins.ts:18`
**Severity:** Low — harmless but confusing

### Problem

`Notification` is declared in `HookEventType` union but no code calls `runHookSafely("Notification", ...)`. It's never fired.

### Fix

Remove from union:

```typescript
type HookEventType =
    | "SessionStart"
    | "SessionEnd"
    | "Setup"
    | "Stop"
    // | "Notification"  // removed — never fired
    | "UserPromptSubmit"
    | "PreToolUse"
    | "PostToolUse"
    | "PreCompact"
    | "PostCompact";
```

---

## Bug #4: Plugin hooks and contract hooks are two separate systems (Medium)

**Files:** `src/agent-core/plugins.ts`, `src/agent-core/hook-bus.ts`
**Severity:** Medium — developer must maintain two hook definitions for same moments

### Problem

Plugin hooks fire shell commands (preToolUse, postToolUse) via JSONL payloads. Contract hooks fire typed TS callbacks (beforeToolCall, afterToolCall). Same events, different APIs. A plugin can't rewrite tool args via `beforeToolCall` — it must also maintain a shell-side hook.

### Fix

Expose `HookBus` publicly so plugins can register typed handlers. Plugin shell commands become a `HookCommandType` that translates into typed callbacks:

```typescript
// hook-bus.ts: export the bus class and add a convenience method
export class HookBus {
    // ... existing code ...

    /** Register a shell-command hook that injects context as a typed handler. */
    registerShellHook(
        eventType: keyof AgentLoopHooks,
        command: string,
        options?: { timeout?: number; source?: string }
    ): () => void {
        return this.on(eventType, async (ctx) => {
            // Execute shell command, parse JSON output, return typed result
            const result = await executeShellCommand(command, ctx, options);
            return parseResultToType(eventType, result);
        }, { source: options?.source || "shell" });
    }
}
```

Plugin manifest hooks field then registers via this API instead of firing independent shell processes:

```json
{
    "name": "my-plugin",
    "hooks": {
        "PreToolUse": [{
            "matcher": "read_file|write_file",
            "hooks": [{
                "type": "command",
                "command": "node audit-read.js"
            }]
        }]
    }
}
```

The plugin loader translates `"type": "command"` into `HookBus.registerShellHook()`.

---

## Bug #5: `composeHooks` runs per-turn (Low)

**File:** `src/agent-core/builtin-hooks.ts:169`
**Severity:** Low — wasted allocation, no persistent state

### Problem

`buildBuiltinHooks` + `composeHooks` run inside `AgentLoop.run()`. Each turn creates a new `HookBus`, new `GuardEngine`, new `BudgetTracker`. No state persists across turns.

### Fix

Move `HookBus` construction to `AgentLoop` constructor. Keep built-in state (guards, budget, compaction cooldown) per-turn by rebuilding those objects in `run()` but reusing the bus.

```typescript
// agent-core/loop.ts
class AgentLoop {
    private hookBus: HookBus;

    constructor(options: AgentLoopOptions) {
        // Build bus once, per-loop lifecycle.
        this.hookBus = new HookBus();
    }

    async run(userMessage: string): Promise<Message[]> {
        // Per-turn: rebuild built-in hooks (they hold per-turn state).
        const builtin = buildBuiltinHooks({ /* deps */ });
        this.hookBus.register(builtin, { source: "builtin" });
        this.hookBus.register(this.config.hooks, { source: "user" });
        // ... run turn ...
        this.hookBus.clear(); // reset for next turn
    }
}
```

---

## Bug #6: `afterToolCall` with `terminate` broken by shadowing (High)

**File:** `src/agent-core/harness.ts:484–488`
**Severity:** High — `terminate` from user hooks is never observed

### Problem

Same shadowing as bug #1. The harness creates a wrapper `afterToolCall` that calls `userAfterToolCall` but then passes it through `userHooks` spread which shadows it. The `terminate` flag set by user hooks is never seen by the loop.

### Fix

Eliminated by the `HookBus` composition in bug #1 fix. When user hooks register via `bus.register()`, the bus's reducer handles `terminate` correctly:

```typescript
// hook-bus.ts: runAfter reducer
private async runAfter(ctx: AfterToolCallContext): Promise<AfterToolCallResult | undefined> {
    // Each handler sees the prior patch. terminate from ANY handler wins.
    let current = ctx;
    let terminate = false;
    for (const { handler } of this.after) {
        const r = await this.guard(() => handler(current), "afterToolCall", source);
        if (!r) continue;
        terminate = terminate || r.terminate === true;
        current = {
            ...current,
            result: r.content ?? current.result,
            isError: r.isError ?? current.isError,
        };
    }
    return { content: current.result, isError: current.isError, terminate };
}
```

---

## Bug #7: Plugin hooks can't observe hook execution (Low)

**File:** `src/agent-core/loop.ts:937`
**Severity:** Low — fire-and-forget, all output discarded

### Problem

`runHookSafely` catches all errors and discards all output. Plugin hooks fire, produce JSON, and it's gone. No way to react to a plugin hook's output from the loop or other plugins.

### Fix

Plugin hooks already have their own output mechanism (`additional_contexts`, `context_messages` via JSONL parsing). This is acceptable for the shell-command model. The fix is documentation: plugin hooks are fire-and-forget observers, not interactive participants. If plugins need interactivity, they should use the typed HookBus (bug #4 fix).

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/agent-core/harness.ts` | Replace `withDrainHook` — use HookBus, remove shadowing, wire `getNextTurnMessages` |
| `src/agent-core/hook-bus.ts` | Add `registerShellHook()`, export `HookBus` publicly |
| `src/agent-core/plugins.ts` | Remove `Notification` from `HookEventType` |
| `src/agent-core/loop.ts` | Move `HookBus` to constructor, clear per-turn |
| `src/agent-core/builtin-hooks.ts` | Update `composeHooks` to use shared bus pattern |
| `src/agent-bridge.ts` | Replace direct `nextTurnQueue` access with hook bus routing |

---

## Implementation Order

1. **Fix #4** (HookBus export + registerShellHook) — enables everything else
2. **Fix #1** (withDrainHook → HookBus) — removes critical shadowing
3. **Fix #2** (getNextTurnMessages wiring) — completes the queue routing
4. **Fix #6** (terminate in afterToolCall reducer) — ensures batch termination works
5. **Fix #5** (bus lifetime) — cleanup, not blocking
6. **Fix #3** (remove Notification) — dead code removal
