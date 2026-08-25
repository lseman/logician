# Legroom SDK Worker Protocol Extensions

## Goal

Extend `legroom.sdk_worker` (the JSONL subprocess protocol) to expose all **compression-relevant** capabilities that the proxy exposes, so the SDK mode can match the proxy mode feature-for-feature without requiring an HTTP server.

---

## Current Protocol (v1)

### Request
```json
{
  "id": "string",
  "method": "compress",
  "messages": [{ "role": "...", "content": "..." }],
  "model": "string",
  "config": { /* CompressConfig fields */ }
}
```

### Response
```json
{
  "id": "string",
  "ok": true,
  "messages": [{ "role": "...", "content": "..." }],
  "stats": {
    "tokens_before": 0,
    "tokens_after": 0,
    "tokens_saved": 0,
    "transforms_applied": [],
    "warnings": []
  }
}
```

### Limitations (the gaps)

| Gap | Proxy has it | SDK worker lacks |
|---|---|---|
| **CCR store** | Full `CompressionStore` + `ccr_retrieve` tool injection + server-side resolution loop | Explicitly rejects `ccr_enabled: true` |
| **Compression result cache** | `CompressionResultCache` keyed by (protocol, model, messages, policy) | Every call recomputes |
| **Calibration** | `CalibrationController` — phase-level quality tracking, auto-disable | No quality feedback loop |
| **Quality evaluation** | Injected `Callable[[msgs, compressed], float]` | No quality feedback |
| **Provider cache** | `ProviderCachePolicy` — adds `prompt_cache_key` / `input_token_ids` to outbound requests | Only compresses messages, doesn't touch request body |
| **Metrics** | Prometheus `/metrics`, `/api/stats`, `/api/history` | No aggregated metrics |
| **Config fields** | `shadow_mode`, `strict`, `disabled_phases` | Only `CompressConfig` fields pass through |

---

## Design Decisions

1. **Backward compatible** — existing `compress` requests continue to work unchanged.
2. **New method names** — each capability gets its own JSON-RPC-style `method` field.
3. **Stateful worker** — the worker maintains a `CompressionStore` and `CompressionResultCache` in-process. These are scoped to the worker lifecycle (same as the current persistent process).
4. **Metadata in response** — the `stats` object is extended with `metadata` carrying phase reports, CCR hashes, salience scores, calibration state.
5. **No HTTP features** — proxy-specific features (upstream forwarding, streaming, health checks, dashboard, SSE/WebSocket) are inherently HTTP and stay proxy-only.
6. **Provider cache is proxy-only** — the provider cache feature modifies the outbound HTTP request body (`prompt_cache_key`, `input_token_ids`). The SDK worker only handles messages. This gap is architectural: the SDK worker doesn't own the request body, so it can't inject cache controls. The runtime (agent-bridge) would need to handle this separately.

---

## Protocol Extension (v2)

### 1. Extended `compress` response

Add optional fields to the `stats` object:

```json
{
  "id": "legroom-1234-1",
  "ok": true,
  "messages": [...],
  "stats": {
    "tokens_before": 15000,
    "tokens_after": 8200,
    "tokens_saved": 6800,
    "transforms_applied": ["read_lifecycle", "cross_turn_dedup", "compress"],
    "warnings": ["read_lifecycle: 3 stale, 1 superseded reads compressed"],
    "metadata": {
      "ccr_hashes": ["a3f8b2c1", "d4e5f6a7"],
      "phase_reports": [
        {
          "name": "read_lifecycle",
          "status": "applied",
          "confidence": 1.0,
          "tokens_before": 15000,
          "tokens_after": 12000,
          "details": { "reads_stale": 3, "reads_superseded": 1, "reads_fresh": 5 }
        },
        {
          "name": "cross_turn_dedup",
          "status": "applied",
          "confidence": 1.0,
          "tokens_before": 12000,
          "tokens_after": 10500,
          "details": { "dedup_count": 7 }
        }
      ],
      "salience_scores_before": [0.9, 0.7, 0.5, ...],
      "salience_scores_after": [0.9, 0.75, 0.55, ...],
      "calibration": {
        "disabled_phases": [],
        "snapshots": [
          { "phase": "compress", "samples": 42, "success_rate": 0.95, "disabled": false }
        ]
      }
    }
  }
}
```

### 2. New methods

#### `compress` (extended)

Same request shape, but now supports additional config fields:

```json
{
  "id": "legroom-1234-1",
  "method": "compress",
  "messages": [...],
  "model": "gpt-4o",
  "config": {
    "optimize": true,
    "protect_recent": 3,
    "ccr_enabled": true,
    "shadow_mode": false,
    "strict": false,
    "disabled_phases": ["output_shaping"],
    "calibration": {
      "min_samples": 20,
      "window_size": 200,
      "minimum_success_rate": 0.75,
      "minimum_quality": 0.95
    }
  }
}
```

New config fields:
- `ccr_enabled: boolean` — enable CCR (Compressed Content Retrieval). When true, the worker creates/manages an in-process `CompressionStore` and injects CCR markers + system instructions.
- `shadow_mode: boolean` — measure compression and quality without mutating outbound context. Returns original messages but includes `stats.metadata.shadow_mode = true` with quality/compression data.
- `strict: boolean` — raise on phase failures instead of falling back.
- `disabled_phases: string[]` — phase names to skip.
- `calibration: CalibrationConfig` — calibration window settings.

#### `compress_with_store` (new)

For callers that need CCR with a persistent store. The worker creates a named store and returns its ID:

```json
// Request
{
  "id": "legroom-1234-2",
  "method": "compress_with_store",
  "store_id": "session-abc",
  "messages": [...],
  "model": "gpt-4o",
  "config": {
    "ccr_enabled": true,
    "protect_recent": 3
  }
}

// Response
{
  "id": "legroom-1234-2",
  "ok": true,
  "messages": [...],
  "stats": {
    "tokens_before": 15000,
    "tokens_after": 8200,
    "tokens_saved": 6800,
    "transforms_applied": ["read_lifecycle", "compress", "ccr_tool_injection"],
    "metadata": {
      "ccr_hashes": ["a3f8b2c1", "d4e5f6a7"],
      "store_id": "session-abc",
      "store_stats": {
        "entries": 2,
        "total_bytes_before": 45000,
        "total_bytes_after": 8200,
        "savings": 36800
      }
    }
  }
}
```

#### `store_retrieve` (new)

Query the CCR store for compressed content:

```json
// Request
{
  "id": "legroom-1234-3",
  "method": "store_retrieve",
  "store_id": "session-abc",
  "hash": "a3f8b2c1"
}

// Response
{
  "id": "legroom-1234-3",
  "ok": true,
  "content": "The full original uncompressed text..."
}

// Not found
{
  "id": "legroom-1234-3",
  "ok": false,
  "error": "hash not found: a3f8b2c1"
}
```

#### `store_stats` (new)

Get CCR store statistics:

```json
// Request
{
  "id": "legroom-1234-4",
  "method": "store_stats",
  "store_id": "session-abc"
}

// Response
{
  "id": "legroom-1234-4",
  "ok": true,
  "stats": {
    "entries": 42,
    "max_entries": 1000,
    "total_bytes_before": 2500000,
    "total_bytes_after": 450000,
    "savings": 2050000
  }
}
```

#### `cache_get` (new)

Query the compression result cache:

```json
// Request
{
  "id": "legroom-1234-5",
  "method": "cache_get",
  "key": "base64-encoded-cache-key"
}

// Response
{
  "id": "legroom-1234-5",
  "ok": true,
  "hit": true,
  "messages": [...],
  "stats": {
    "tokens_before": 15000,
    "tokens_after": 8200,
    "transforms_applied": ["read_lifecycle", "compress"]
  }
}

// Miss
{
  "id": "legroom-1234-5",
  "ok": true,
  "hit": false
}
```

#### `calibration_record` (new)

Record quality feedback for calibration. The caller (runtime) evaluates compressed vs original quality and reports it back:

```json
// Request
{
  "id": "legroom-1234-6",
  "method": "calibration_record",
  "phase_reports": [
    { "name": "read_lifecycle", "status": "applied" },
    { "name": "compress", "status": "applied" }
  ],
  "quality": 0.97
}

// Response
{
  "id": "legroom-1234-6",
  "ok": true,
  "calibration": {
    "disabled_phases": [],
    "snapshots": [
      { "phase": "read_lifecycle", "samples": 15, "success_rate": 0.93, "disabled": false },
      { "phase": "compress", "samples": 42, "success_rate": 0.98, "disabled": false }
    ]
  }
}
```

#### `calibration_status` (new)

Query current calibration state:

```json
// Request
{
  "id": "legroom-1234-7",
  "method": "calibration_status"
}

// Response
{
  "id": "legroom-1234-7",
  "ok": true,
  "calibration": {
    "disabled_phases": ["output_shaping"],
    "snapshots": [
      { "phase": "output_shaping", "samples": 150, "success_rate": 0.62, "disabled": true },
      { "phase": "compress", "samples": 42, "success_rate": 0.98, "disabled": false }
    ]
  }
}
```

#### `worker_stats` (new)

Aggregate worker statistics (replaces proxy's `/api/stats`):

```json
// Request
{
  "id": "legroom-1234-8",
  "method": "worker_stats"
}

// Response
{
  "id": "legroom-1234-8",
  "ok": true,
  "stats": {
    "total_requests": 1247,
    "total_tokens_before": 18500000,
    "total_tokens_after": 9200000,
    "total_tokens_saved": 9300000,
    "compression_ratio": 50.3,
    "total_reads_stale": 342,
    "total_reads_superseded": 89,
    "total_reads_fresh": 1816,
    "strategy_counts": {
      "read_lifecycle": 1247,
      "cross_turn_dedup": 1180,
      "compress": 1247,
      "thinking_compactor": 1100
    },
    "cache_hits": 342,
    "cache_misses": 905,
    "uptime_seconds": 86400
  }
}
```

#### `worker_history` (new)

Recent request history (replaces proxy's `/api/history`):

```json
// Request
{
  "id": "legroom-1234-9",
  "method": "worker_history",
  "limit": 10,
  "offset": 0
}

// Response
{
  "id": "legroom-1234-9",
  "ok": true,
  "history": [
    {
      "request_id": "legroom-1234-1",
      "timestamp": 1697184000.0,
      "model": "gpt-4o",
      "messages_before": 12,
      "tokens_before": 15000,
      "tokens_after": 8200,
      "tokens_saved": 6800,
      "transforms_applied": ["read_lifecycle", "compress"],
      "warnings": []
    }
  ],
  "total": 1247
}
```

### 3. Updated `sdk_worker.py`

The `_response` function dispatches on `method`:

```python
_METHOD_HANDLERS = {
    "compress": _handle_compress,
    "compress_with_store": _handle_compress_with_store,
    "store_retrieve": _handle_store_retrieve,
    "store_stats": _handle_store_stats,
    "cache_get": _handle_cache_get,
    "calibration_record": _handle_calibration_record,
    "calibration_status": _handle_calibration_status,
    "worker_stats": _handle_worker_stats,
    "worker_history": _handle_worker_history,
}
```

### 4. Worker state

The worker maintains:

```python
class LegroomWorkerState:
    stores: dict[str, CompressionStore]  # store_id -> store
    cache: CompressionResultCache        # compression result cache
    calibration: CalibrationController   # phase calibration
    metrics: ProxyMetrics                # aggregate stats
    history: deque[RequestEvent]         # recent requests
    started_at: float                    # process start time
```

Stores are created on first `compress_with_store` call with a given `store_id`, or lazily on `compress` with `ccr_enabled: true` (uses a default store named `"_default"`).

### 5. Config field mapping

New config fields accepted by the SDK worker:

| Field | Type | Default | Description |
|---|---|---|---|
| `ccr_enabled` | `bool` | `false` | Enable CCR store + tool injection |
| `shadow_mode` | `bool` | `false` | Don't mutate messages, return originals with stats |
| `strict` | `bool` | `false` | Raise on phase failures |
| `disabled_phases` | `string[]` | `[]` | Phase names to skip |
| `calibration` | `CalibrationConfig` | default | Calibration window settings |

These are merged into the `CompressConfig` before calling `compress()`.

---

## Implementation Plan

### Phase 1: Core extensions (compression parity)

1. **Extend `sdk_worker.py`** — add dispatch for new methods, worker state class.
2. **Add `compress_with_store`** — creates/manages `CompressionStore`, enables CCR.
3. **Add `store_retrieve` / `store_stats`** — CCR store query methods.
4. **Add `cache_get`** — compression result cache query.
5. **Extend compress response** — add `metadata` with phase reports, CCR hashes, salience scores.
6. **Accept new config fields** — `ccr_enabled`, `shadow_mode`, `strict`, `disabled_phases`.

### Phase 2: Observability

7. **Add `worker_stats`** — aggregate request/compression metrics.
8. **Add `worker_history`** — recent request history.
9. **Add `calibration_record` / `calibration_status`** — phase quality tracking.

### Phase 3: TypeScript worker client updates

10. **Update `LegroomWorker` in `worker.ts`** — add methods for new protocol calls.
11. **Update `buildLegroomHooks` in `agent-bridge.ts`** — use `compress_with_store` when CCR is needed.
12. **Add quality feedback loop** — if the runtime can evaluate compression quality, call `calibration_record`.

---

## What Stays Proxy-Only

These features are inherently HTTP/proxy and have no SDK equivalent:

| Feature | Why proxy-only |
|---|---|
| Upstream forwarding | The SDK worker doesn't make HTTP requests to LLM APIs |
| Streaming response passthrough | No upstream connection to stream |
| Health checks (`/livez`, `/readyz`) | HTTP liveness probes |
| Dashboard (`/`) | HTTP HTML serving |
| SSE / WebSocket live events | HTTP transport features |
| Provider cache injection | Modifies outbound HTTP request body; SDK worker only handles messages |
| CCR tool call resolution loop | Proxy resolves `ccr_retrieve` tool calls server-side; in SDK mode the runtime owns tool resolution |

---

## TypeScript Client Changes (worker.ts)

```typescript
export class LegroomWorker {
  // Existing
  async compress(messages, model): Promise<Record<string, unknown>[]>

  // New
  async compressWithStore(
    storeId: string,
    messages: Record<string, unknown>[],
    model: string,
  ): Promise<CompressWithStoreResult>

  async storeRetrieve(storeId: string, hash: string): Promise<string | null>

  async storeStats(storeId: string): Promise<StoreStats>

  async cacheGet(key: string): Promise<CacheEntry | null>

  async calibrationRecord(
    phaseReports: PhaseReport[],
    quality: number,
  ): Promise<CalibrationStatus>

  async calibrationStatus(): Promise<CalibrationStatus>

  async workerStats(): Promise<WorkerStats>

  async workerHistory(limit?: number, offset?: number): Promise<WorkerHistory>
}
```

---

## Request/Response Type Definitions

```python
# Python side (sdk_worker.py)

@dataclass
class CompressWithStoreResult:
    messages: list[dict[str, Any]]
    stats: CompressionStats
    metadata: dict[str, Any]  # ccr_hashes, store_id, store_stats

@dataclass
class StoreStats:
    entries: int
    max_entries: int
    total_bytes_before: int
    total_bytes_after: int
    savings: int

@dataclass
class CacheEntry:
    hit: bool
    messages: list[dict[str, Any]] | None
    stats: CompressionStats | None

@dataclass
class CalibrationStatus:
    disabled_phases: tuple[str, ...]
    snapshots: tuple[CalibrationSnapshot, ...]

@dataclass
class WorkerStats:
    total_requests: int
    total_tokens_before: int
    total_tokens_after: int
    total_tokens_saved: int
    compression_ratio: float
    total_reads_stale: int
    total_reads_superseded: int
    total_reads_fresh: int
    strategy_counts: dict[str, int]
    cache_hits: int
    cache_misses: int
    uptime_seconds: float

@dataclass
class WorkerHistory:
    history: list[RequestEvent]
    total: int
```

---

## Edge Cases & Error Handling

1. **Store not found** — `store_retrieve` and `store_stats` return `{ok: false, error: "store not found: <id>"}` for unknown store IDs.
2. **Shadow mode + CCR** — if both `shadow_mode: true` and `ccr_enabled: true`, the worker still populates the store and returns CCR hashes, but returns the original (uncompressed) messages.
3. **Calibration with no quality data** — `calibration_status` returns empty snapshots until `calibration_record` has been called.
4. **Cache key generation** — the worker generates cache keys the same way the proxy does (`CompressionResultCache.key()`), so cache hits work across proxy and SDK modes for identical inputs.
5. **CCR tool injection in SDK mode** — the worker injects `ccr_retrieve` system instructions into the messages (same as proxy's `CCRToolInjector`), but does NOT inject the tool definition itself. The runtime (agent-bridge) must add the `ccr_retrieve` tool to the provider request, same as it does now for the proxy mode. The worker returns `ccr_hashes` in metadata so the runtime knows which hashes are available.
6. **Max history** — bounded at 1000 entries (same as proxy's `ProxyState.max_history`).
7. **Store max entries** — each `CompressionStore` has `max_entries=1000` (same as proxy default).

---

## Migration Path

1. **Current behavior preserved** — existing `compress` requests with `ccr_enabled` absent or `false` work exactly as before.
2. **Gradual rollout** — the runtime can opt into new methods by setting `ccr_enabled: true` in config. Old SDK workers (without CCR support) will reject it with the existing error; the runtime's `failOpen` mode handles this gracefully.
3. **Protocol version** — optionally add a `version` field to requests/responses for future-proofing:
   ```json
   { "id": "...", "method": "compress", "version": 2, ... }
   ```
   Workers that don't recognize `version` ignore it (forward-compatible).

---

## Summary of Changes

| Component | Changes |
|---|---|
| `legroom/sdk_worker.py` | New dispatch table, worker state class, 8 new handler functions, extended config validation |
| `legroom/compress.py` | No changes — all new features compose through existing `compress()` + config |
| `legroom/pipeline.py` | No changes — CCR, calibration, cache are already pipeline phases |
| `packages/log-runtime/.../legroom/worker.ts` | New methods for each protocol call, typed request/response |
| `packages/log-runtime/.../agent-bridge.ts` | Use `compressWithStore` when CCR needed, wire quality feedback to `calibration_record` |
