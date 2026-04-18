use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

// ── Events emitted by the Python process ─────────────────────────────────────

#[derive(Debug, Clone)]
pub enum BridgeEvent {
    Token(String),
    ThinkingToken(String),
    Phase {
        state: String,
        note: String,
    },
    ToolStart {
        name: String,
        args: Option<Value>,
        sequence: u64,
    },
    ToolEnd {
        name: String,
        sequence: u64,
        status: String,
        duration_ms: u64,
        cache_hit: bool,
        error: Option<String>,
        result_preview: Option<String>,
        result_output: Option<String>,
        args: Option<Value>, // Include args for consistency with tool_start
    },
    Image {
        tool: String,
        path: String,
    },
    Skill {
        skill_ids: Vec<String>,
        selected_tools: Vec<String>,
    },
    Decision {
        mode: String,
        stage: String,
        message: String,
    },
    ToolRepair {
        stage: String,
        attempt: u64,
        tool: String,
        error_type: String,
        message: String,
    },
    Stderr(String),
    Exit(Option<i32>),
    FileDiff {
        /// Debugging field: tracks which tool generated the diff (e.g., "edit_file", "write_file").
        /// Currently unused in rendering but preserved for future debugging.
        #[allow(dead_code)]
        tool: String,
        path: String,
        diff: String,
    },
    /// Lifecycle state update for a subsystem (mcp or plugin)
    #[allow(dead_code)]
    Lifecycle {
        subsystem: String,
        payload: Value,
    },
    /// Compact session memory (compaction event)
    #[allow(dead_code)]
    Compaction {
        payload: Value,
    },
    /// Session summary event
    #[allow(dead_code)]
    Summary {
        payload: Value,
    },
}

#[derive(Debug, Clone, Default)]
pub struct TodoItem {
    pub title: String,
    pub status: String,
    pub note: String,
}

#[derive(Debug, Clone, Default)]
pub struct ContextMount {
    pub path: String,
    pub glob: String,
    pub file_count: u64,
    pub token_count: u64,
    pub map_depth: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RagDoc {
    pub path: String,
    pub label: String,
    pub chunks: u64,
    pub token_count: u64,
    pub kind: String,
}

#[derive(Debug, Clone, Default)]
pub struct RepoRecord {
    pub id: String,
    pub name: String,
    pub path: String,
    pub files_processed: u64,
    pub chunks_added: u64,
    pub graph_nodes: u64,
    pub graph_edges: u64,
    pub graph_symbols: u64,
    pub branch: String,
    pub commit: String,
    pub last_ingested_at: String,
    pub last_graph_built_at: String,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RetrievalFileRecord {
    pub rel_path: String,
    pub score: u64,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RetrievalSymbolRecord {
    pub name: String,
    pub symbol_kind: String,
    pub rel_path: String,
    pub line: u64,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RetrievalInsight {
    pub query: String,
    pub repo_id: String,
    pub repo_name: String,
    pub seed_paths: Vec<String>,
    pub retrieved_paths: Vec<String>,
    pub related_files: Vec<RetrievalFileRecord>,
    pub related_symbols: Vec<RetrievalSymbolRecord>,
}

#[derive(Debug, Clone, Default)]
pub struct RagInventoryRepo {
    pub repo_id: String,
    pub repo_name: String,
    pub chunks: u64,
    pub files: u64,
    pub last_ingested_at: String,
}

#[derive(Debug, Clone, Default)]
pub struct RagInventory {
    pub vector_path: String,
    pub vector_backend: String,
    pub active_doc_count: u64,
    pub active_doc_chunks: u64,
    pub repo_count: u64,
    pub active_repo_count: u64,
    pub repo_chunks: u64,
    pub retrieval_count: u64,
    pub legacy_paths: Vec<String>,
    pub top_repos: Vec<RagInventoryRepo>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RagQueryHit {
    pub repo_name: String,
    pub repo_id: String,
    pub path: String,
    pub distance: String,
}

#[derive(Debug, Clone, Default)]
pub struct RecentRagQuery {
    pub query: String,
    pub top_k: u64,
    pub count: u64,
    pub repo_filter: Vec<String>,
    pub hits: Vec<RagQueryHit>,
}

// ── Bridge state returned by the Python process ───────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct BridgeState {
    pub active: String,
    pub session: String,
    pub msg_count: u64,
    pub agents: Vec<String>,
    pub mcp_servers: Vec<String>,
    pub pipeline: Option<Value>,
    pub rapidfuzz: bool,
    pub tiktoken: bool,
    pub tool_count: u64,
    #[allow(dead_code)]
    pub skill_count: u64,
    pub loaded_tools: Vec<String>,
    pub loaded_skills: Vec<String>,
    pub todo: Vec<TodoItem>,
    pub active_repos: Vec<RepoRecord>,
    pub retrieval_insights: Vec<RetrievalInsight>,
    pub repo_library: Vec<RepoRecord>,
    pub mounted_paths: Vec<ContextMount>,
    pub rag_docs: Vec<RagDoc>,
    pub rag_inventory: RagInventory,
    pub recent_rag_queries: Vec<RecentRagQuery>,
    pub tool_call_count: u64,
    pub plan_mode: bool,
    pub lifecycle: Option<Value>,
}

impl BridgeState {
    pub fn from_value(v: &Value) -> Self {
        Self {
            active: v["active"].as_str().unwrap_or("-").to_string(),
            session: v["session"].as_str().unwrap_or("-").to_string(),
            msg_count: v["msg_count"].as_u64().unwrap_or(0),
            agents: v["agents"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            mcp_servers: v["mcp_servers"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            pipeline: v.get("pipeline").filter(|p| !p.is_null()).cloned(),
            rapidfuzz: v["rapidfuzz"].as_bool().unwrap_or(false),
            tiktoken: v["tiktoken"].as_bool().unwrap_or(false),
            tool_count: v["tool_count"].as_u64().unwrap_or(0),
            skill_count: v["skill_count"].as_u64().unwrap_or(0),
            loaded_tools: v["loaded_tools"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            loaded_skills: v["loaded_skills"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            todo: v["todo"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| TodoItem {
                            title: x["title"]
                                .as_str()
                                .or_else(|| x["content"].as_str())
                                .unwrap_or("")
                                .to_string(),
                            status: x["status"].as_str().unwrap_or("not-started").to_string(),
                            note: x["note"]
                                .as_str()
                                .or_else(|| x["activeForm"].as_str())
                                .unwrap_or("")
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            active_repos: v["active_repos"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| RepoRecord {
                            id: x["id"].as_str().unwrap_or("").to_string(),
                            name: x["name"].as_str().unwrap_or("").to_string(),
                            path: x["path"].as_str().unwrap_or("").to_string(),
                            files_processed: x["files_processed"].as_u64().unwrap_or(0),
                            chunks_added: x["chunks_added"].as_u64().unwrap_or(0),
                            graph_nodes: x["graph_nodes"].as_u64().unwrap_or(0),
                            graph_edges: x["graph_edges"].as_u64().unwrap_or(0),
                            graph_symbols: x["graph_symbols"].as_u64().unwrap_or(0),
                            branch: x["branch"]
                                .as_str()
                                .or_else(|| x["git"]["branch"].as_str())
                                .unwrap_or("")
                                .to_string(),
                            commit: x["commit"]
                                .as_str()
                                .or_else(|| x["git"]["commit"].as_str())
                                .unwrap_or("")
                                .to_string(),
                            last_ingested_at: x["last_ingested_at"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                            last_graph_built_at: x["last_graph_built_at"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                        })
                        .filter(|item| !item.id.trim().is_empty() || !item.path.trim().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            retrieval_insights: v["retrieval_insights"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| RetrievalInsight {
                            query: x["query"].as_str().unwrap_or("").to_string(),
                            repo_id: x["repo_id"].as_str().unwrap_or("").to_string(),
                            repo_name: x["repo_name"].as_str().unwrap_or("").to_string(),
                            seed_paths: x["seed_paths"]
                                .as_array()
                                .map(|items| {
                                    items
                                        .iter()
                                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            retrieved_paths: x["retrieved_paths"]
                                .as_array()
                                .map(|items| {
                                    items
                                        .iter()
                                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            related_files: x["related_files"]
                                .as_array()
                                .map(|items| {
                                    items
                                        .iter()
                                        .map(|item| RetrievalFileRecord {
                                            rel_path: item["rel_path"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            score: item["score"].as_u64().unwrap_or(0),
                                        })
                                        .filter(|item| !item.rel_path.trim().is_empty())
                                        .collect()
                                })
                                .unwrap_or_default(),
                            related_symbols: x["related_symbols"]
                                .as_array()
                                .map(|items| {
                                    items
                                        .iter()
                                        .map(|item| RetrievalSymbolRecord {
                                            name: item["name"].as_str().unwrap_or("").to_string(),
                                            symbol_kind: item["symbol_kind"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            rel_path: item["rel_path"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            line: item["line"].as_u64().unwrap_or(0),
                                        })
                                        .filter(|item| !item.name.trim().is_empty())
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                        .filter(|item| {
                            !item.repo_id.trim().is_empty()
                                || !item.repo_name.trim().is_empty()
                                || !item.query.trim().is_empty()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            repo_library: v["repo_library"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| RepoRecord {
                            id: x["id"].as_str().unwrap_or("").to_string(),
                            name: x["name"].as_str().unwrap_or("").to_string(),
                            path: x["path"].as_str().unwrap_or("").to_string(),
                            files_processed: x["files_processed"].as_u64().unwrap_or(0),
                            chunks_added: x["chunks_added"].as_u64().unwrap_or(0),
                            graph_nodes: x["graph_nodes"].as_u64().unwrap_or(0),
                            graph_edges: x["graph_edges"].as_u64().unwrap_or(0),
                            graph_symbols: x["graph_symbols"].as_u64().unwrap_or(0),
                            branch: x["git"]["branch"].as_str().unwrap_or("").to_string(),
                            commit: x["git"]["commit"].as_str().unwrap_or("").to_string(),
                            last_ingested_at: x["last_ingested_at"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                            last_graph_built_at: x["last_graph_built_at"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                        })
                        .filter(|item| !item.id.trim().is_empty() || !item.path.trim().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            mounted_paths: v["mounted_paths"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| ContextMount {
                            path: x["path"].as_str().unwrap_or("").to_string(),
                            glob: x["glob"].as_str().unwrap_or("").to_string(),
                            file_count: x["file_count"].as_u64().unwrap_or(0),
                            token_count: x["token_count"].as_u64().unwrap_or(0),
                            map_depth: x["map_depth"].as_u64().unwrap_or(0),
                        })
                        .filter(|item| !item.path.trim().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            rag_docs: v["rag_docs"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| RagDoc {
                            path: x["path"].as_str().unwrap_or("").to_string(),
                            label: x["label"].as_str().unwrap_or("").to_string(),
                            chunks: x["chunks"].as_u64().unwrap_or(0),
                            token_count: x["token_count"].as_u64().unwrap_or(0),
                            kind: x["kind"].as_str().unwrap_or("").to_string(),
                        })
                        .filter(|item| !item.path.trim().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            rag_inventory: RagInventory {
                vector_path: v["rag_inventory"]["vector_path"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                vector_backend: v["rag_inventory"]["vector_backend"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                active_doc_count: v["rag_inventory"]["active_doc_count"]
                    .as_u64()
                    .unwrap_or(0),
                active_doc_chunks: v["rag_inventory"]["active_doc_chunks"]
                    .as_u64()
                    .unwrap_or(0),
                repo_count: v["rag_inventory"]["repo_count"].as_u64().unwrap_or(0),
                active_repo_count: v["rag_inventory"]["active_repo_count"]
                    .as_u64()
                    .unwrap_or(0),
                repo_chunks: v["rag_inventory"]["repo_chunks"].as_u64().unwrap_or(0),
                retrieval_count: v["rag_inventory"]["retrieval_count"]
                    .as_u64()
                    .unwrap_or(0),
                legacy_paths: v["rag_inventory"]["legacy_paths"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                top_repos: v["rag_inventory"]["top_repos"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .map(|item| RagInventoryRepo {
                                repo_id: item["repo_id"].as_str().unwrap_or("").to_string(),
                                repo_name: item["repo_name"].as_str().unwrap_or("").to_string(),
                                chunks: item["chunks"].as_u64().unwrap_or(0),
                                files: item["files"].as_u64().unwrap_or(0),
                                last_ingested_at: item["last_ingested_at"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string(),
                            })
                            .filter(|item| {
                                !item.repo_id.trim().is_empty()
                                    || !item.repo_name.trim().is_empty()
                                    || item.chunks > 0
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            },
            recent_rag_queries: v["recent_rag_queries"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .map(|item| RecentRagQuery {
                            query: item["query"].as_str().unwrap_or("").to_string(),
                            top_k: item["top_k"].as_u64().unwrap_or(0),
                            count: item["count"].as_u64().unwrap_or(0),
                            repo_filter: item["repo_filter"]
                                .as_array()
                                .map(|values| {
                                    values
                                        .iter()
                                        .filter_map(|value| value.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            hits: item["hits"]
                                .as_array()
                                .map(|values| {
                                    values
                                        .iter()
                                        .map(|value| RagQueryHit {
                                            repo_name: value["repo_name"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            repo_id: value["repo_id"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            path: value["path"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                            distance: match value.get("distance") {
                                                Some(Value::String(text)) => text.clone(),
                                                Some(other) => other.to_string(),
                                                None => String::new(),
                                            },
                                        })
                                        .filter(|value| {
                                            !value.path.trim().is_empty()
                                                || !value.repo_name.trim().is_empty()
                                                || !value.repo_id.trim().is_empty()
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                        .filter(|item| !item.query.trim().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            tool_call_count: v["tool_call_count"].as_u64().unwrap_or(0),
            plan_mode: v["plan_mode"].as_bool().unwrap_or(false),
            lifecycle: v.get("lifecycle").filter(|p| !p.is_null()).cloned(),
        }
    }
}

// ── PythonBridge ─────────────────────────────────────────────────────────────

type PendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>>;

#[derive(Clone)]
pub struct PythonBridge {
    stdin: Arc<Mutex<ChildStdin>>,
    pending: PendingMap,
}

impl PythonBridge {
    /// Spawn the bridge process and return a handle + child guard.
    pub async fn spawn(
        program: &str,
        args: &[&str],
        event_tx: mpsc::UnboundedSender<BridgeEvent>,
    ) -> Result<(Self, Child)> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));

        // ── stdout reader task ──────────────────────────────────────────────
        {
            let pending = pending.clone();
            let event_tx = event_tx.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) if !line.trim().is_empty() => {
                            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                                Self::route(&v, &pending, &event_tx).await;
                            }
                        }
                        Ok(None) | Err(_) => break,
                        _ => {}
                    }
                }
                // Drain pending senders so all in-flight call()s unblock immediately
                // with Err("bridge response channel closed") instead of hanging.
                pending.lock().await.drain();
                let _ = event_tx.send(BridgeEvent::Exit(None));
            });
        }

        // ── stderr reader task ──────────────────────────────────────────────
        {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = event_tx.send(BridgeEvent::Stderr(line));
                }
            });
        }

        Ok((
            Self {
                stdin: Arc::new(Mutex::new(stdin)),
                pending,
            },
            child,
        ))
    }

    /// Send a fire-and-forget cancel notification to the Python process.
    /// The bridge is single-threaded so cancellation is best-effort — the Rust
    /// side drops the in-flight oneshot receiver, making the UI responsive
    /// immediately while the Python agent finishes its current turn in background.
    pub async fn send_cancel(&self) {
        let msg = json!({"method": "cancel"});
        let line = format!("{}\n", serde_json::to_string(&msg).unwrap_or_default());
        let _ = self.stdin.lock().await.write_all(line.as_bytes()).await;
    }

    /// Send a JSON-RPC call and await the result.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = Uuid::new_v4().to_string();
        let msg = json!({"id": &id, "method": method, "params": params});
        let line = format!("{}\n", serde_json::to_string(&msg)?);

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.stdin.lock().await.write_all(line.as_bytes()).await?;

        rx.await
            .map_err(|_| anyhow!("bridge response channel closed"))?
    }

    async fn route(v: &Value, pending: &PendingMap, event_tx: &mpsc::UnboundedSender<BridgeEvent>) {
        // Responses carry an "id" field.
        if let Some(id) = v.get("id").and_then(|i| i.as_str()) {
            if let Some(tx) = pending.lock().await.remove(id) {
                let result = if let Some(err) = v.get("error") {
                    Err(anyhow!("{}", err))
                } else {
                    Ok(v.get("result").cloned().unwrap_or(Value::Null))
                };
                let _ = tx.send(result);
            }
            return;
        }

        // Events carry an "event" field.
        let ev = match v.get("event").and_then(|e| e.as_str()) {
            Some(e) => e,
            None => return,
        };

        let bridge_event = match ev {
            "token" => BridgeEvent::Token(v["token"].as_str().unwrap_or("").to_string()),
            "thinking_token" => {
                BridgeEvent::ThinkingToken(v["token"].as_str().unwrap_or("").to_string())
            }
            "phase" => BridgeEvent::Phase {
                state: v["state"].as_str().unwrap_or("ready").to_string(),
                note: v
                    .get("note")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
            "tool_start" => BridgeEvent::ToolStart {
                name: v["name"].as_str().unwrap_or("unknown").to_string(),
                args: Some(v.get("args").cloned().unwrap_or_default()),
                sequence: v["sequence"].as_u64().unwrap_or(0),
            },
            "tool_end" => BridgeEvent::ToolEnd {
                name: v["name"].as_str().unwrap_or("unknown").to_string(),
                sequence: v["sequence"].as_u64().unwrap_or(0),
                status: v["status"].as_str().unwrap_or("ok").to_string(),
                duration_ms: v["duration_ms"].as_u64().unwrap_or(0),
                cache_hit: v["cache_hit"].as_bool().unwrap_or(false),
                error: v
                    .get("error")
                    .and_then(|e| e.as_str())
                    .map(|e| e.to_string()),
                result_preview: v
                    .get("result_preview")
                    .and_then(|p| p.as_str())
                    .filter(|p| !p.is_empty())
                    .map(|p| p.to_string()),
                result_output: v
                    .get("result_output")
                    .and_then(|p| p.as_str())
                    .map(|p| p.to_string()),
                args: Some(v.get("args").cloned().unwrap_or_default()),
            },
            "tool" => BridgeEvent::ToolStart {
                // Backward compatibility for older bridge event schema.
                name: v["name"].as_str().unwrap_or("unknown").to_string(),
                args: Some(v.get("args").cloned().unwrap_or_default()),
                sequence: v["sequence"].as_u64().unwrap_or(0),
            },
            "image" => BridgeEvent::Image {
                tool: v["tool"].as_str().unwrap_or("unknown").to_string(),
                path: v["path"].as_str().unwrap_or("").to_string(),
            },
            "file_diff" => BridgeEvent::FileDiff {
                tool: v["tool"].as_str().unwrap_or("unknown").to_string(),
                path: v["path"].as_str().unwrap_or("").to_string(),
                diff: v["diff"].as_str().unwrap_or("").to_string(),
            },
            "skill" => BridgeEvent::Skill {
                skill_ids: v["skill_ids"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                selected_tools: v["selected_tools"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
            },
            "decision" => BridgeEvent::Decision {
                mode: v["mode"].as_str().unwrap_or("").to_string(),
                stage: v["stage"].as_str().unwrap_or("").to_string(),
                message: v["message"].as_str().unwrap_or("").to_string(),
            },
            "tool_repair" => BridgeEvent::ToolRepair {
                stage: v["stage"].as_str().unwrap_or("").to_string(),
                attempt: v["attempt"].as_u64().unwrap_or(0),
                tool: v["tool"].as_str().unwrap_or("unknown").to_string(),
                error_type: v["error_type"].as_str().unwrap_or("").to_string(),
                message: v["message"].as_str().unwrap_or("").to_string(),
            },
            "bridge_stderr" => BridgeEvent::Stderr(v["text"].as_str().unwrap_or("").to_string()),
            "bridge_exit" => {
                BridgeEvent::Exit(v.get("code").and_then(|c| c.as_i64()).map(|c| c as i32))
            }
            "lifecycle" => BridgeEvent::Lifecycle {
                subsystem: v["subsystem"].as_str().unwrap_or("").to_string(),
                payload: v.get("payload").cloned().unwrap_or_default(),
            },
            "compaction" => BridgeEvent::Compaction {
                payload: v.get("payload").cloned().unwrap_or_default(),
            },
            "summary" => BridgeEvent::Summary {
                payload: v.get("payload").cloned().unwrap_or_default(),
            },
            _ => return,
        };

        let _ = event_tx.send(bridge_event);
    }
}
