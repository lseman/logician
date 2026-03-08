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
    Phase {
        state: String,
        note: String,
    },
    Tool {
        name: String,
        args: Value,
    },
    Image {
        tool: String,
        path: String,
    },
    Skill {
        skill_ids: Vec<String>,
        selected_tools: Vec<String>,
    },
    Stderr(String),
    Exit(Option<i32>),
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
    pub skill_count: u64,
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
            "phase" => BridgeEvent::Phase {
                state: v["state"].as_str().unwrap_or("ready").to_string(),
                note: v
                    .get("note")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
            "tool" => BridgeEvent::Tool {
                name: v["name"].as_str().unwrap_or("unknown").to_string(),
                args: v.get("args").cloned().unwrap_or_default(),
            },
            "image" => BridgeEvent::Image {
                tool: v["tool"].as_str().unwrap_or("unknown").to_string(),
                path: v["path"].as_str().unwrap_or("").to_string(),
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
            "bridge_stderr" => BridgeEvent::Stderr(v["text"].as_str().unwrap_or("").to_string()),
            "bridge_exit" => {
                BridgeEvent::Exit(v.get("code").and_then(|c| c.as_i64()).map(|c| c as i32))
            }
            _ => return,
        };

        let _ = event_tx.send(bridge_event);
    }
}
