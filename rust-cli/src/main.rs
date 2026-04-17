use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, EventStream, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::process::Child;
use tokio::sync::mpsc;

mod app;
mod bridge;
mod image;
mod markdown;
mod ui;

use app::{App, KeyAction, ParsedSlashCommand, SlashDispatch};
use bridge::{BridgeEvent, PythonBridge};

fn resolve_default_bridge_script() -> String {
    if let Ok(explicit) = std::env::var("LOGICIAN_BRIDGE_SCRIPT") {
        if !explicit.trim().is_empty() {
            return explicit;
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("logician_bridge.py"));
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("logician_bridge.py"));
        candidates.push(cwd.join("cli").join("logician_bridge.py"));
        if let Some(parent) = cwd.parent() {
            candidates.push(parent.join("logician_bridge.py"));
            candidates.push(parent.join("cli").join("logician_bridge.py"));
        }
    }

    if let Some(path) = candidates.into_iter().find(|p| p.exists()) {
        return path.to_string_lossy().to_string();
    }

    "logician_bridge.py".to_string()
}

// ── CLI Arguments ───────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "logician")]
#[command(about = "Logician CLI - AI-assisted engineering agent")]
struct Args {
    /// Python command to execute bridge script
    #[arg(long, short = 'p', env = "PYTHON")]
    python_cmd: Option<String>,

    /// Bridge script path
    #[arg(long, short = 'b', env = "LOGICIAN_BRIDGE_SCRIPT")]
    bridge_script: Option<String>,
}

// ── Command results sent back to the main loop from spawned tasks ──────────────

enum CmdResult {
    Bootstrap(Result<(PythonBridge, Child), anyhow::Error>),
    Chat(Result<serde_json::Value, anyhow::Error>),
    Slash(Result<serde_json::Value, anyhow::Error>),
    State(Result<serde_json::Value, anyhow::Error>),
    WarmState(Result<serde_json::Value, anyhow::Error>),
    StartupContext(Result<serde_json::Value, anyhow::Error>),
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments with clap
    let args = Args::parse();

    // Resolve bridge command and script
    let bridge_cmd = args
        .python_cmd
        .or_else(|| std::env::var("PYTHON").ok())
        .unwrap_or_else(|| "python3".into());

    let bridge_script = args
        .bridge_script
        .or_else(|| std::env::var("LOGICIAN_BRIDGE_SCRIPT").ok())
        .unwrap_or_else(resolve_default_bridge_script);

    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.hide_cursor()?;

    let result = run(
        &mut terminal,
        &bridge_cmd,
        std::slice::from_ref(&bridge_script),
    )
    .await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture,
    )?;
    terminal.show_cursor()?;

    if let Err(ref e) = result {
        eprintln!("logician error: {e}");
    }

    result
}

// ── Main run loop ─────────────────────────────────────────────────────────────

async fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    bridge_cmd: &str,
    bridge_args: &[String],
) -> Result<()> {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<BridgeEvent>();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<CmdResult>();
    let bridge_args_refs: Vec<&str> = bridge_args.iter().map(|s| s.as_str()).collect();

    let mut app = App::new();
    app.phase = app::Phase::Thinking;
    app.phase_note = "starting bridge".into();

    let endpoint = std::env::var("OPENROUTER_URL")
        .or_else(|_| std::env::var("OPENROUTER_API_URL"))
        .or_else(|_| std::env::var("LLAMA_CPP_URL"))
        .unwrap_or_else(|_| "local".to_string());
    let provider = if endpoint.contains("openrouter") {
        "OpenRouter"
    } else if endpoint.contains("openai") {
        "OpenAI"
    } else if endpoint.contains("anthropic") {
        "Anthropic"
    } else {
        "Local"
    };
    let model = std::env::var("MODEL")
        .or_else(|_| std::env::var("AGENT_MODEL"))
        .unwrap_or_else(|_| "logician/bridge".to_string());
    let _version = env!("CARGO_PKG_VERSION");
    let startup_banner = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        "  _                 _      _           ",
        " | |    ___   __ _(_) ___(_) __ _ _ __ ",
        " | |   / _ \\ / _` | |/ __| |/ _` | '_ \\",
        " | |__| (_) | (_| | | (__| | (_| | | | |",
        " |_____\\___/ \\__, |_|\\___|_|\\__,_|_| |_|",
        "             |___/                      ",
        "",
        format!("Provider: {provider}"),
        format!("Model:    {model}"),
        format!("Endpoint: {endpoint}"),
        // format!("logician v{version}")
    );
    app.add_preformatted_system_message(startup_banner);
    app.add_system_message(format!(
        "Starting bridge `{bridge_cmd}` using {}",
        bridge_args.join(" ")
    ));

    {
        let tx = cmd_tx.clone();
        let event_tx = event_tx.clone();
        let bridge_cmd = bridge_cmd.to_string();
        let bridge_args_owned = bridge_args_refs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        tokio::spawn(async move {
            let arg_refs = bridge_args_owned
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>();
            let result = PythonBridge::spawn(&bridge_cmd, &arg_refs, event_tx).await;
            let _ = tx.send(CmdResult::Bootstrap(result));
        });
    }

    run_loop(terminal, app, None, &mut event_rx, &mut cmd_rx, cmd_tx).await
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    mut bridge: Option<PythonBridge>,
    event_rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    cmd_rx: &mut mpsc::UnboundedReceiver<CmdResult>,
    cmd_tx: mpsc::UnboundedSender<CmdResult>,
) -> Result<()> {
    let mut crossterm_events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut active_task: Option<tokio::task::AbortHandle> = None;
    let mut _bridge_child: Option<Child> = None;

    loop {
        // Draw
        terminal.draw(|f| ui::draw(f, &mut app))?;

        tokio::select! {
            biased;

            // ── Tick ─────────────────────────────────────────────────────────
            _ = tick.tick() => {
                app.tick();
            }

            // ── Bridge events ─────────────────────────────────────────────────
            Some(ev) = event_rx.recv() => {
                app.handle_bridge_event(ev);
            }

            // ── Command results ───────────────────────────────────────────────
            Some(cr) = cmd_rx.recv() => {
                if matches!(
                    cr,
                    CmdResult::Bootstrap(_)
                        | CmdResult::Chat(_)
                        | CmdResult::Slash(_)
                        | CmdResult::State(_)
                ) {
                    active_task = None;
                }
                match cr {
                    CmdResult::Bootstrap(r) => match r {
                        Ok((spawned_bridge, child)) => {
                            bridge = Some(spawned_bridge.clone());
                            _bridge_child = Some(child);
                            app.connected = true;
                            app.phase = app::Phase::Thinking;
                            app.phase_note = "initializing bridge".into();
                            app.add_system_message("Bridge connected. Initializing session state...");

                            let tx = cmd_tx.clone();
                            tokio::spawn(async move {
                                let result = spawned_bridge
                                    .call(
                                        "init",
                                        serde_json::json!({
                                            "config_path": "agent_config.json",
                                            "fast": true
                                        }),
                                    )
                                    .await;
                                let _ = tx.send(CmdResult::State(result));
                            });
                        }
                        Err(e) => {
                            app.connected = false;
                            app.phase = app::Phase::Error;
                            app.phase_note = "bridge offline".into();
                            app.add_system_message(format!(
                                "Could not start bridge: {e}\n\nRunning in offline mode. Commands that require the bridge will fail."
                            ));
                        }
                    },
                    CmdResult::Chat(r)  => app.handle_chat_result(r),
                    CmdResult::Slash(r) => app.handle_slash_result(r),
                    CmdResult::State(r) => {
                        match r {
                            Ok(v) => {
                                let needs_startup_context = !v
                                    .get("hook_context_complete")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(true);
                                app.handle_init(v);
                                if let Some(b) = &bridge {
                                    if needs_startup_context {
                                        let context_bridge = b.clone();
                                        let context_tx = cmd_tx.clone();
                                        tokio::spawn(async move {
                                            let result = context_bridge
                                                .call(
                                                    "startup_context",
                                                    serde_json::json!({
                                                        "include_memory_summary": false,
                                                        "include_hooks": true,
                                                        "fast_hooks": false
                                                    }),
                                                )
                                                .await;
                                            let _ = context_tx.send(CmdResult::StartupContext(result));
                                        });
                                    }
                                    let warm_bridge = b.clone();
                                    let warm_tx = cmd_tx.clone();
                                    tokio::spawn(async move {
                                        let result = warm_bridge
                                            .call(
                                                "state",
                                                serde_json::json!({
                                                    "include_repo_library": true,
                                                    "include_rag_inventory": true
                                                }),
                                            )
                                            .await;
                                        let _ = warm_tx.send(CmdResult::WarmState(result));
                                    });
                                }
                            }
                            Err(e) => app.add_system_message(format!("Bridge init failed: {e}")),
                        }
                    },
                    CmdResult::WarmState(r) => {
                        if let Ok(v) = r {
                            app.apply_bridge_state(&v);
                        }
                    }
                    CmdResult::StartupContext(r) => {
                        if let Ok(v) = r {
                            app.handle_startup_context(v);
                        }
                    }
                }
            }

            // ── Keyboard ──────────────────────────────────────────────────────
            Some(Ok(event)) = crossterm_events.next() => {
                match event {
                    Event::Key(key) => {
                        match app.handle_key(key) {
                            KeyAction::Quit => break,
                            KeyAction::Interrupt => {
                                // Abort the in-flight tokio task (drops the oneshot receiver)
                                if let Some(handle) = active_task.take() {
                                    handle.abort();
                                }
                                // Best-effort cancel notification to Python bridge
                                if let Some(b) = &bridge {
                                    let b = b.clone();
                                    tokio::spawn(async move { b.send_cancel().await });
                                }
                                app.handle_interrupt();
                            }
                            KeyAction::Submit(mut text) => {
                                let mut parsed_slash = if text.starts_with('/') {
                                    app.parse_slash_command(&text).ok()
                                } else {
                                    None
                                };
                                let trimmed = text.trim();
                                if trimmed == "/mount" || trimmed == "/mount-code" {
                                    // 1. Suspend UI
                                    let _ = disable_raw_mode();
                                    let _ = execute!(
                                        terminal.backend_mut(),
                                        LeaveAlternateScreen,
                                        DisableBracketedPaste,
                                        DisableMouseCapture,
                                    );
                                    let _ = terminal.show_cursor();

                                    // 2. Run fzf over directories
                                    let sh_script = "if command -v fd >/dev/null 2>&1; then fd -t d; else find . -type d -not -path '*/\\.*' 2>/dev/null; fi | fzf --prompt='Mount dir> '";
                                    let output = std::process::Command::new("sh")
                                        .arg("-c")
                                        .arg(sh_script)
                                        .stdin(std::process::Stdio::inherit())
                                        .stdout(std::process::Stdio::piped())
                                        .stderr(std::process::Stdio::inherit())
                                        .output();

                                    // 3. Resume UI
                                    let _ = enable_raw_mode();
                                    let _ = execute!(
                                        terminal.backend_mut(),
                                        EnterAlternateScreen,
                                        EnableBracketedPaste,
                                        EnableMouseCapture
                                    );
                                    let _ = terminal.hide_cursor();
                                    let _ = terminal.clear();

                                    if let Ok(out) = output {
                                        if out.status.success() {
                                            if let Ok(dir) = String::from_utf8(out.stdout) {
                                                let dir = dir.trim();
                                                if !dir.is_empty() {
                                                    text = format!("{} {} **/*.{{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}}", trimmed, dir);
                                                    parsed_slash = app.parse_slash_command(&text).ok();
                                                } else {
                                                    continue;
                                                }
                                            } else {
                                                continue;
                                            }
                                        } else if out.status.code() == Some(127) {
                                            app.add_system_message("fzf is not installed. Please install fzf for interactive folder selection.");
                                            continue;
                                        } else {
                                            continue;
                                        }
                                    } else {
                                        app.add_system_message("Failed to execute shell command for fzf.");
                                        continue;
                                    }
                                }

                                if let Some(b) = &bridge {
                                    active_task = dispatch_command(
                                        b.clone(),
                                        text,
                                        parsed_slash,
                                        cmd_tx.clone(),
                                    );
                                } else {
                                    app.add_system_message("No bridge connected.");
                                    app.handle_chat_result(Err(anyhow::anyhow!("no bridge")));
                                }
                            }
                            KeyAction::ToggleTrace => app.toggle_trace(),
                            KeyAction::ToggleRawStream => app.toggle_raw_stream(),
                            KeyAction::ToggleTasks => app.toggle_todo(),
                            KeyAction::ToggleContextExplorer => app.toggle_context_explorer(),
                            KeyAction::ToggleRagPanel => app.toggle_rag_panel(),
                            KeyAction::ToggleToolOutput => app.toggle_tool_output(),
                            KeyAction::ExpandThinking => app.expand_all_thinking_messages(),
                            KeyAction::None => {}
                        }
                    }
                    Event::Paste(text) => {
                        app.handle_paste(&text);
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            app.handle_mouse_scroll(mouse.column, mouse.row, true)
                        }
                        MouseEventKind::ScrollDown => {
                            app.handle_mouse_scroll(mouse.column, mouse.row, false)
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            app.handle_mouse_click(mouse.column, mouse.row)
                        }
                        _ => {}
                    }
                    Event::Resize(_, _) => {
                        terminal.autoresize()?;
                    }
                    _ => {}
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

// ── Dispatch a submit to the bridge ──────────────────────────────────────────

/// Returns an AbortHandle for long-running bridge calls, or None for immediate
/// local dispatches (/quit, /status) that don't spawn a background task.
fn dispatch_command(
    bridge: PythonBridge,
    text: String,
    parsed_slash: Option<ParsedSlashCommand>,
    tx: mpsc::UnboundedSender<CmdResult>,
) -> Option<tokio::task::AbortHandle> {
    if let Some(parsed) = parsed_slash {
        match parsed.dispatch() {
            SlashDispatch::State => {
                let handle = tokio::spawn(async move {
                    let result = bridge.call("state", serde_json::json!({})).await;
                    let _ = tx.send(CmdResult::Slash(result.map(|v| {
                        serde_json::json!({
                            "messages": [],
                            "state": v,
                        })
                    })));
                });
                return Some(handle.abort_handle());
            }
            SlashDispatch::Quit => {
                let _ = tx.send(CmdResult::Slash(Ok(serde_json::json!({
                    "messages": ["Goodbye."],
                    "state": {},
                    "exit": true,
                }))));
                return None;
            }
            SlashDispatch::Bridge | SlashDispatch::Local => {}
        }

        let handle = tokio::spawn(async move {
            let result = bridge
                .call(
                    "slash",
                    serde_json::json!({
                        "raw": text,
                        "command": parsed.command(),
                        "usage": parsed.usage(),
                        "positionals": parsed.positionals,
                        "named_args": parsed.named_args,
                        "config_path": "agent_config.json"
                    }),
                )
                .await;
            let _ = tx.send(CmdResult::Slash(result));
        });
        Some(handle.abort_handle())
    } else {
        let handle = tokio::spawn(async move {
            let result = bridge
                .call("chat", serde_json::json!({"message": text}))
                .await;
            let _ = tx.send(CmdResult::Chat(result));
        });
        Some(handle.abort_handle())
    }
}
