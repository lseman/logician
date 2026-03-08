use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, EventStream, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

mod app;
mod bridge;
mod image;
mod markdown;
mod ui;

use app::{App, KeyAction};
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

// ── Command results sent back to the main loop from spawned tasks ──────────────

enum CmdResult {
    Chat(Result<serde_json::Value, anyhow::Error>),
    Slash(Result<serde_json::Value, anyhow::Error>),
    State(Result<serde_json::Value, anyhow::Error>),
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Parse bridge command from args or env  (default: python bridge.py)
    let bridge_cmd: String;
    let bridge_args: Vec<String>;

    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() {
        bridge_cmd = std::env::var("LOGICIAN_BRIDGE_CMD")
            .or_else(|_| std::env::var("PYTHON"))
            .unwrap_or_else(|_| "python3".into());
        bridge_args = vec![resolve_default_bridge_script()];
    } else {
        bridge_cmd = raw_args[0].clone();
        bridge_args = raw_args[1..].to_vec();
    }

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

    let result = run(&mut terminal, &bridge_cmd, &bridge_args).await;

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

    // Attempt to spawn bridge
    let bridge_args_refs: Vec<&str> = bridge_args.iter().map(|s| s.as_str()).collect();
    let (bridge, _child) = match PythonBridge::spawn(bridge_cmd, &bridge_args_refs, event_tx).await
    {
        Ok(pair) => pair,
        Err(e) => {
            // Non-fatal: run in disconnected mode
            // Create a placeholder — we'll show an error on init
            let mut app = App::new();
            app.add_system_message(format!(
                "⚠ Could not start bridge `{bridge_cmd}`: {e}\n\n\
                 Running in **offline mode** — commands will error.\n\n\
                 Set `LOGICIAN_BRIDGE_CMD` / `LOGICIAN_BRIDGE_SCRIPT` or pass `<cmd> [args]`."
            ));
            return run_loop(terminal, app, None, &mut event_rx, &mut cmd_rx, cmd_tx).await;
        }
    };

    let app = App::new();

    // Initialize bridge
    {
        let b = bridge.clone();
        let tx = cmd_tx.clone();
        tokio::spawn(async move {
            let result = b
                .call(
                    "init",
                    serde_json::json!({"config_path": "agent_config.json"}),
                )
                .await;
            match result {
                Ok(v) => {
                    let _ = tx.send(CmdResult::State(Ok(v)));
                }
                Err(e) => {
                    let _ = tx.send(CmdResult::State(Err(e)));
                }
            }
        });
    }

    run_loop(
        terminal,
        app,
        Some(bridge),
        &mut event_rx,
        &mut cmd_rx,
        cmd_tx,
    )
    .await
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    bridge: Option<PythonBridge>,
    event_rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    cmd_rx: &mut mpsc::UnboundedReceiver<CmdResult>,
    cmd_tx: mpsc::UnboundedSender<CmdResult>,
) -> Result<()> {
    let mut crossterm_events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));

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
                match cr {
                    CmdResult::Chat(r)  => app.handle_chat_result(r),
                    CmdResult::Slash(r) => app.handle_slash_result(r),
                    CmdResult::State(r) => {
                        match r {
                            Ok(v) => app.handle_init(v),
                            Err(e) => app.add_system_message(format!("Bridge init failed: {e}")),
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
                            KeyAction::Submit(text) => {
                                if let Some(b) = &bridge {
                                    dispatch_command(b.clone(), text, cmd_tx.clone());
                                } else {
                                    app.add_system_message("No bridge connected.");
                                    app.handle_chat_result(Err(anyhow::anyhow!("no bridge")));
                                }
                            }
                            KeyAction::ToggleTrace => app.toggle_trace(),
                            KeyAction::ToggleRawStream => app.toggle_raw_stream(),
                            KeyAction::None => {}
                        }
                    }
                    Event::Paste(text) => {
                        app.handle_paste(&text);
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => app.scroll_up(3),
                        MouseEventKind::ScrollDown => app.scroll_down(3),
                        _ => {}
                    },
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

fn dispatch_command(bridge: PythonBridge, text: String, tx: mpsc::UnboundedSender<CmdResult>) {
    if text.starts_with('/') {
        let lower = text.to_lowercase();
        // These are handled locally in app.handle_key → app.handle_local_slash
        // but /status and bridge commands reach here.
        let cmd = lower.trim();

        if cmd == "/status" {
            tokio::spawn(async move {
                let result = bridge.call("state", serde_json::json!({})).await;
                let _ = tx.send(CmdResult::Slash(result.map(|v| {
                    serde_json::json!({
                        "messages": [],
                        "state": v,
                    })
                })));
            });
            return;
        }

        if cmd == "/quit" || cmd == "/exit" {
            let _ = tx.send(CmdResult::Slash(Ok(serde_json::json!({
                "messages": ["Goodbye."],
                "state": {},
                "exit": true,
            }))));
            return;
        }

        tokio::spawn(async move {
            let result = bridge
                .call(
                    "slash",
                    serde_json::json!({
                        "raw": text,
                        "config_path": "agent_config.json"
                    }),
                )
                .await;
            let _ = tx.send(CmdResult::Slash(result));
        });
    } else {
        tokio::spawn(async move {
            let result = bridge
                .call("chat", serde_json::json!({"message": text}))
                .await;
            let _ = tx.send(CmdResult::Chat(result));
        });
    }
}
