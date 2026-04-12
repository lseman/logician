from __future__ import annotations

__skill__ = {
    "name": "Shell",
    "description": "Use for shell commands, Python execution, background processes, and environment control.",
    "aliases": ["terminal", "command line", "subprocess", "run command"],
    "triggers": [
        "run this command",
        "run the tests",
        "start the dev server",
        "execute this python snippet",
    ],
    "preferred_tools": ["run_python", "run_shell", "start_background_process"],
    "example_queries": [
        "run pytest for this package",
        "start the app locally and capture logs",
        "execute a short Python snippet against the project",
    ],
    "when_not_to_use": ["the task is just reading or editing files and does not need execution"],
    "next_skills": ["quality", "git", "explore"],
    "preferred_sequence": ["set_working_directory", "set_venv", "run_shell", "quality"],
    "entry_criteria": [
        "You need a runtime fact, command output, or environment interaction that static inspection cannot provide.",
        "The user explicitly asked to execute a command, run tests, or launch a process.",
    ],
    "decision_rules": [
        "Prefer run_python for short Python snippets and run_shell for external CLIs.",
        "Set cwd or venv explicitly when repository or project context matters.",
        "Use background processes only for long-running servers, watchers, or tailing logs.",
    ],
    "workflow": [
        "Prefer the narrowest command that answers the question.",
        "Set cwd or venv explicitly when project state matters.",
        "Use background processes only for long-running servers or watchers.",
        "Follow execution with quality, explore, or git when needed.",
    ],
    "failure_recovery": [
        "If a command times out, retry with a narrower scope or a shorter-lived command.",
        "If a binary or module is missing, inspect the environment before retrying blindly.",
    ],
    "exit_criteria": [
        "The command produced the fact, artifact, or process state the turn needed.",
        "Any follow-up fix or verification step is clear from the output.",
    ],
    "anti_patterns": [
        "Using shell one-liners for tasks that are clearer with dedicated file or quality tools.",
        "Starting long-running processes when a short foreground command would answer the question.",
    ],
}
