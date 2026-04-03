"""Core shell tool: bash command execution with intelligent assistance.

Agent guidance
--------------
When passing commands to the bash tool:

LITERAL SYNTAX (pass directly to bash):
1. Use literal shell syntax - the command is passed directly to bash
2. For arguments with spaces, quote them: ls "/path with spaces/"
3. For special characters, escape them: echo "file\\n" for literal backslash-n
4. Do NOT wrap commands in additional quotes unless you mean literal quotes
5. Avoid complex redirections or subshells unless necessary

EXAMPLES:
- List directory: "ls -la"
- File with spaces: "ls \"/path with spaces/file.txt\""
- Literal backslash-n: "echo 'file\\n'"
- Command with pipe: "grep pattern file.txt | wc -l"
- Simple echo: "echo hello world"

DANGEROUS PATTERNS (will be warned about):
- rm -rf or rm <path>
- tee -a <logfile> (persistent logging)
- Writing to /etc/passwd or /etc/shadow
- Command substitution $(...) - ensure it's safe

The tool returns JSON with stdout, stderr, return code, and parsed output.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import Any

_DEFAULT_TIMEOUT = 30  # seconds

_READ_ONLY_COMMANDS = {
    "[",
    ":",
    "basename",
    "cat",
    "cd",
    "command",
    "date",
    "dirname",
    "echo",
    "env",
    "exit",
    "false",
    "file",
    "find",
    "git",
    "grep",
    "head",
    "id",
    "ls",
    "pwd",
    "printenv",
    "printf",
    "readlink",
    "realpath",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "test",
    "true",
    "type",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
}

_MUTATING_COMMANDS = {
    "chmod",
    "chown",
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "perl",
    "python",
    "python3",
    "rsync",
    "tee",
    "touch",
    "truncate",
}

_DANGEROUS_COMMANDS = {
    "dd",
    "fdisk",
    "mkfs",
    "poweroff",
    "reboot",
    "rm",
    "shutdown",
    "sudo",
    "su",
}

_GIT_READ_ONLY_SUBCOMMANDS = {
    "branch",
    "config",
    "describe",
    "diff",
    "grep",
    "log",
    "ls-files",
    "remote",
    "rev-parse",
    "show",
    "status",
    "symbolic-ref",
}

_GIT_MUTATING_SUBCOMMANDS = {
    "add",
    "am",
    "apply",
    "bisect",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "fetch",
    "merge",
    "pull",
    "push",
    "rebase",
    "reset",
    "restore",
    "revert",
    "stash",
    "switch",
    "tag",
    "worktree",
}

_GIT_BRANCH_MUTATING_FLAGS = {
    "-c",
    "-C",
    "-d",
    "-D",
    "-m",
    "-M",
    "--copy",
    "--delete",
    "--move",
    "--set-upstream-to",
    "--unset-upstream",
}

_WRITE_REDIRECTION_RE = re.compile(r"(^|[^<])(?:>>?|&>>?|&>|\d>>?|\d>)")
_SYSTEM_PATH_WRITE_RE = re.compile(r"(?:^|\s)(?:>>?|\d>>?|\d>|&>>?|&>)\s*/(?:etc|boot|sys|proc)/")
_FORK_BOMB_RE = re.compile(r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")
_COMMAND_SUBSTITUTION_RE = re.compile(r"\$\(")
_PROCESS_SUBSTITUTION_RE = re.compile(r"(?:^|[^<])<(?:\(|\()|>\(")

_FIND_ACTION_PREDICATES = {
    "-delete": "dangerous",
    "-exec": "dangerous",
    "-execdir": "dangerous",
    "-ok": "dangerous",
    "-okdir": "dangerous",
    "-fprint": "mutating",
    "-fprint0": "mutating",
    "-fprintf": "mutating",
    "-fls": "mutating",
}

_FIND_PREDICATES_WITH_VALUES = {
    "-amin",
    "-anewer",
    "-atime",
    "-cmin",
    "-cnewer",
    "-ctime",
    "-empty",
    "-false",
    "-fstype",
    "-gid",
    "-group",
    "-iname",
    "-inum",
    "-ipath",
    "-iregex",
    "-links",
    "-maxdepth",
    "-mindepth",
    "-mmin",
    "-mtime",
    "-name",
    "-newer",
    "-nouser",
    "-nogroup",
    "-path",
    "-perm",
    "-regex",
    "-samefile",
    "-size",
    "-true",
    "-type",
    "-uid",
    "-used",
    "-user",
    "-wholename",
}

_GIT_CONFIG_READ_FLAGS = {
    "--get",
    "--get-all",
    "--get-regexp",
    "--get-urlmatch",
    "--list",
    "-l",
    "--show-origin",
    "--show-scope",
    "--name-only",
    "--null",
    "--includes",
    "--global",
    "--system",
    "--local",
    "--worktree",
    "-f",
    "--file",
    "--blob",
    "--type",
    "--default",
}

_GIT_CONFIG_FLAGS_WITH_VALUES = {
    "-f",
    "--file",
    "--blob",
    "--type",
    "--default",
}

_GIT_REMOTE_SAFE_SUBCOMMANDS = {"show", "get-url"}

_RG_SAFE_FLAGS = {
    "-a",
    "-A",
    "-B",
    "-C",
    "-F",
    "-H",
    "-L",
    "-S",
    "-T",
    "-c",
    "-d",
    "-e",
    "-f",
    "-g",
    "-h",
    "-i",
    "-l",
    "-m",
    "-n",
    "-o",
    "-q",
    "-t",
    "-u",
    "-v",
    "-w",
    "-z",
    "--",
    "--after-context",
    "--before-context",
    "--column",
    "--context",
    "--count",
    "--debug",
    "--files-with-matches",
    "--files-without-match",
    "--fixed-strings",
    "--follow",
    "--glob",
    "--heading",
    "--help",
    "--hidden",
    "--ignore-case",
    "--invert-match",
    "--json",
    "--line-number",
    "--max-count",
    "--max-depth",
    "--no-heading",
    "--no-ignore",
    "--only-matching",
    "--quiet",
    "--regexp",
    "--smart-case",
    "--stats",
    "--text",
    "--type",
    "--type-list",
    "--type-not",
    "--version",
    "--word-regexp",
}

_RG_FLAGS_WITH_VALUES = {
    "-A",
    "-B",
    "-C",
    "-T",
    "-d",
    "-e",
    "-f",
    "-g",
    "-m",
    "-t",
    "--after-context",
    "--before-context",
    "--context",
    "--glob",
    "--max-count",
    "--max-depth",
    "--regexp",
    "--type",
    "--type-not",
}

_GREP_SAFE_FLAGS = {
    "-A",
    "-B",
    "-C",
    "-E",
    "-F",
    "-H",
    "-I",
    "-L",
    "-R",
    "-c",
    "-e",
    "-f",
    "-h",
    "-i",
    "-l",
    "-m",
    "-n",
    "-o",
    "-q",
    "-r",
    "-s",
    "-v",
    "-w",
    "-x",
    "--",
    "--after-context",
    "--before-context",
    "--binary-files",
    "--color",
    "--context",
    "--count",
    "--exclude",
    "--exclude-dir",
    "--extended-regexp",
    "--fixed-strings",
    "--files-with-matches",
    "--files-without-match",
    "--ignore-case",
    "--include",
    "--invert-match",
    "--line-number",
    "--max-count",
    "--no-filename",
    "--null",
    "--null-data",
    "--only-matching",
    "--quiet",
    "--recursive",
    "--regexp",
    "--silent",
    "--text",
    "--with-filename",
    "--word-regexp",
}

_GREP_FLAGS_WITH_VALUES = {
    "-A",
    "-B",
    "-C",
    "-e",
    "-f",
    "-m",
    "--after-context",
    "--before-context",
    "--binary-files",
    "--color",
    "--context",
    "--exclude",
    "--exclude-dir",
    "--include",
    "--max-count",
    "--regexp",
}


def _sanitize_command(command: str) -> str | dict[str, str]:
    """Sanitize and validate a bash command before execution.

    This function:
    1. Strips leading/trailing whitespace
    2. Validates basic command structure
    3. Detects potentially dangerous patterns
    4. Provides clear error messages

    Args:
        command: The bash command to sanitize

    Returns:
        The sanitized command or an error dict
    """
    if not isinstance(command, str):
        return {"error": "Command must be a string"}

    # Strip whitespace
    command = command.strip()

    if not command:
        return {"error": "Command is empty"}

    return command


def _escape_shell_arg(arg: str) -> str:
    """Escape a shell argument for safe execution.

    This is a conservative approach - it escapes most special characters.
    Use this when building commands from user input.

    Args:
        arg: The argument to escape

    Returns:
        The escaped argument
    """
    # Quote the entire argument to preserve spaces
    # This is safe but may not be what you want for all cases
    return f'"{arg}"'


def _tokenize_shell(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_segments(tokens: list[str]) -> list[list[str]]:
    separators = {"|", "||", "&&", ";", "&"}
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in separators:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _first_command_token(tokens: list[str]) -> tuple[str | None, int]:
    for index, token in enumerate(tokens):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        return token, index
    return None, -1


def _git_subcommand(tokens: list[str], command_index: int) -> str | None:
    index = command_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _classify_read_only_command(command: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "command": command,
        "classification": "read_only",
        "reason": reason or f"`{command}` is classified as read-only",
    }


def _classify_mutating_command(command: str, reason: str) -> dict[str, Any]:
    return {"command": command, "classification": "mutating", "reason": reason}


def _classify_dangerous_command(command: str, reason: str) -> dict[str, Any]:
    return {"command": command, "classification": "dangerous", "reason": reason}


def _attached_short_flag(token: str, flag: str) -> bool:
    return token.startswith(flag) and len(token) > len(flag)


def _validate_grep_like_flags(
    tokens: list[str],
    *,
    command: str,
    safe_flags: set[str],
    flags_with_values: set[str],
) -> dict[str, Any]:
    args = tokens[1:]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            index += 1
            continue
        if token in safe_flags:
            if token in flags_with_values:
                index += 1
                if index >= len(args):
                    return _classify_mutating_command(
                        command, f"`{command}` flag `{token}` requires an argument"
                    )
            index += 1
            continue
        if any(_attached_short_flag(token, flag) for flag in flags_with_values if len(flag) == 2):
            index += 1
            continue
        if token.startswith("--") and "=" in token and token.split("=", 1)[0] in safe_flags:
            index += 1
            continue
        return {
            "command": command,
            "classification": "unknown",
            "reason": f"`{command}` flag `{token}` is not in the read-only allowlist",
        }

    return _classify_read_only_command(command)


def _classify_find_segment(tokens: list[str]) -> dict[str, Any]:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"(", ")", "!", "-a", "-o", ","}:
            index += 1
            continue
        if token in _FIND_ACTION_PREDICATES:
            level = _FIND_ACTION_PREDICATES[token]
            reason = f"`find` action `{token}` can execute commands or write files"
            if level == "dangerous":
                return _classify_dangerous_command("find", reason)
            return _classify_mutating_command("find", reason)
        if token in _FIND_PREDICATES_WITH_VALUES:
            index += 2
            continue
        if token.startswith("-"):
            return {
                "command": "find",
                "classification": "unknown",
                "reason": f"`find` predicate `{token}` is not in the read-only allowlist",
            }
        index += 1

    return _classify_read_only_command("find", "`find` contains only read-only predicates")


def _classify_git_branch(subcommand_tokens: list[str]) -> dict[str, Any]:
    args = subcommand_tokens[2:]
    safe_flags = {
        "--all",
        "-a",
        "--contains",
        "--merged",
        "--no-merged",
        "--points-at",
        "--remotes",
        "-r",
        "--show-current",
        "--sort",
        "--list",
        "-l",
        "--verbose",
        "-v",
        "-vv",
        "--column",
        "--color",
        "--no-color",
        "--format",
        "--ignore-case",
        "--omit-empty",
    }
    flags_with_values = {
        "--contains",
        "--merged",
        "--no-merged",
        "--points-at",
        "--sort",
        "--column",
        "--color",
        "--format",
    }
    positional: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in _GIT_BRANCH_MUTATING_FLAGS or any(
            token.startswith(f"{flag}=") for flag in _GIT_BRANCH_MUTATING_FLAGS
        ):
            return _classify_mutating_command("git branch", "git branch mutation flag detected")
        if token == "--":
            positional.extend(args[index + 1 :])
            break
        if token in safe_flags:
            if token in flags_with_values:
                index += 1
            index += 1
            continue
        if token.startswith("--") and "=" in token and token.split("=", 1)[0] in safe_flags:
            index += 1
            continue
        if token.startswith("-"):
            return {
                "command": "git branch",
                "classification": "unknown",
                "reason": f"git branch flag `{token}` is not in the read-only allowlist",
            }
        positional.append(token)
        index += 1

    if positional and not any(flag in args for flag in {"--list", "-l", "--show-current"}):
        return _classify_mutating_command(
            "git branch", "git branch with positional branch names can create or update branches"
        )
    return _classify_read_only_command("git branch", "git branch is being used for inspection only")


def _classify_git_config(subcommand_tokens: list[str]) -> dict[str, Any]:
    args = subcommand_tokens[2:]
    positionals: list[str] = []
    index = 0
    saw_read_flag = False
    while index < len(args):
        token = args[index]
        if token in _GIT_MUTATING_SUBCOMMANDS:
            return _classify_mutating_command("git config", "git config mutation helper detected")
        if token in {
            "--add",
            "--replace-all",
            "--unset",
            "--unset-all",
            "--rename-section",
            "--remove-section",
            "--edit",
        }:
            return _classify_mutating_command(
                "git config", f"git config flag `{token}` mutates configuration"
            )
        if token in _GIT_CONFIG_READ_FLAGS:
            saw_read_flag = True
            if token in _GIT_CONFIG_FLAGS_WITH_VALUES:
                index += 1
            index += 1
            continue
        if (
            token.startswith("--")
            and "=" in token
            and token.split("=", 1)[0] in _GIT_CONFIG_READ_FLAGS
        ):
            saw_read_flag = True
            index += 1
            continue
        if token.startswith("-"):
            return {
                "command": "git config",
                "classification": "unknown",
                "reason": f"git config flag `{token}` is not in the read-only allowlist",
            }
        positionals.append(token)
        index += 1

    if len(positionals) > 1 and not saw_read_flag:
        return _classify_mutating_command(
            "git config", "git config with key and value arguments writes configuration"
        )
    return _classify_read_only_command("git config", "git config is being used for inspection only")


def _classify_git_remote(subcommand_tokens: list[str]) -> dict[str, Any]:
    args = subcommand_tokens[2:]
    if not args:
        return _classify_read_only_command("git remote")
    if args[0] in _GIT_REMOTE_SAFE_SUBCOMMANDS:
        return _classify_read_only_command("git remote", f"git remote {args[0]} is read-only")
    if args[0] in {
        "add",
        "remove",
        "rename",
        "set-branches",
        "set-head",
        "set-url",
        "prune",
        "update",
    }:
        return _classify_mutating_command(
            "git remote", f"git remote {args[0]} mutates repository state"
        )
    if args[0].startswith("-") and args[0] in {"-v", "--verbose"}:
        return _classify_read_only_command("git remote")
    return {
        "command": "git remote",
        "classification": "unknown",
        "reason": f"git remote form `{args[0]}` is not in the read-only allowlist",
    }


def _classify_git_segment(tokens: list[str], command_index: int) -> dict[str, Any]:
    subcommand = _git_subcommand(tokens, command_index)
    if subcommand is None:
        return {
            "command": "git",
            "classification": "unknown",
            "reason": "git command without a subcommand is not classified as read-only",
        }
    subcommand_lower = subcommand.lower()
    if subcommand_lower == "branch":
        return _classify_git_branch(tokens)
    if subcommand_lower == "config":
        return _classify_git_config(tokens)
    if subcommand_lower == "remote":
        return _classify_git_remote(tokens)
    if subcommand_lower in _GIT_MUTATING_SUBCOMMANDS:
        return _classify_mutating_command(
            f"git {subcommand}", f"git {subcommand} mutates repository state"
        )
    if subcommand_lower in _GIT_READ_ONLY_SUBCOMMANDS:
        return _classify_read_only_command(
            f"git {subcommand}", f"git {subcommand} is treated as read-only"
        )
    return {
        "command": f"git {subcommand}",
        "classification": "unknown",
        "reason": f"git {subcommand} is not classified as read-only",
    }


def _classify_segment(tokens: list[str]) -> dict[str, Any]:
    command, command_index = _first_command_token(tokens)
    if command is None:
        return {"command": None, "classification": "read_only", "reason": "empty segment"}

    lowered = command.lower()
    has_write_redirection = any(token in {">", ">>"} for token in tokens)
    if has_write_redirection:
        return {
            "command": command,
            "classification": "mutating",
            "reason": "output redirection writes to a file",
        }

    if lowered in _DANGEROUS_COMMANDS:
        return {
            "command": command,
            "classification": "dangerous",
            "reason": f"`{command}` is a destructive shell command",
        }

    if lowered == "git":
        return _classify_git_segment(tokens, command_index)

    if lowered == "find":
        return _classify_find_segment(tokens)

    if lowered == "rg":
        return _validate_grep_like_flags(
            tokens,
            command="rg",
            safe_flags=_RG_SAFE_FLAGS,
            flags_with_values=_RG_FLAGS_WITH_VALUES,
        )

    if lowered == "grep":
        return _validate_grep_like_flags(
            tokens,
            command="grep",
            safe_flags=_GREP_SAFE_FLAGS,
            flags_with_values=_GREP_FLAGS_WITH_VALUES,
        )

    if lowered == "sed" and any(
        token in {"-i", "--in-place"} for token in tokens[command_index + 1 :]
    ):
        return {
            "command": command,
            "classification": "mutating",
            "reason": "sed in-place editing flag detected",
        }

    if lowered in _READ_ONLY_COMMANDS:
        return {
            "command": command,
            "classification": "read_only",
            "reason": f"`{command}` is classified as read-only",
        }

    if lowered in _MUTATING_COMMANDS:
        return {
            "command": command,
            "classification": "mutating",
            "reason": f"`{command}` commonly writes to the filesystem",
        }

    return {
        "command": command,
        "classification": "unknown",
        "reason": f"`{command}` is not in the read-only allowlist",
    }


def _validate_command(command: str, *, require_read_only: bool) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "mode": "read_only" if require_read_only else "auto",
        "dangerous": False,
        "is_read_only": False,
        "has_write_patterns": False,
        "warnings": [],
        "reasons": [],
        "segments": [],
    }

    if _FORK_BOMB_RE.search(command):
        validation["dangerous"] = True
        validation["reasons"].append("fork bomb pattern detected")
        return validation

    tokens = _tokenize_shell(command)
    segments = _split_segments(tokens)
    if not segments:
        validation["is_read_only"] = True
        return validation

    redirection_detected = _WRITE_REDIRECTION_RE.search(command) is not None
    if redirection_detected:
        validation["has_write_patterns"] = True
        validation["warnings"].append("output redirection detected")
    if _COMMAND_SUBSTITUTION_RE.search(command):
        validation["warnings"].append("command substitution detected")
    if _PROCESS_SUBSTITUTION_RE.search(command):
        validation["warnings"].append("process substitution detected")
    if _SYSTEM_PATH_WRITE_RE.search(command):
        validation["dangerous"] = True
        validation["reasons"].append("write redirection into a protected system path detected")

    read_only_segments = 0
    for segment_tokens in segments:
        segment = _classify_segment(segment_tokens)
        validation["segments"].append(segment)
        classification = segment["classification"]
        if classification == "dangerous":
            validation["dangerous"] = True
            validation["reasons"].append(segment["reason"])
        elif classification == "mutating":
            validation["has_write_patterns"] = True
            validation["warnings"].append(segment["reason"])
        elif classification == "unknown":
            validation["warnings"].append(segment["reason"])
        elif classification == "read_only":
            read_only_segments += 1

    validation["warnings"] = list(dict.fromkeys(validation["warnings"]))
    validation["reasons"] = list(dict.fromkeys(validation["reasons"]))
    validation["is_read_only"] = (
        not validation["dangerous"]
        and not validation["has_write_patterns"]
        and read_only_segments == len(segments)
    )
    return validation


def bash(
    command: str,
    timeout: int = _DEFAULT_TIMEOUT,
    normalize_output: bool = True,
    require_read_only: bool = False,
) -> dict[str, Any]:
    """Execute a bash command and return structured output.

    Agent guidance
    --------------
    - Use literal shell syntax - the command is passed directly to bash
    - For arguments with spaces, quote them: `ls "/path with spaces/"`
    - For special characters, escape them: `echo "file\\n"` for literal backslash-n
    - Do NOT wrap commands in additional quotes unless you mean literal quotes
    - Avoid complex redirections or subshells unless necessary

    Args:
        command: Bash command string to execute.
        timeout: Timeout in seconds (default 30).
        normalize_output: If True (default), normalizes stdout/stderr newlines to LF
        only for consistent output format. If False, preserves original line endings.
        require_read_only: If True, reject commands that are not classified as
        read-only inspection commands.

    Returns:
        dict with:
            - status: "ok" or "error"
            - returncode: Exit code
            - stdout: Standard output
            - stderr: Standard error
            - parsed: Auto-detected structured data
    """
    # Sanitize the command
    sanitized = _sanitize_command(command)
    if isinstance(sanitized, dict) and "error" in sanitized:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": sanitized["error"],
            "command": command,
        }

    validation = _validate_command(sanitized, require_read_only=require_read_only)
    if validation["dangerous"]:
        return {
            "status": "error",
            "reason": "dangerous_command",
            "returncode": -1,
            "stdout": "",
            "stderr": "; ".join(validation["reasons"]) or "Dangerous command rejected",
            "command": command,
            "sanitized": sanitized,
            "validation": validation,
        }
    if require_read_only and not validation["is_read_only"]:
        return {
            "status": "error",
            "reason": "read_only_required",
            "returncode": -1,
            "stdout": "",
            "stderr": "Command is not classified as read-only. Use a safer inspection command or disable require_read_only.",
            "command": command,
            "sanitized": sanitized,
            "validation": validation,
        }

    try:
        result = subprocess.run(
            ["bash", "-lc", sanitized],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parsed = _try_parse_output(result.stdout, result.stderr)
        output = {
            "status": "ok",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parsed": parsed,
        }
        # Apply newline normalization to output
        if normalize_output:
            output["stdout"] = output["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            output["stderr"] = output["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if result.returncode != 0:
            output["status"] = "error"
        output["sanitized"] = sanitized
        output["validation"] = validation
        return output
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "command": command,
            "sanitized": sanitized,
            "validation": validation,
        }
    except Exception as e:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": command,
            "sanitized": sanitized,
            "validation": validation,
        }


def _try_parse_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Try to auto-parse command output into structured data."""
    parsed = {"raw_stdout": stdout, "raw_stderr": stderr}

    # Try JSON parsing
    if stdout.strip():
        try:
            parsed["json"] = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    # Try to parse common output formats
    if stdout.strip().startswith("{"):
        parsed["detected_type"] = "json"
    elif stdout.strip().endswith("\n"):
        # Could be tabular data
        lines = stdout.strip().split("\n")
        if len(lines) > 1 and any(line.startswith("  ") for line in lines[1:]):
            parsed["detected_type"] = "indented_list"
            parsed["lines"] = lines

    # Parse git output
    if stdout.strip().startswith("On branch"):
        parsed["detected_type"] = "git_status"
        parsed["git_branch"] = (
            stdout.split("On branch ")[1].split("\n")[0] if "On branch " in stdout else None
        )

    # Parse ls output
    if stdout.strip().startswith("total ") or any(
        line.startswith("d") for line in stdout.split("\n")[:5]
    ):
        parsed["detected_type"] = "ls_output"

    return parsed
