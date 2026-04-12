"""Core Rust and Cargo execution helpers."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..execution_runtime import get_shared_execution_runtime

_DEFAULT_OUTPUT_LIMIT = 20_000
_OUTPUT_SLICE_LIMIT = 4_000


def _get_coding_runtime() -> Any | None:
    return get_shared_execution_runtime()


def _resolve_effective_cwd(cwd: str = "") -> str | None:
    runtime = _get_coding_runtime()
    if runtime is not None and hasattr(runtime, "resolve_cwd"):
        return runtime.resolve_cwd(cwd or None)
    if cwd:
        return str(Path(cwd).expanduser().resolve())
    return None


def _preview_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _truncate_text(text: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    max_chars = max(0, int(max_chars or 0))
    if max_chars <= 0 or len(text) <= max_chars:
        return text, {
            "truncated": False,
            "original_chars": len(text),
            "returned_chars": len(text),
            "omitted_chars": 0,
        }
    note = "\n...[truncated; narrow the Rust/Cargo output or increase max_output_chars]"
    keep = max(0, max_chars - len(note))
    head_len = keep // 2
    tail_len = keep - head_len
    truncated = text[:head_len] + note + (text[-tail_len:] if tail_len else "")
    return truncated, {
        "truncated": True,
        "original_chars": len(text),
        "returned_chars": len(truncated),
        "omitted_chars": max(0, len(text) - len(truncated)),
    }


def _output_slices(text: str, limit: int = _OUTPUT_SLICE_LIMIT) -> dict[str, Any]:
    if not text:
        return {"head": "", "tail": "", "line_count": 0}
    return {
        "head": text[:limit],
        "tail": text[-limit:] if len(text) > limit else text,
        "line_count": len(text.splitlines()),
    }


def _attach_output_metadata(
    output: dict[str, Any],
    *,
    raw_stdout: str,
    raw_stderr: str,
    max_output_chars: int,
) -> None:
    stdout, stdout_meta = _truncate_text(raw_stdout, max_output_chars)
    stderr, stderr_meta = _truncate_text(raw_stderr, max_output_chars)
    stdout_slices = _output_slices(raw_stdout)
    stderr_slices = _output_slices(raw_stderr)
    output["stdout"] = stdout
    output["stderr"] = stderr
    output["stdout_truncated"] = stdout_meta["truncated"]
    output["stderr_truncated"] = stderr_meta["truncated"]
    output["output_limits"] = {
        "max_output_chars": max_output_chars,
        "stdout": stdout_meta,
        "stderr": stderr_meta,
    }
    output["stdout_head"] = stdout_slices["head"]
    output["stdout_tail"] = stdout_slices["tail"]
    output["stderr_head"] = stderr_slices["head"]
    output["stderr_tail"] = stderr_slices["tail"]
    output["stdout_line_count"] = stdout_slices["line_count"]
    output["stderr_line_count"] = stderr_slices["line_count"]


def _tool_missing(name: str) -> dict[str, Any] | None:
    if shutil.which(name):
        return None
    return {
        "status": "error",
        "error": f"`{name}` executable not found on PATH",
        "executable": name,
    }


def _split_extra_args(extra_args: str = "") -> tuple[list[str], str | None]:
    text = str(extra_args or "").strip()
    if not text:
        return [], None
    try:
        return shlex.split(text), None
    except ValueError as exc:
        return [], str(exc)


def _extend_extra_args(command: list[str], extra_args: str = "") -> dict[str, Any] | None:
    args, error = _split_extra_args(extra_args)
    if error is not None:
        return {
            "status": "error",
            "error": f"Could not parse extra_args: {error}",
            "command": command,
        }
    command.extend(args)
    return None


def _cargo_feature_args(
    *,
    package: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
) -> list[str]:
    args: list[str] = []
    if package:
        args.extend(["-p", str(package).strip()])
    if all_features:
        args.append("--all-features")
    if no_default_features:
        args.append("--no-default-features")
    if features:
        args.extend(["--features", str(features).strip()])
    return args


def _run_command(
    command: list[str],
    *,
    cwd: str | None,
    timeout: int,
    max_output_chars: int,
    normalize_output: bool = True,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        raw_stdout = proc.stdout
        raw_stderr = proc.stderr
        if normalize_output:
            raw_stdout = raw_stdout.replace("\r\n", "\n").replace("\r", "\n")
            raw_stderr = raw_stderr.replace("\r\n", "\n").replace("\r", "\n")
        result: dict[str, Any] = {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "command": command,
            "cwd": cwd or os.getcwd(),
        }
        _attach_output_metadata(
            result,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            max_output_chars=max_output_chars,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        raw_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        raw_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        if normalize_output:
            raw_stdout = raw_stdout.replace("\r\n", "\n").replace("\r", "\n")
            raw_stderr = raw_stderr.replace("\r\n", "\n").replace("\r", "\n")
        result = {
            "status": "error",
            "error": f"Command timed out after {timeout}s",
            "exit_code": None,
            "command": command,
            "cwd": cwd or os.getcwd(),
        }
        _attach_output_metadata(
            result,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            max_output_chars=max_output_chars,
        )
        return result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "command": command,
            "cwd": cwd or os.getcwd(),
        }


def run_rust(
    code: str,
    cwd: str = "",
    timeout: int = 60,
    edition: str = "2021",
    mode: str = "run",
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Compile/check/run a standalone Rust snippet with rustc."""
    missing = _tool_missing("rustc")
    if missing is not None:
        return missing

    normalized_mode = str(mode or "run").strip().lower()
    if normalized_mode not in {"run", "check", "compile"}:
        return {
            "status": "error",
            "error": "mode must be one of: run, check, compile",
            "mode": mode,
        }
    normalized_edition = str(edition or "2021").strip()
    if normalized_edition not in {"2015", "2018", "2021", "2024"}:
        return {
            "status": "error",
            "error": "edition must be one of: 2015, 2018, 2021, 2024",
            "edition": edition,
        }

    resolved_cwd = _resolve_effective_cwd(cwd)
    with tempfile.TemporaryDirectory(prefix="logician-rust-") as tmpdir:
        tmp = Path(tmpdir)
        source_path = tmp / "main.rs"
        output_path = tmp / "main"
        source_path.write_text(str(code or ""), encoding="utf-8")
        rustc_command = [
            "rustc",
            "--edition",
            normalized_edition,
            str(source_path),
            "-o",
            str(output_path),
        ]
        compile_result = _run_command(
            rustc_command,
            cwd=resolved_cwd,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        compile_result["mode"] = normalized_mode
        compile_result["rustc"] = shutil.which("rustc") or "rustc"
        compile_result["source_preview"] = _preview_text(str(code or ""))
        if compile_result["status"] != "ok" or normalized_mode in {"check", "compile"}:
            return compile_result

        run_result = _run_command(
            [str(output_path)],
            cwd=resolved_cwd,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        run_result["mode"] = normalized_mode
        run_result["compile"] = {
            "status": compile_result["status"],
            "exit_code": compile_result["exit_code"],
            "stderr": compile_result["stderr"],
            "stderr_truncated": compile_result["stderr_truncated"],
        }
        run_result["rustc"] = compile_result["rustc"]
        run_result["source_preview"] = compile_result["source_preview"]
        return run_result


def cargo_check(
    cwd: str = "",
    package: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
    extra_args: str = "",
    timeout: int = 120,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo check` with structured bounded output."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "check"]
    command.extend(
        _cargo_feature_args(
            package=package,
            features=features,
            all_features=all_features,
            no_default_features=no_default_features,
        )
    )
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_build(
    cwd: str = "",
    package: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
    release: bool = False,
    extra_args: str = "",
    timeout: int = 180,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo build` with common package/feature/release controls."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "build"]
    command.extend(
        _cargo_feature_args(
            package=package,
            features=features,
            all_features=all_features,
            no_default_features=no_default_features,
        )
    )
    if release:
        command.append("--release")
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_test(
    test_name: str = "",
    cwd: str = "",
    package: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
    extra_args: str = "",
    timeout: int = 180,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo test`, optionally filtered to one test name."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "test"]
    command.extend(
        _cargo_feature_args(
            package=package,
            features=features,
            all_features=all_features,
            no_default_features=no_default_features,
        )
    )
    if test_name:
        command.append(str(test_name).strip())
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_clippy(
    cwd: str = "",
    package: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
    extra_args: str = "",
    timeout: int = 180,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo clippy` with bounded output."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "clippy"]
    command.extend(
        _cargo_feature_args(
            package=package,
            features=features,
            all_features=all_features,
            no_default_features=no_default_features,
        )
    )
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_run(
    cwd: str = "",
    package: str = "",
    bin: str = "",
    example: str = "",
    features: str = "",
    all_features: bool = False,
    no_default_features: bool = False,
    release: bool = False,
    run_args: str = "",
    extra_args: str = "",
    timeout: int = 180,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo run`, with cargo args and binary args kept separate."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    if bin and example:
        return {
            "status": "error",
            "error": "Pass only one of bin or example.",
            "bin": bin,
            "example": example,
        }
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "run"]
    command.extend(
        _cargo_feature_args(
            package=package,
            features=features,
            all_features=all_features,
            no_default_features=no_default_features,
        )
    )
    if release:
        command.append("--release")
    if bin:
        command.extend(["--bin", str(bin).strip()])
    if example:
        command.extend(["--example", str(example).strip()])
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    run_argv, run_arg_error = _split_extra_args(run_args)
    if run_arg_error is not None:
        return {
            "status": "error",
            "error": f"Could not parse run_args: {run_arg_error}",
            "command": command,
        }
    if run_argv:
        command.append("--")
        command.extend(run_argv)
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_fmt(
    cwd: str = "",
    check: bool = True,
    extra_args: str = "",
    timeout: int = 60,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo fmt`; defaults to `--check` to avoid rewriting files."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "fmt"]
    if check:
        command.append("--check")
    arg_error = _extend_extra_args(command, extra_args)
    if arg_error is not None:
        return arg_error
    return _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )


def cargo_metadata(
    cwd: str = "",
    no_deps: bool = True,
    timeout: int = 30,
    max_output_chars: int = _DEFAULT_OUTPUT_LIMIT,
) -> dict[str, Any]:
    """Run `cargo metadata` and parse JSON when possible."""
    missing = _tool_missing("cargo")
    if missing is not None:
        return missing
    resolved_cwd = _resolve_effective_cwd(cwd)
    command = ["cargo", "metadata", "--format-version", "1"]
    if no_deps:
        command.append("--no-deps")
    result = _run_command(
        command,
        cwd=resolved_cwd,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )
    try:
        raw = str(result.get("stdout") or "").strip()
        if raw and not result.get("stdout_truncated"):
            result["metadata"] = json.loads(raw)
    except Exception as exc:
        result["json_parse_error"] = str(exc)
    return result
