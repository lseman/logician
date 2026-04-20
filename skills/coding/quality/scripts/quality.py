from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from skills.coding.bootstrap.runtime_access import get_coding_runtime

__skill__ = {
    "name": "Quality",
    "description": "Use for linting, formatting, tests, type checks, and quality-gate workflows.",
    "aliases": ["tests", "linting", "formatting", "verification"],
    "triggers": ["run tests", "lint the code", "format the files", "run the quality checks"],
    "preferred_tools": ["run_pytest", "run_ruff", "run_mypy", "run_quality_check"],
    "example_queries": [
        "run the targeted test file",
        "lint and format these changes",
        "run the quality gate before commit",
    ],
    "when_not_to_use": ["no validation step is needed and the task is purely exploratory"],
    "next_skills": ["git", "shell"],
    "preferred_sequence": ["run targeted checks", "inspect failures", "broaden coverage if needed"],
    "entry_criteria": [
        "A meaningful code change was made and needs validation.",
        "A failure report or regression needs to be reproduced and summarized.",
    ],
    "decision_rules": [
        "Start with the narrowest check that matches the touched files or language.",
        "Use Python-native tools for Python changes; use shell-driven project checks for other ecosystems such as Rust or Node.",
        "Escalate from targeted checks to broader suites only when the narrow pass is insufficient.",
    ],
    "workflow": [
        "Prefer targeted checks before full-suite runs.",
        "Match the checks to the files and languages touched.",
        "Return failures with the shortest actionable summary.",
    ],
    "failure_recovery": [
        "If a tool is unavailable, fall back to the closest project-native check through the shell skill.",
        "If failures are noisy, summarize the first actionable error and identify the file or test to inspect next.",
    ],
    "exit_criteria": [
        "At least one meaningful verification step has run for the changed behavior.",
        "Failures are reduced to a short actionable summary or the checks pass cleanly.",
    ],
    "anti_patterns": [
        "Jumping straight to a full test suite when a targeted check would do.",
        "Treating Python quality tools as the only validation path in a multi-language repo.",
    ],
}


def _runtime():
    return get_coding_runtime(globals())


def _resolve_target(path: str) -> str:
    """Resolved absolute path, falling back to configured cwd."""
    if path:
        return str(_runtime().resolve_path(path))
    cwd = _runtime().cwd()
    return cwd or "."


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def run_pytest(
    path: str = "",
    flags: str = "",
    timeout: int = 120,
    venv_path: str = "",
) -> str:
    """Use when: Verify correctness of code by running the test suite.

    Triggers: run tests, pytest, test suite, check tests, unit tests, failing tests, pass tests.
    Avoid when: You only want to lint — use run_ruff or run_mypy instead.
    Inputs:
      path (str, optional): File or directory to test (default: configured cwd).
      flags (str, optional): Extra pytest flags, e.g. "-x -k test_foo".
      timeout (int, optional): Seconds before kill (default 120).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with summary (passed/failed/errors) and per-test results.
    Side effects: Executes test suite; may have side effects from tests.
    """
    target = _resolve_target(path)
    cmd = f"python -m pytest {target} {flags} --tb=short -q --no-header 2>&1"
    r = _runtime().run_cmd(
        cmd,
        cwd=_runtime().cwd(),
        timeout=timeout,
        venv_path=venv_path or None,
    )
    out = r["stdout"] + r["stderr"]

    # Parse summary line: "3 passed, 1 failed, 2 errors in 0.42s"
    summary = {"passed": 0, "failed": 0, "errors": 0, "warnings": 0}
    for m in re.finditer(r"(\d+) (passed|failed|error|warning)", out, re.IGNORECASE):
        key = m.group(2).lower().rstrip("s")
        key = (
            "errors" if key == "error" else key + ("s" if not key.endswith("s") else "")
        )
        if key in summary:
            summary[key] = int(m.group(1))

    # Parse individual failures: "FAILED path/test_foo.py::TestFoo::test_bar - AssertionError"
    failures = []
    for m in re.finditer(r"FAILED (.+?)::(.+?) - (.+)", out):
        failures.append(
            {
                "file": m.group(1),
                "test": m.group(2),
                "reason": m.group(3).strip(),
            }
        )

    # Parse error locations from short traceback: "path/file.py:42: SomeError"
    error_locs = []
    for m in re.finditer(r"^([\w/\\.]+\.py):(\d+): (\w.*)", out, re.MULTILINE):
        error_locs.append(
            {"file": m.group(1), "line": int(m.group(2)), "message": m.group(3)}
        )

    return _safe_json(
        {
            "status": "ok" if r["exit_code"] == 0 else "failed",
            "exit_code": r["exit_code"],
            "summary": summary,
            "failures": failures,
            "error_locations": error_locs[:20],
            "raw_output": out[-4000:] if len(out) > 4000 else out,
        }
    )


def run_ruff(
    path: str = "",
    fix: bool = False,
    venv_path: str = "",
) -> str:
    """Use when: Lint Python code to find style/correctness issues before running tests.

    Triggers: lint, ruff, style check, pep8, flake8, linting errors, code quality, violations.
    Avoid when: You want type errors — use run_mypy; you want test failures — use run_pytest.
    Inputs:
      path (str, optional): File or directory to lint (default: configured cwd).
      fix (bool, optional): Apply auto-fixable fixes in-place (default False).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with list of violations {file, line, col, code, message}.
    Side effects: If fix=True, modifies source files.
    """
    target = _resolve_target(path)
    fix_flag = "--fix" if fix else ""
    cmd = f"python -m ruff check {fix_flag} --output-format=json {target} 2>&1"
    r = _runtime().run_cmd(
        cmd,
        cwd=_runtime().cwd(),
        timeout=60,
        venv_path=venv_path or None,
    )

    # ruff --output-format=json writes JSON to stdout
    violations = []
    try:
        raw = r["stdout"].strip()
        # find the JSON array even if there's extra output
        start = raw.find("[")
        if start != -1:
            data = json.loads(raw[start:])
            for item in data:
                loc = item.get("location", {})
                violations.append(
                    {
                        "file": item.get("filename", ""),
                        "line": loc.get("row", 0),
                        "col": loc.get("column", 0),
                        "code": item.get("code", ""),
                        "message": item.get("message", ""),
                        "fixable": item.get("fix") is not None,
                    }
                )
    except Exception:
        # fallback: parse text format "path.py:10:5: E501 ..."
        for m in re.finditer(
            r"^(.+?):(\d+):(\d+): ([A-Z]\d+) (.+)$", r["stdout"], re.MULTILINE
        ):
            violations.append(
                {
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "col": int(m.group(3)),
                    "code": m.group(4),
                    "message": m.group(5),
                }
            )

    return _safe_json(
        {
            "status": "ok" if r["exit_code"] == 0 else "violations_found",
            "exit_code": r["exit_code"],
            "count": len(violations),
            "fixed": fix,
            "violations": violations[:100],
        }
    )


def auto_format(path: str = "", venv_path: str = "") -> str:
    """Use when: You want to automatically format a file or directory to meet style guidelines.

    Triggers: format code, auto-format, run black, run ruff format, reformat.
    Avoid when: You only want to see errors without changing files — use run_ruff instead.
    Inputs:
      path (str, optional): File or directory to format (default: configured cwd).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with formatting status and output.
    Side effects: Modifies source files in-place.
    """
    target = _resolve_target(path)
    cwd = _runtime().cwd()
    v = venv_path or None

    # First try ruff (fixes + formatting)
    fixed = _runtime().run_cmd(f"python -m ruff check --fix {target} 2>&1", cwd=cwd, timeout=60, venv_path=v)
    formatted = _runtime().run_cmd(f"python -m ruff format {target} 2>&1", cwd=cwd, timeout=60, venv_path=v)

    if "No module named ruff" not in formatted["stdout"] and formatted["exit_code"] == 0:
        return _safe_json({
            "status": "ok",
            "tool": "ruff",
            "fix_output": fixed["stdout"].strip()[-1000:],
            "format_output": formatted["stdout"].strip()[-1000:],
        })

    # Fallback to black and isort
    isort_res = _runtime().run_cmd(f"python -m isort {target} 2>&1", cwd=cwd, timeout=60, venv_path=v)
    black_res = _runtime().run_cmd(f"python -m black {target} 2>&1", cwd=cwd, timeout=60, venv_path=v)

    overall_exit = black_res["exit_code"] if black_res["exit_code"] != 0 else isort_res["exit_code"]

    return _safe_json({
        "status": "ok" if overall_exit == 0 else "error",
        "tool": "black+isort",
        "isort_output": isort_res["stdout"].strip()[-500:],
        "black_output": black_res["stdout"].strip()[-500:],
    })


def run_mypy(
    path: str = "",
    strict: bool = False,
    venv_path: str = "",
) -> str:
    """Use when: Check for type errors in Python code.

    Triggers: mypy, type check, type errors, annotation errors, typing, static analysis.
    Avoid when: You want runtime errors — use run_pytest; style issues — use run_ruff.
    Inputs:
      path (str, optional): File or directory to check (default: configured cwd).
      strict (bool, optional): Enable --strict mode (default False).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with list of errors {file, line, severity, message}.
    Side effects: Read-only.
    """
    target = _resolve_target(path)
    strict_flag = "--strict" if strict else ""
    cmd = f"python -m mypy {strict_flag} --show-column-numbers --no-error-summary {target} 2>&1"
    r = _runtime().run_cmd(
        cmd,
        cwd=_runtime().cwd(),
        timeout=120,
        venv_path=venv_path or None,
    )

    errors = []
    # Format: "file.py:10:5: error: message  [code]"
    for m in re.finditer(
        r"^(.+?):(\d+):(\d+): (error|warning|note): (.+?)(?:\s+\[(.+?)\])?$",
        r["stdout"],
        re.MULTILINE,
    ):
        errors.append(
            {
                "file": m.group(1),
                "line": int(m.group(2)),
                "col": int(m.group(3)),
                "severity": m.group(4),
                "message": m.group(5),
                "code": m.group(6) or "",
            }
        )

    err_count = sum(1 for e in errors if e["severity"] == "error")
    warn_count = sum(1 for e in errors if e["severity"] == "warning")

    return _safe_json(
        {
            "status": "ok" if r["exit_code"] == 0 else "type_errors",
            "exit_code": r["exit_code"],
            "error_count": err_count,
            "warning_count": warn_count,
            "errors": errors[:100],
        }
    )


def run_quality_check(
    path: str = "",
    skip_tests: bool = False,
    venv_path: str = "",
) -> str:
    """Use when: Verify overall code health in one call before committing or finishing a task.

    Triggers: check quality, full check, verify code, run all checks, is it ready, before commit.
    Avoid when: You only want one specific checker — call run_ruff / run_mypy / run_pytest directly.
    Inputs:
      path (str, optional): File or directory to check (default: configured cwd).
      skip_tests (bool, optional): Skip pytest (useful for untestable code, default False).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with sections for ruff, mypy, pytest and an overall pass/fail.
    Side effects: Executes test suite if skip_tests=False.
    """
    import json as _json

    target = _resolve_target(path)
    venv = venv_path or _runtime().venv_path()

    def _parse(tool_result_json: str) -> dict:
        try:
            return _json.loads(tool_result_json)
        except Exception:
            return {"status": "error", "raw": tool_result_json[:500]}

    ruff_result = _parse(run_ruff(path=target, venv_path=venv or ""))
    mypy_result = _parse(run_mypy(path=target, venv_path=venv or ""))
    pytest_result = (
        _parse(run_pytest(path=target, venv_path=venv or ""))
        if not skip_tests
        else {"status": "skipped"}
    )

    lint_ok = ruff_result.get("count", 0) == 0
    type_ok = mypy_result.get("error_count", 0) == 0
    test_ok = pytest_result.get("status") in ("ok", "skipped")

    overall = "pass" if (lint_ok and type_ok and test_ok) else "fail"

    return _safe_json(
        {
            "status": overall,
            "ruff": {
                "ok": lint_ok,
                "violations": ruff_result.get("count", "?"),
                "top_issues": ruff_result.get("violations", [])[:5],
            },
            "mypy": {
                "ok": type_ok,
                "errors": mypy_result.get("error_count", "?"),
                "top_issues": mypy_result.get("errors", [])[:5],
            },
            "pytest": {
                "ok": test_ok,
                "summary": pytest_result.get("summary", {}),
                "failures": pytest_result.get("failures", [])[:3],
            },
        }
    )


def parse_traceback(text: str) -> str:
    """Use when: Analyse an error message or stack trace to find the root cause and relevant files.

    Triggers: traceback, stack trace, exception, error message, what went wrong, debug error, analyse error.
    Avoid when: The text is not a Python traceback — the result will be empty.
    Inputs:
      text (str): Raw traceback text (can include surrounding context; the relevant part is extracted).
    Returns: JSON with {exception_type, message, frames, cause_chain}.
    Side effects: Read-only.
    """
    result = {
        "exception_type": None,
        "message": None,
        "frames": [],
        "cause_chain": [],
    }

    # Extract all "Traceback (most recent call last):" blocks for chained exceptions
    tb_blocks = re.split(
        r"(?:During handling of the above exception|The above exception was the direct cause)",
        text,
    )

    def _parse_block(block: str) -> dict:
        frames = []
        # Frame lines: '  File "path", line N, in func_name'
        for m in re.finditer(
            r'^\s+File "([^"]+)", line (\d+), in (.+)$', block, re.MULTILINE
        ):
            code_line = ""
            # Try to grab the next non-empty line after the frame header as the code
            pos = m.end()
            rest = block[pos:]
            code_m = re.match(r"\n\s{4,}(.+)", rest)
            if code_m:
                code_line = code_m.group(1).strip()
            frames.append(
                {
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "function": m.group(3).strip(),
                    "code": code_line,
                }
            )

        # Last line of block is usually "ExcType: message"
        exc_type = None
        exc_msg = None
        last_lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        for ln in reversed(last_lines):
            exc_m = re.match(
                r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Stop|Break|KeyboardInterrupt|GeneratorExit|SystemExit|BaseException)[^\s:]*)\s*:\s*(.*)",
                ln,
            )
            if not exc_m:
                # generic "Type: message" pattern
                exc_m = re.match(r"^([A-Za-z_][\w.]*)\s*:\s*(.+)", ln)
            if exc_m:
                exc_type = exc_m.group(1)
                exc_msg = exc_m.group(2).strip()
                break

        return {"exception_type": exc_type, "message": exc_msg, "frames": frames}

    parsed_blocks = [
        _parse_block(b)
        for b in tb_blocks
        if "File" in b or re.search(r"[A-Za-z]+Error", b)
    ]

    if parsed_blocks:
        main = parsed_blocks[-1]
        result["exception_type"] = main["exception_type"]
        result["message"] = main["message"]
        result["frames"] = main["frames"]
        result["cause_chain"] = parsed_blocks[:-1]

    return _safe_json(result)


def detect_project(path: str = ".") -> str:
    """Use when: Starting work on an unfamiliar codebase or after switching directories.

    Triggers: what kind of project, detect project, project setup, what framework, how to run tests,
              setup project, inspect project, project info, auto detect.
    Avoid when: You already know the project type.
    Inputs:
      path (str, optional): Root directory to inspect (default: ".").
    Returns: JSON with {type, language, build_tool, test_runner, lint_tool, venv_path, entrypoints, config_files}.
    Side effects: If a .venv or venv directory is found AND no venv is currently active, auto-calls set_venv.
    """
    root = Path(_resolve_target(path))

    def _exists(*parts: str) -> bool:
        return (root / Path(*parts)).exists()

    def _read_first(filename: str) -> str:
        try:
            return (root / filename).read_text(errors="replace")[:2000]
        except OSError:
            return ""

    config_files: list[str] = []
    project_type = "unknown"
    language = "unknown"
    build_tool = None
    test_runner = None
    lint_tool = None
    venv_path = None
    entrypoints: list[str] = []

    # ---- Language detection ----
    if (
        _exists("pyproject.toml")
        or _exists("setup.py")
        or _exists("setup.cfg")
        or _exists("requirements.txt")
    ):
        language = "python"
    elif _exists("Cargo.toml"):
        language = "rust"
    elif _exists("package.json"):
        language = "javascript"
    elif _exists("go.mod"):
        language = "go"
    elif _exists("CMakeLists.txt") or _exists("Makefile"):
        language = "c/c++"

    # ---- Python specifics ----
    if language == "python":
        pyproject_text = _read_first("pyproject.toml")

        if _exists("pyproject.toml"):
            config_files.append("pyproject.toml")
            if "poetry" in pyproject_text:
                build_tool = "poetry"
                project_type = "poetry"
            elif "flit" in pyproject_text:
                build_tool = "flit"
                project_type = "flit"
            elif "hatchling" in pyproject_text or "hatch" in pyproject_text:
                build_tool = "hatch"
                project_type = "hatch"
            elif "setuptools" in pyproject_text:
                build_tool = "setuptools"
                project_type = "setuptools"
            else:
                build_tool = "setuptools"
                project_type = "python-package"

        if _exists("setup.py"):
            config_files.append("setup.py")
            if build_tool is None:
                build_tool = "setuptools"
                project_type = "python-package"

        if _exists("requirements.txt"):
            config_files.append("requirements.txt")
            if project_type == "unknown":
                project_type = "python-script"

        # Test runner
        if (
            _exists("pytest.ini")
            or _exists("conftest.py")
            or "pytest" in pyproject_text
        ):
            test_runner = "pytest"
            config_files += [f for f in ("pytest.ini", "conftest.py") if _exists(f)]
        elif _exists("tox.ini"):
            test_runner = "tox"
            config_files.append("tox.ini")

        # Linter
        if "ruff" in pyproject_text or _exists("ruff.toml"):
            lint_tool = "ruff"
        elif _exists(".flake8"):
            lint_tool = "flake8"
            config_files.append(".flake8")

        # Venv detection
        for candidate in (".venv", "venv", ".env", "env"):
            venv_candidate = root / candidate
            if (venv_candidate / "bin" / "python").exists():
                venv_path = str(venv_candidate.resolve())
                break

        # Entrypoints
        for ep in ("main.py", "app.py", "run.py", "cli.py", "__main__.py", "manage.py"):
            if _exists(ep):
                entrypoints.append(ep)
        # Check pyproject [project.scripts]
        for m in re.finditer(
            r"^\s*\[project\.scripts\].*?(?=\[|\Z)",
            pyproject_text,
            re.DOTALL | re.MULTILINE,
        ):
            for script_m in re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]+)"', m.group()):
                entrypoints.append(f"{script_m.group(1)} = {script_m.group(2)}")

    elif language == "javascript":
        config_files.append("package.json")
        pkg_text = _read_first("package.json")
        try:
            pkg = __import__("json").loads(pkg_text)
        except Exception:
            pkg = {}
        if "next" in pkg.get("dependencies", {}) or "next" in pkg.get(
            "devDependencies", {}
        ):
            project_type = "next.js"
        elif "react" in pkg.get("dependencies", {}):
            project_type = "react"
        elif "vue" in pkg.get("dependencies", {}):
            project_type = "vue"
        build_tool = "npm"
        if _exists("yarn.lock"):
            build_tool = "yarn"
        elif _exists("pnpm-lock.yaml"):
            build_tool = "pnpm"
        scripts = pkg.get("scripts", {})
        if "test" in scripts:
            test_runner = scripts["test"].split()[0] if scripts["test"] else None

    elif language == "rust":
        config_files.append("Cargo.toml")
        build_tool = "cargo"
        test_runner = "cargo test"
        project_type = "rust"

    # ---- Docker / CI ----
    extra_config = []
    for f in (
        "Dockerfile",
        ".github/workflows",
        "docker-compose.yml",
        ".gitlab-ci.yml",
        "Makefile",
    ):
        if _exists(f):
            extra_config.append(f)
    config_files.extend(extra_config)

    # ---- Auto-apply venv if found and not already set ----
    auto_venv = False
    current_venv = _runtime().venv_path()
    if venv_path and not current_venv:
        try:
            set_venv(venv_path)
            auto_venv = True
        except Exception:
            pass

    return _safe_json(
        {
            "status": "ok",
            "path": str(root),
            "type": project_type,
            "language": language,
            "build_tool": build_tool,
            "test_runner": test_runner,
            "lint_tool": lint_tool,
            "venv_path": venv_path,
            "venv_auto_applied": auto_venv,
            "entrypoints": entrypoints,
            "config_files": config_files,
        }
    )


def smart_quality_gate(
    path: str = ".",
    mode: Literal["fast", "balanced", "full"] = "balanced",
    venv_path: str = "",
) -> str:
    """Use when: You want a stronger, SOTA-like coding validation pass with actionable next steps.

    Triggers: intelligent check, quality gate, pre-commit check, is code ready, prioritize fixes.
    Avoid when: You only need one checker quickly; use run_ruff/run_mypy/run_pytest directly.
    Inputs:
      path (str, optional): Project/file path to validate (default ".").
      mode (str, optional): "fast" (ruff+mypy), "balanced" (ruff+mypy+pytest if configured),
                            "full" (ruff+mypy+pytest always).
      venv_path (str, optional): Virtualenv to use.
    Returns: JSON with project context, checker outputs, and prioritized fix plan.
    Side effects: Executes shell quality commands; may run tests.
    """
    import json as _json

    target = _resolve_target(path)
    selected_mode = str(mode or "balanced").strip().lower()
    if selected_mode not in {"fast", "balanced", "full"}:
        selected_mode = "balanced"

    def _parse(payload: str) -> dict:
        try:
            return _json.loads(payload)
        except Exception:
            return {"status": "error", "raw": str(payload)[:500]}

    project = _parse(detect_project(path=target))
    venv = venv_path or _runtime().venv_path() or ""

    ruff_res = _parse(run_ruff(path=target, fix=False, venv_path=venv))
    mypy_res = _parse(run_mypy(path=target, strict=False, venv_path=venv))

    should_run_tests = selected_mode == "full" or (
        selected_mode == "balanced" and bool(project.get("test_runner"))
    )
    pytest_res = (
        _parse(run_pytest(path=target, venv_path=venv))
        if should_run_tests
        else {"status": "skipped", "reason": "mode_or_project"}
    )

    lint_count = int(ruff_res.get("count", 0) or 0)
    type_errs = int(mypy_res.get("error_count", 0) or 0)
    test_fail = int((pytest_res.get("summary", {}) or {}).get("failed", 0) or 0)
    test_err = int((pytest_res.get("summary", {}) or {}).get("errors", 0) or 0)

    prioritized: list[dict] = []
    if test_fail or test_err:
        prioritized.append(
            {
                "priority": 1,
                "category": "tests",
                "count": test_fail + test_err,
                "action": "Fix failing tests first to recover behavioral correctness.",
                "examples": (pytest_res.get("failures", []) or [])[:3],
            }
        )
    if type_errs:
        prioritized.append(
            {
                "priority": 2,
                "category": "typing",
                "count": type_errs,
                "action": "Fix mypy errors next to tighten contracts and prevent runtime bugs.",
                "examples": (mypy_res.get("errors", []) or [])[:5],
            }
        )
    if lint_count:
        prioritized.append(
            {
                "priority": 3,
                "category": "lint",
                "count": lint_count,
                "action": "Resolve lint issues last; start with auto-fixable items via run_ruff(fix=True).",
                "examples": (ruff_res.get("violations", []) or [])[:5],
            }
        )

    overall = "pass" if not prioritized else "fail"
    next_commands = [
        "run_ruff(path=..., fix=True)",
        "run_mypy(path=...)",
        "run_pytest(path=...)",
    ]
    if overall == "pass":
        next_commands = ["run_quality_check(path=...)"]

    return _safe_json(
        {
            "status": overall,
            "mode": selected_mode,
            "project": {
                "type": project.get("type", "unknown"),
                "language": project.get("language", "unknown"),
                "build_tool": project.get("build_tool"),
                "test_runner": project.get("test_runner"),
                "venv_path": project.get("venv_path")
                or _runtime().venv_path(),
            },
            "summary": {
                "lint_violations": lint_count,
                "type_errors": type_errs,
                "test_failures": test_fail,
                "test_errors": test_err,
            },
            "checks": {
                "ruff": {
                    "status": ruff_res.get("status"),
                    "count": lint_count,
                },
                "mypy": {
                    "status": mypy_res.get("status"),
                    "error_count": type_errs,
                },
                "pytest": {
                    "status": pytest_res.get("status"),
                    "summary": pytest_res.get("summary", {}),
                },
            },
            "prioritized_fix_plan": prioritized,
            "recommended_next_commands": next_commands,
        }
    )


