"""Structured Jupyter notebook editing helpers."""

from __future__ import annotations

import json
from typing import Any

from ..FileEditTool.tool import (
    _atomic_write_text,
    _coerce_to_newline_style,
    _detect_newline_style,
    _err,
    _unified_diff,
)
from ..FileReadTool.state import (
    ensure_snapshot_allows_existing_file_write,
    parse_structured_patch,
    refresh_snapshot_after_write,
    resolve_tool_path,
)


def notebook_edit(
    path: str,
    action: str,
    *,
    cell_index: int | None = None,
    source: str | None = None,
    cell_type: str | None = None,
    new_index: int | None = None,
    strip_outputs: bool = True,
) -> dict[str, Any]:
    try:
        notebook_path = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))

    if not notebook_path.exists():
        return _err(f"Notebook not found: {path}")
    if notebook_path.suffix.lower() != ".ipynb":
        return _err(f"Path is not a Jupyter notebook: {path}")

    prepared = ensure_snapshot_allows_existing_file_write(
        globals().get("ctx"),
        notebook_path,
        operation="edit",
    )
    if isinstance(prepared, dict):
        return prepared
    _, _, original, _ = prepared

    try:
        notebook = json.loads(original)
    except json.JSONDecodeError as exc:
        return _err(f"Notebook contains invalid JSON: {exc}")

    cells_raw = notebook.get("cells")
    if not isinstance(cells_raw, list):
        return _err("Notebook JSON is missing a valid `cells` list")

    cells = cells_raw
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"replace", "insert", "append", "delete", "clear_outputs"}:
        return _err("action must be one of: replace, insert, append, delete, clear_outputs")

    index = cell_index if cell_index is not None else new_index
    changed_index: int | None = None
    removed_index: int | None = None

    if normalized_action == "replace":
        if index is None:
            return _err("cell_index is required for replace")
        if source is None:
            return _err("source is required for replace")
        cell = _get_cell(cells, index)
        if isinstance(cell, dict) and cell.get("status") == "error":
            return cell
        assert isinstance(cell, dict)
        updated_cell = dict(cell)
        effective_type = _normalize_cell_type(cell_type or updated_cell.get("cell_type"))
        if effective_type is None:
            return _err("cell_type must be `code`, `markdown`, or `raw`")
        updated_cell["cell_type"] = effective_type
        updated_cell["source"] = str(source)
        _normalize_cell_payload(updated_cell)
        cells[index] = updated_cell
        changed_index = index
    elif normalized_action == "insert":
        if index is None:
            return _err("cell_index or new_index is required for insert")
        if source is None:
            return _err("source is required for insert")
        if index < 0 or index > len(cells):
            return _err(f"cell_index out of range for insert: {index}")
        effective_type = _normalize_cell_type(cell_type or "code")
        if effective_type is None:
            return _err("cell_type must be `code`, `markdown`, or `raw`")
        cells.insert(index, _new_cell(effective_type, str(source)))
        changed_index = index
    elif normalized_action == "append":
        if source is None:
            return _err("source is required for append")
        effective_type = _normalize_cell_type(cell_type or "code")
        if effective_type is None:
            return _err("cell_type must be `code`, `markdown`, or `raw`")
        cells.append(_new_cell(effective_type, str(source)))
        changed_index = len(cells) - 1
    elif normalized_action == "delete":
        if index is None:
            return _err("cell_index is required for delete")
        removed = _get_cell(cells, index)
        if isinstance(removed, dict) and removed.get("status") == "error":
            return removed
        del cells[index]
        removed_index = index
    elif normalized_action == "clear_outputs":
        if index is not None:
            cell = _get_cell(cells, index)
            if isinstance(cell, dict) and cell.get("status") == "error":
                return cell
            assert isinstance(cell, dict)
            _clear_outputs(cell)
            changed_index = index
        else:
            for cell in cells:
                if isinstance(cell, dict):
                    _clear_outputs(cell)

    outputs_cleared = 0
    if strip_outputs:
        for cell in cells:
            if isinstance(cell, dict) and _clear_outputs(cell):
                outputs_cleared += 1

    updated = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    updated = _coerce_to_newline_style(updated, _detect_newline_style(original))
    if updated == original:
        return {
            "status": "ok",
            "path": str(notebook_path),
            "action": normalized_action,
            "cell_count": len(cells),
            "unchanged": True,
            "outputs_cleared": outputs_cleared,
        }

    try:
        _atomic_write_text(notebook_path, updated)
    except OSError as exc:
        return _err(f"Cannot write notebook: {exc}")

    result = {
        "status": "ok",
        "path": str(notebook_path),
        "action": normalized_action,
        "cell_count": len(cells),
        "edited_cell_index": changed_index,
        "deleted_cell_index": removed_index,
        "outputs_cleared": outputs_cleared,
        "diff": _unified_diff(original, updated, str(notebook_path)),
    }
    result["structured_patch"] = parse_structured_patch(result["diff"])
    result["snapshot"] = refresh_snapshot_after_write(
        globals().get("ctx"),
        notebook_path,
        content=updated,
    )
    return result


def _get_cell(cells: list[Any], index: int) -> dict[str, Any] | dict[str, str]:
    if index < 0 or index >= len(cells):
        return _err(f"cell_index out of range: {index}")
    cell = cells[index]
    if not isinstance(cell, dict):
        return _err(f"Notebook cell at index {index} is not an object")
    return cell


def _normalize_cell_type(value: Any) -> str | None:
    cell_type = str(value or "").strip().lower()
    if cell_type in {"code", "markdown", "raw"}:
        return cell_type
    return None


def _new_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    _normalize_cell_payload(cell)
    return cell


def _normalize_cell_payload(cell: dict[str, Any]) -> None:
    cell_type = _normalize_cell_type(cell.get("cell_type")) or "code"
    cell["cell_type"] = cell_type
    cell["metadata"] = dict(cell.get("metadata") or {})
    cell["source"] = str(cell.get("source") or "")
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = list(cell.get("outputs") or [])
    else:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)


def _clear_outputs(cell: dict[str, Any]) -> bool:
    if _normalize_cell_type(cell.get("cell_type")) != "code":
        return False
    had_output = bool(cell.get("outputs")) or cell.get("execution_count") is not None
    cell["outputs"] = []
    cell["execution_count"] = None
    return had_output
