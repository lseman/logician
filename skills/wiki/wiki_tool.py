"""Tool registration for wiki operations."""

from __future__ import annotations

from typing import Any

from src.tools import ToolResult


def _resolve_tool_name() -> str:
    """Return the canonical tool name for wiki operations."""
    return "wiki_ops"


def _resolve_skill_id() -> str:
    """Return the canonical skill ID for wiki operations."""
    return "wiki"


def _resolve_description() -> str:
    """Return the tool description."""
    return "Perform markdown-backed wiki operations including building, reading source notes, searching the wiki or ingested repositories, ingesting raw sources or repositories, promoting follow-up suggestions, adding documents, and health checks."


def _resolve_parameters() -> dict[str, Any]:
    """Return the tool parameter schema."""
    return {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform: list, list_raw, build, ingest_raw, ingest_repo, add_raw_paper, list_suggestions, promote_suggestion, add_document, add_raw_document, add_file, add_raw_file, update_document, write_output, load_index, list_sources, read_source_note, search, search_repo, get_document, health, lint, verify",
                "enum": [
                    "list",
                    "list_raw",
                    "build",
                    "ingest_raw",
                    "ingest_repo",
                    "add_raw_paper",
                    "list_suggestions",
                    "promote_suggestion",
                    "add_document",
                    "add_raw_document",
                    "add_file",
                    "add_raw_file",
                    "update_document",
                    "write_output",
                    "load_index",
                    "list_sources",
                    "read_source_note",
                    "search",
                    "search_repo",
                    "get_document",
                    "health",
                    "lint",
                    "verify",
                ],
            },
            "source_dir": {"type": "string", "description": "Path to the wiki source directory."},
            "raw_dir": {"type": "string", "description": "Path to the raw artifacts directory."},
            "wiki_path": {"type": "string", "description": "Path to the wiki.md file."},
            "vault_dir": {"type": "string", "description": "Path to the compiled vault directory."},
            "relative_path": {
                "type": "string",
                "description": "Relative path within the source/raw directory.",
            },
            "content": {
                "type": "string",
                "description": "Document content for add/update operations.",
            },
            "pattern": {"type": "string", "description": "Search pattern for wiki search."},
            "repo": {
                "type": "string",
                "description": "Repository identifier for repo-aware search, such as a slug, URL, or checkout path.",
            },
            "literal": {"type": "boolean", "description": "Treat pattern as literal text."},
            "case_sensitive": {"type": "boolean", "description": "Case-sensitive search."},
            "max_matches": {"type": "integer", "description": "Maximum matches to return."},
            "max_hits_per_file": {
                "type": "integer",
                "description": "Maximum individual line hits to keep per matching repo file.",
            },
            "rebuild": {"type": "boolean", "description": "Rebuild wiki after add/update."},
            "raw_relative_path": {
                "type": "string",
                "description": "Relative path of the raw artifact to promote into a source note.",
            },
            "repo_path": {
                "type": "string",
                "description": "Local filesystem path of the repository to snapshot and ingest.",
            },
            "repo_name": {
                "type": "string",
                "description": "Optional stable display name for the repository note and raw snapshot folder.",
            },
            "paper": {
                "type": "string",
                "description": "Paper PDF URL, arXiv URL/id, or local PDF path for add_raw_paper.",
            },
            "paper_title": {
                "type": "string",
                "description": "Optional title override for an ingested paper note.",
            },
            "target_name": {
                "type": "string",
                "description": "Optional raw artifact filename or paper filename stem override.",
            },
            "source_note_path": {
                "type": "string",
                "description": "Target source note path to create when promoting raw material.",
            },
            "update_related": {
                "type": "boolean",
                "description": "Whether ingest_raw should append backlink updates to related source notes.",
            },
            "max_related_notes": {
                "type": "integer",
                "description": "Maximum related source notes to link/update during raw ingest.",
            },
            "max_suggested_pages": {
                "type": "integer",
                "description": "Maximum follow-up page suggestions to return during raw ingest.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum source text/excerpt characters for raw or paper ingest.",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum PDF pages to inspect for paper text extraction.",
            },
            "timeout_sec": {
                "type": "number",
                "description": "Download timeout in seconds for add_raw_paper.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum PDF size in bytes for add_raw_paper.",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum repo files to include in the raw snapshot catalog.",
            },
            "max_tree_entries": {
                "type": "integer",
                "description": "Maximum repo paths to include in the tree preview.",
            },
            "max_excerpt_files": {
                "type": "integer",
                "description": "Maximum representative repo files to excerpt into the snapshot.",
            },
            "max_excerpt_chars": {
                "type": "integer",
                "description": "Maximum characters to keep from each representative repo file excerpt.",
            },
            "suggestion_path": {
                "type": "string",
                "description": "Relative path of a suggested follow-up page to promote.",
            },
        },
        "required": ["operation"],
    }


def _resolve_return_type() -> str:
    """Return the expected return type."""
    return "dict"


def _resolve_example_usage() -> str:
    """Return an example usage snippet."""
    return """
# List wiki source documents
result = wiki_ops(operation="list")

# Build the wiki
result = wiki_ops(operation="build", source_dir="/path/to/source", vault_dir="/path/to/dist")

# Add a new document
result = wiki_ops(
    operation="add_document",
    relative_path="notes/my-note.md",
    content="# My Note\\n\\nThis is the content.",
    rebuild=True
)

# Promote a raw artifact into a source note scaffold
result = wiki_ops(
    operation="ingest_raw",
    raw_relative_path="research/article.txt",
    source_note_path="sources/research/article.md",
    rebuild=True
)

# Ingest a local code repository into the wiki
result = wiki_ops(
    operation="ingest_repo",
    repo_path="/path/to/repo",
    repo_name="my-repo",
    rebuild=True
)

# Download/copy and ingest a paper PDF into raw/papers/
result = wiki_ops(
    operation="add_raw_paper",
    paper="https://arxiv.org/pdf/2604.04921",
    rebuild=True
)

# Promote one suggested page from that source note
result = wiki_ops(
    operation="promote_suggestion",
    source_note_path="sources/research/article.md",
    suggestion_path="concepts/article-topic.md",
    rebuild=True
)

# Search the wiki
result = wiki_ops(
    operation="search",
    pattern="machine learning",
    literal=True,
    case_sensitive=False,
    max_matches=10
)

# Search an ingested repository checkout
result = wiki_ops(
    operation="search_repo",
    repo="highs",
    pattern="branch and bound",
    literal=True,
    max_matches=10
)

# Read one source note directly from source
result = wiki_ops(
    operation="read_source_note",
    relative_path="repos/scip.md"
)

# Check health
result = wiki_ops(operation="health", source_dir="/path/to/source")
"""


def _resolve_execute() -> ToolResult:
    """Execute the tool with the provided arguments."""
    import importlib.util
    from pathlib import Path

    try:
        module_path = Path(__file__).parent / "scripts" / "wiki_ops.py"
        spec = importlib.util.spec_from_file_location("wiki_ops", module_path)
        if spec is None or spec.loader is None:
            return ToolResult(
                status="error",
                message="Failed to load wiki_ops module",
                data={},
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        operation = arguments.get("operation", "")
        if operation not in [
            "list",
            "list_raw",
            "build",
            "ingest_raw",
            "ingest_repo",
            "add_raw_paper",
            "list_suggestions",
            "promote_suggestion",
            "add_document",
            "add_raw_document",
            "add_file",
            "add_raw_file",
            "update_document",
            "write_output",
            "load_index",
            "list_sources",
            "read_source_note",
            "search",
            "search_repo",
            "get_document",
            "health",
            "lint",
            "verify",
        ]:
            return ToolResult(
                status="error",
                message=f"Unknown operation: {operation}",
                data={},
            )

        # Map operation names to function names
        op_map = {
            "list": "wiki_list",
            "list_raw": "wiki_list_raw",
            "build": "wiki_build",
            "ingest_raw": "wiki_ingest_raw",
            "ingest_repo": "wiki_ingest_repo",
            "add_raw_paper": "wiki_add_raw_paper",
            "list_suggestions": "wiki_list_suggestions",
            "promote_suggestion": "wiki_promote_suggestion",
            "add_document": "wiki_add_document",
            "add_raw_document": "wiki_add_raw_document",
            "add_file": "wiki_add_file",
            "add_raw_file": "wiki_add_raw_file",
            "update_document": "wiki_update_document",
            "write_output": "wiki_write_output",
            "load_index": "wiki_load_index",
            "list_sources": "wiki_list_sources",
            "read_source_note": "wiki_read_source_note",
            "search": "wiki_search",
            "search_repo": "wiki_search_repo",
            "get_document": "wiki_get_document",
            "health": "wiki_health",
            "lint": "wiki_lint",
            "verify": "wiki_verify",
        }

        func_name = op_map[operation]
        func = getattr(module, func_name, None)
        if func is None:
            return ToolResult(
                status="error",
                message=f"Function {func_name} not found in wiki_ops module",
                data={},
            )

        # Prepare arguments for the function
        func_args = {}
        for param_name, param_value in arguments.items():
            if param_name in ["operation"]:
                continue
            if param_value is not None and param_value != "":
                func_args[param_name] = param_value

        try:
            result = func(**func_args)
            return ToolResult(
                status="success",
                message=f"Operation '{operation}' completed successfully",
                data=result,
            )
        except Exception as exc:
            return ToolResult(
                status="error",
                message=f"Operation failed: {str(exc)}",
                data={"operation": operation, "arguments": func_args},
            )

    except Exception as exc:
        return ToolResult(
            status="error",
            message=f"Tool execution failed: {str(exc)}",
            data={},
        )


def resolve() -> dict[str, Any]:
    """Resolve the tool configuration."""
    return {
        "name": _resolve_tool_name(),
        "skill_id": _resolve_skill_id(),
        "description": _resolve_description(),
        "parameters": _resolve_parameters(),
        "return_type": _resolve_return_type(),
        "example_usage": _resolve_example_usage(),
        "execute": _resolve_execute,
    }
