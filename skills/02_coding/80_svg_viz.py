from __future__ import annotations

# ===========================================================================
# CODE VISUALIZATION SKILL  —  80_svg_viz.py
#
# Generates self-contained SVG diagrams for code understanding:
#   • Call graphs          (function-level callers / callees)
#   • Class diagrams       (UML-style inheritance + members)
#   • Import dependency    (inter-module dependency graph)
#   • Pipeline diagrams    (sequential processing stages)
#   • Data-flow diagrams   (arbitrary node/edge JSON)
#   • Directory trees      (filesystem structure)
#
# Zero external dependencies — uses only CPython stdlib.
# All generated SVGs embed fonts / styles inline so they render correctly
# in any browser or SVG viewer without network access.
# ===========================================================================

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()  # type: ignore

if "_safe_json" not in globals():
    import json as _json_mod

    def _safe_json(obj: Any) -> str:  # type: ignore
        try:
            return _json_mod.dumps(obj, ensure_ascii=False)
        except Exception:
            return _json_mod.dumps({"status": "error", "error": repr(obj)})


import ast
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Theme — mirrors the TUI agent colour palette
# ---------------------------------------------------------------------------

_T = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "surface2": "#21262d",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "blue": "#58a6ff",
    "green": "#7ee787",
    "orange": "#ffa657",
    "red": "#f85149",
    "purple": "#bc8cff",
    "teal": "#39d353",
    "yellow": "#e3b341",
}

# Node type → fill colour
_NODE_COLORS: dict[str, str] = {
    "function": _T["surface"],
    "class": _T["surface2"],
    "module": _T["surface2"],
    "file": _T["surface"],
    "dir": _T["surface2"],
    "step": _T["surface"],
    "input": "#0d2137",
    "output": "#0d2818",
    "default": _T["surface"],
}

_FONT = "ui-monospace, 'Cascadia Code', 'SF Mono', Consolas, monospace"
_FONT_SANS = "'Segoe UI', system-ui, -apple-system, sans-serif"

# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """XML-escape a string for use in SVG text / attributes."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _trunc(s: str, n: int = 28) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _svg_header(w: int, h: int, title: str = "") -> str:
    title_tag = f"<title>{_esc(title)}</title>\n  " if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f"  {title_tag}"
        f'  <rect width="{w}" height="{h}" fill="{_T["bg"]}"/>\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _svg_defs(arrow_color: str = _T["text_muted"]) -> str:
    """Arrow-head + drop-shadow filter definitions."""
    return f"""  <defs>
    <marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{arrow_color}"/>
    </marker>
    <marker id="arr-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{_T["blue"]}"/>
    </marker>
    <marker id="arr-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{_T["green"]}"/>
    </marker>
    <marker id="arr-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{_T["orange"]}"/>
    </marker>
    <filter id="shadow" x="-15%" y="-15%" width="130%" height="130%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.5"/>
    </filter>
  </defs>\n"""


def _svg_node_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    sublabel: str = "",
    node_type: str = "default",
    accent: str = "",
    badge: str = "",
) -> str:
    fill = _NODE_COLORS.get(node_type, _NODE_COLORS["default"])
    stroke = accent or _T["border"]
    lines: list[str] = []

    # Shadow + outer rect
    lines.append(
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#shadow)"/>'
    )

    # Accent top-bar (3 px)
    if accent:
        lines.append(
            f'  <rect x="{x}" y="{y}" width="{w}" height="3" rx="0" ry="0" fill="{accent}"/>'
        )

    cx = x + w // 2
    text_y = y + (h // 2) - (9 if sublabel else 0)

    # Main label
    lines.append(
        f'  <text x="{cx}" y="{text_y}" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="{_FONT}" font-size="12" fill="{_T["text"]}" font-weight="600">'
        f"{_esc(_trunc(label, w // 7))}</text>"
    )

    if sublabel:
        lines.append(
            f'  <text x="{cx}" y="{text_y + 18}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="{_FONT}" font-size="10" fill="{_T["text_muted"]}">'
            f"{_esc(_trunc(sublabel, w // 6))}</text>"
        )

    if badge:
        bx, by = x + w - 2, y + 2
        bw = max(20, len(badge) * 6 + 8)
        lines.append(
            f'  <rect x="{bx - bw}" y="{by}" width="{bw}" height="14" rx="7" fill="{_T["surface2"]}" '
            f'stroke="{_T["border"]}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{bx - bw // 2}" y="{by + 7}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="{_FONT}" font-size="9" fill="{_T["text_muted"]}">{_esc(badge)}</text>'
        )

    return "\n".join(lines)


def _svg_edge(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "",
    marker: str = "arr",
    dashed: bool = False,
    label: str = "",
) -> str:
    c = color or _T["text_muted"]
    dash = 'stroke-dasharray="5,4" ' if dashed else ""
    # Bezier control points for a smooth curve
    mx = (x1 + x2) // 2
    path = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
    lines = [
        f'  <path d="{path}" fill="none" stroke="{c}" stroke-width="1.5" '
        f'{dash}marker-end="url(#{marker})" opacity="0.85"/>'
    ]
    if label:
        lx = (x1 + x2) // 2
        ly = (y1 + y2) // 2 - 6
        lines.append(
            f'  <text x="{lx}" y="{ly}" text-anchor="middle" '
            f'font-family="{_FONT_SANS}" font-size="9" fill="{_T["text_muted"]}">{_esc(label)}</text>'
        )
    return "\n".join(lines)


def _svg_title_bar(w: int, title: str, subtitle: str = "") -> str:
    lines = [
        f'  <rect x="0" y="0" width="{w}" height="{44 if subtitle else 32}" '
        f'fill="{_T["surface2"]}"/>',
        f'  <text x="16" y="20" font-family="{_FONT_SANS}" font-size="13" '
        f'font-weight="700" fill="{_T["blue"]}">{_esc(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'  <text x="16" y="36" font-family="{_FONT_SANS}" font-size="10" '
            f'fill="{_T["text_muted"]}">{_esc(subtitle)}</text>'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph layout helpers
# ---------------------------------------------------------------------------


def _topological_layers(
    nodes: list[str], edges: list[tuple[str, str]]
) -> dict[str, int]:
    """Assign each node a layer via Kahn's algorithm (topological BFS)."""
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in in_deg and dst in in_deg and src != dst:
            in_deg[dst] += 1
            succ[src].append(dst)

    q: deque[str] = deque(n for n in nodes if in_deg[n] == 0)
    layer: dict[str, int] = {}
    order = 0
    while q:
        node = q.popleft()
        layer[node] = layer.get(node, 0)
        order += 1
        for nxt in succ[node]:
            layer[nxt] = max(layer.get(nxt, 0), layer[node] + 1)
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    # Nodes not reached (cycles etc.) get their own layer
    max_layer = max(layer.values(), default=0)
    for n in nodes:
        if n not in layer:
            max_layer += 1
            layer[n] = max_layer
    return layer


def _layered_positions(
    nodes: list[str],
    edges: list[tuple[str, str]],
    node_w: int = 160,
    node_h: int = 48,
    h_gap: int = 80,
    v_gap: int = 20,
    top_pad: int = 60,
    left_pad: int = 40,
) -> dict[str, tuple[int, int]]:
    """Return {node_id: (cx, cy)} for a layered DAG layout."""
    if not nodes:
        return {}
    layer_map = _topological_layers(nodes, edges)
    layers: dict[int, list[str]] = defaultdict(list)
    for n, lyr in layer_map.items():
        layers[lyr].append(n)

    max_per_col = max(len(v) for v in layers.values()) if layers else 1

    pos: dict[str, tuple[int, int]] = {}
    for col, col_nodes in sorted(layers.items()):
        x = left_pad + col * (node_w + h_gap) + node_w // 2
        total_h = len(col_nodes) * (node_h + v_gap) - v_gap
        start_y = (
            top_pad + (max_per_col * (node_h + v_gap) - total_h) // 2 + node_h // 2
        )
        for row, n in enumerate(col_nodes):
            y = start_y + row * (node_h + v_gap)
            pos[n] = (x, y)
    return pos


def _tree_positions_recursive(
    node: str,
    children: dict[str, list[str]],
    depth: int,
    counter: dict[str, int],
    node_h: int,
    v_gap: int,
    h_indent: int,
) -> dict[str, tuple[int, int]]:
    pos: dict[str, tuple[int, int]] = {}
    x = 40 + depth * h_indent
    y = 60 + counter["row"] * (node_h + v_gap)
    pos[node] = (x, y)
    counter["row"] += 1
    for child in children.get(node, []):
        pos.update(
            _tree_positions_recursive(
                child, children, depth + 1, counter, node_h, v_gap, h_indent
            )
        )
    return pos


# ---------------------------------------------------------------------------
# AST parser helpers
# ---------------------------------------------------------------------------


def _parse_python_file(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _collect_python_files(root: Path, max_depth: int = 2) -> list[Path]:
    result: list[Path] = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if len(rel.parts) <= max_depth:
            result.append(p)
    return result[:60]  # cap to avoid enormous graphs


def _extract_functions_and_calls(
    tree: ast.Module,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (function names, list of caller→callee edges) from an AST."""
    funcs: list[str] = []
    edges: list[tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._scope: str = "<module>"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore
            funcs.append(node.name)
            prev = self._scope
            self._scope = node.name
            self.generic_visit(node)
            self._scope = prev

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

        def visit_Call(self, node: ast.Call) -> None:  # type: ignore
            callee: str | None = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee and self._scope != "<module>":
                edges.append((self._scope, callee))
            self.generic_visit(node)

    Visitor().visit(tree)
    return funcs, edges


def _extract_classes(
    tree: ast.Module,
) -> list[dict[str, Any]]:
    """Return [{name, bases, methods, attrs}] for each class in an AST."""
    classes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [
            (b.id if isinstance(b, ast.Name) else getattr(b, "attr", "?"))
            for b in node.bases
        ]
        methods: list[str] = []
        attrs: list[str] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        attrs.append(t.id)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attrs.append(item.target.id)
        classes.append(
            {
                "name": node.name,
                "bases": bases,
                "methods": methods[:10],
                "attrs": attrs[:8],
            }
        )
    return classes


def _extract_imports(path: Path, root: Path) -> list[tuple[str, str]]:
    """Return (importer_module, importee_module) pairs for local imports."""
    tree = _parse_python_file(path)
    if not tree:
        return []
    try:
        importer = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    except ValueError:
        importer = path.stem
    edges: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            edges.append((importer, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                edges.append((importer, alias.name.split(".")[0]))
    return edges


# ---------------------------------------------------------------------------
# SVG assembly helpers
# ---------------------------------------------------------------------------


def _save_and_return(svg: str, output_path: str, default_name: str) -> str:
    if output_path:
        p = Path(output_path).expanduser()
    else:
        p = Path("/tmp") / default_name
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")
        saved_path = str(p.resolve())
    except Exception as exc:
        return _safe_json({"status": "error", "error": f"Could not write SVG: {exc}"})
    return _safe_json(
        {
            "status": "ok",
            "path": saved_path,
            "size_bytes": len(svg.encode()),
        }
    )


# ---------------------------------------------------------------------------
# Input-format helpers
# ---------------------------------------------------------------------------


def _parse_steps_input(steps: str) -> list[dict[str, Any]]:
    """Accept steps in multiple formats so the LLM doesn't need to construct
    perfectly escaped JSON-in-JSON.

    Supported formats (tried in order):
    1. JSON array  — '[{"name": "Load", "desc": "read CSV"}, ...]'
    2. JSON object with a "steps" key — '{"steps": [...]}'
    3. Comma-separated names — 'Load, Process, Export'
    4. Newline-separated names — 'Load\nProcess\nExport'
    5. Single name — 'Pipeline'
    """
    s = steps.strip()
    # 1 & 2: try JSON first
    if s.startswith(("[", "{")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # {"steps": [...]}
                inner = parsed.get("steps") or parsed.get("stages")
                if isinstance(inner, list):
                    return inner
                # single step dict
                if "name" in parsed:
                    return [parsed]
        except json.JSONDecodeError:
            pass
    # 3: comma-separated
    if "," in s and "[" not in s:
        names = [n.strip() for n in s.split(",") if n.strip()]
        if names:
            return [{"name": n} for n in names]
    # 4: newline-separated
    lines = [ln.strip().lstrip("-•*·").strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) > 1:
        return [{"name": ln} for ln in lines]
    # 5: single name / fallback
    if lines:
        return [{"name": lines[0]}]
    return [{"name": s}]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@llm.tool(
    description=(
        "Analyse a Python file or directory and generate an SVG call graph showing "
        "which functions call which other functions."
    )
)
def svg_call_graph(
    path: str,
    output_path: str = "",
    max_depth: int = 2,
    focus: str = "",
) -> str:
    """Use when: Visualise function-level call relationships in Python source code.

    Triggers: call graph, function calls, who calls what, call diagram, callers callees,
              code flow svg, visualise functions.
    Avoid when: You need a class diagram or an import graph.
    Inputs:
      path (str, required): Python file or directory to analyse.
      output_path (str, required-ish): Absolute path for the SVG, e.g. "/tmp/call_graph.svg".
          Always specify so you know where to find the file. Defaults to /tmp/call_graph.svg.
      max_depth (int, optional): Directory recursion depth (default 2).
      focus (str, optional): If set, only include nodes reachable from/to this function name.
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes an SVG file to disk.
    """
    root = Path(path).expanduser().resolve()
    if root.is_dir():
        files = _collect_python_files(root, max_depth)
    elif root.is_file():
        files = [root]
    else:
        return _safe_json({"status": "error", "error": f"Path not found: {path}"})

    all_funcs: set[str] = set()
    all_edges: list[tuple[str, str]] = []
    for f in files:
        tree = _parse_python_file(f)
        if tree:
            funcs, edges = _extract_functions_and_calls(tree)
            all_funcs.update(funcs)
            all_edges.extend(edges)

    # Only keep edges where both endpoints are defined functions
    internal_edges = [
        (s, t) for s, t in all_edges if s in all_funcs and t in all_funcs and s != t
    ]
    # Deduplicate
    internal_edges = list(dict.fromkeys(internal_edges))

    # Focus filter: keep only nodes within 1 hop of focus
    if focus and focus in all_funcs:
        connected = {focus}
        for s, t in internal_edges:
            if s == focus:
                connected.add(t)
            if t == focus:
                connected.add(s)
        all_funcs = connected
        internal_edges = [
            (s, t) for s, t in internal_edges if s in connected and t in connected
        ]

    nodes = sorted(all_funcs)[:50]  # cap nodes
    node_w, node_h = 160, 40
    h_gap, v_gap = 80, 18
    pos = _layered_positions(
        nodes, internal_edges, node_w, node_h, h_gap, v_gap, top_pad=70
    )

    if not pos:
        return _safe_json({"status": "error", "error": "No functions found in source."})

    max_x = max(x for x, y in pos.values()) + node_w // 2 + 40
    max_y = max(y for x, y in pos.values()) + node_h // 2 + 40
    W, H = max(640, max_x), max(400, max_y)

    parts = [
        _svg_header(W, H, "Call Graph"),
        _svg_defs(_T["blue"]),
        _svg_title_bar(
            W, "Call Graph", f"{len(nodes)} functions · {len(internal_edges)} calls"
        ),
    ]

    # Edges first (behind nodes)
    for src, dst in internal_edges:
        if src in pos and dst in pos:
            sx, sy = pos[src]
            dx, dy = pos[dst]
            parts.append(
                _svg_edge(
                    sx + node_w // 2,
                    sy,
                    dx - node_w // 2,
                    dy,
                    color=_T["blue"],
                    marker="arr-blue",
                )
            )

    # Nodes
    for n in nodes:
        cx, cy = pos[n]
        x, y = cx - node_w // 2, cy - node_h // 2
        callee_count = sum(1 for _, t in internal_edges if t == n)
        caller_count = sum(1 for s, _ in internal_edges if s == n)
        badge = (
            f"→{caller_count} ←{callee_count}" if (caller_count or callee_count) else ""
        )
        parts.append(
            _svg_node_rect(
                x,
                y,
                node_w,
                node_h,
                n,
                badge=badge,
                node_type="function",
                accent=_T["blue"] if n == focus else "",
            )
        )

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "call_graph.svg")


@llm.tool(
    description=(
        "Analyse a Python file or directory and generate an SVG class diagram "
        "showing class names, inheritance, attributes, and methods."
    )
)
def svg_class_diagram(
    path: str,
    output_path: str = "",
    show_private: bool = False,
) -> str:
    """Use when: Visualise OOP structure — class hierarchy, inheritance, members.

    Triggers: class diagram, UML, class hierarchy, inheritance, OOP structure,
              class relationships, visualise classes.
    Avoid when: You want call-level or import-level relationships.
    Inputs:
      path (str, required): Python file or directory.
      output_path (str, required-ish): Absolute path for the SVG, e.g. "/tmp/class_diagram.svg".
          Always specify so you know where to find the file. Defaults to /tmp/class_diagram.svg.
      show_private (bool, optional): Include _private / __dunder members (default False).
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes an SVG file.
    """
    root = Path(path).expanduser().resolve()
    files = (
        _collect_python_files(root, 3)
        if root.is_dir()
        else ([root] if root.is_file() else [])
    )
    if not files:
        return _safe_json({"status": "error", "error": f"Path not found: {path}"})

    classes: list[dict[str, Any]] = []
    for f in files:
        tree = _parse_python_file(f)
        if tree:
            classes.extend(_extract_classes(tree))

    if not classes:
        return _safe_json({"status": "error", "error": "No classes found."})

    classes = classes[:24]  # cap

    if not show_private:
        for cls in classes:
            cls["methods"] = [
                m for m in cls["methods"] if not m.startswith("_") or m in ("__init__",)
            ]
            cls["attrs"] = [a for a in cls["attrs"] if not a.startswith("_")]

    # Calculate class box height based on member count
    LINE_H = 16
    PAD = 10
    HDR_H = 36

    def _box_height(cls: dict[str, Any]) -> int:
        n = len(cls["attrs"]) + len(cls["methods"])
        return HDR_H + max(n, 1) * LINE_H + PAD * 2

    BOX_W = 200
    H_GAP = 60
    V_GAP = 30

    # Build inheritance edges
    name_set = {c["name"] for c in classes}
    inh_edges = [
        (c["name"], base) for c in classes for base in c["bases"] if base in name_set
    ]

    nodes = [c["name"] for c in classes]
    pos_centers = _layered_positions(
        nodes,
        [(s, t) for s, t in inh_edges],
        BOX_W,
        80,
        H_GAP,
        V_GAP,
        top_pad=70,
        left_pad=40,
    )

    all_heights = {c["name"]: _box_height(c) for c in classes}
    max_h = max(all_heights.values(), default=120)

    max_cx = max((x for x, _ in pos_centers.values()), default=0) + BOX_W // 2
    max_cy = max((y for _, y in pos_centers.values()), default=0) + max_h // 2 + V_GAP
    W = max(700, max_cx + 60)
    H = max(500, max_cy + 60)

    parts = [
        _svg_header(W, H, "Class Diagram"),
        _svg_defs(_T["green"]),
        _svg_title_bar(W, "Class Diagram", f"{len(classes)} classes"),
    ]

    # Inheritance arrows (child → parent = generalisation)
    for child, parent in inh_edges:
        if child in pos_centers and parent in pos_centers:
            cx, cy = pos_centers[child]
            px, py = pos_centers[parent]
            parts.append(
                _svg_edge(
                    cx + BOX_W // 2,
                    cy,
                    px - BOX_W // 2,
                    py,
                    color=_T["green"],
                    marker="arr-green",
                    label="inherits",
                )
            )

    # Class boxes
    for c in classes:
        name = c["name"]
        if name not in pos_centers:
            continue
        cx, cy = pos_centers[name]
        bh = all_heights[name]
        bx, by = cx - BOX_W // 2, cy - bh // 2

        # Outer box
        parts.append(
            f'  <rect x="{bx}" y="{by}" width="{BOX_W}" height="{bh}" rx="6" ry="6" '
            f'fill="{_T["surface2"]}" stroke="{_T["green"]}" stroke-width="1.5" filter="url(#shadow)"/>'
        )
        # Class name header
        parts.append(
            f'  <rect x="{bx}" y="{by}" width="{BOX_W}" height="{HDR_H}" rx="6" ry="6" '
            f'fill="{_T["surface"]}"/>'
        )
        parts.append(
            f'  <rect x="{bx}" y="{by + HDR_H - 4}" width="{BOX_W}" height="4" fill="{_T["surface"]}"/>'
        )
        # Accent bar
        parts.append(
            f'  <rect x="{bx}" y="{by}" width="{BOX_W}" height="3" rx="0" fill="{_T["green"]}"/>'
        )
        # Class name
        base_str = f"({', '.join(c['bases'][:2])})" if c["bases"] else ""
        parts.append(
            f'  <text x="{cx}" y="{by + 20}" text-anchor="middle" '
            f'font-family="{_FONT}" font-size="12" font-weight="700" fill="{_T["text"]}">'
            f"{_esc(_trunc(name, 22))}</text>"
        )
        if base_str:
            parts.append(
                f'  <text x="{cx}" y="{by + 28}" text-anchor="middle" '
                f'font-family="{_FONT}" font-size="9" fill="{_T["green"]}">'
                f"{_esc(base_str)}</text>"
            )

        # Separator
        parts.append(
            f'  <line x1="{bx}" y1="{by + HDR_H}" x2="{bx + BOX_W}" y2="{by + HDR_H}" '
            f'stroke="{_T["border"]}" stroke-width="1"/>'
        )

        # Members
        y_cur = by + HDR_H + PAD + 4
        for attr in c["attrs"]:
            parts.append(
                f'  <text x="{bx + 10}" y="{y_cur}" font-family="{_FONT}" '
                f'font-size="10" fill="{_T["orange"]}">'
                f"+ {_esc(_trunc(attr, 22))}</text>"
            )
            y_cur += LINE_H
        for meth in c["methods"]:
            icon = "⚙" if meth.startswith("__") else "→"
            parts.append(
                f'  <text x="{bx + 10}" y="{y_cur}" font-family="{_FONT}" '
                f'font-size="10" fill="{_T["blue"]}">'
                f"{icon} {_esc(_trunc(meth + '()', 23))}</text>"
            )
            y_cur += LINE_H

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "class_diagram.svg")


@llm.tool(
    description=(
        "Analyse Python files in a directory and generate an SVG import dependency "
        "graph showing how modules depend on each other."
    )
)
def svg_import_graph(
    path: str,
    output_path: str = "",
    max_depth: int = 2,
    local_only: bool = True,
) -> str:
    """Use when: Understand module-level dependencies, coupling, and import order.

    Triggers: import graph, module dependencies, dependency diagram, imports, module coupling,
              visualise imports, package structure.
    Avoid when: You need function-level or class-level relationships.
    Inputs:
      path (str, required): Directory (or file) to scan.
      output_path (str, required-ish): Absolute path for the SVG, e.g. "/tmp/imports.svg".
          Always specify so you know where to find the file. Defaults to /tmp/import_graph.svg.
      max_depth (int, optional): How deep to recurse (default 2).
      local_only (bool, optional): Only show inter-project imports (default True).
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes SVG to disk.
    """
    root = Path(path).expanduser().resolve()
    if root.is_dir():
        files = _collect_python_files(root, max_depth)
    elif root.is_file():
        files = [root]
    else:
        return _safe_json({"status": "error", "error": f"Not found: {path}"})

    all_edges: list[tuple[str, str]] = []
    all_modules: set[str] = set()
    for f in files:
        for src, dst in _extract_imports(f, root):
            all_modules.add(src)
            if (
                not local_only
                or dst.split(".")[0] in {m.split(".")[0] for m in all_modules}
                or (root / (dst.replace(".", "/") + ".py")).exists()
                or (root / dst.replace(".", "/")).is_dir()
            ):
                all_modules.add(dst)
                all_edges.append((src, dst))

    if not all_modules:
        return _safe_json({"status": "error", "error": "No modules found."})

    if local_only:
        local_names = set()
        for f in files:
            try:
                local_names.add(
                    f.relative_to(root).with_suffix("").as_posix().replace("/", ".")
                )
            except ValueError:
                pass
        all_edges = [
            (s, t) for s, t in all_edges if t in local_names or s in local_names
        ]
        all_modules = {n for e in all_edges for n in e}

    nodes = sorted(set(all_modules))[:40]
    edges_deduped = list(
        dict.fromkeys(
            (s, t) for s, t in all_edges if s in set(nodes) and t in set(nodes)
        )
    )

    node_w, node_h = 180, 40
    pos = _layered_positions(nodes, edges_deduped, node_w, node_h, 80, 18, 70, 40)

    max_x = max((x for x, _ in pos.values()), default=200) + node_w // 2 + 40
    max_y = max((y for _, y in pos.values()), default=200) + node_h // 2 + 40
    W, H = max(680, max_x), max(420, max_y)

    parts = [
        _svg_header(W, H, "Import Graph"),
        _svg_defs(_T["orange"]),
        _svg_title_bar(
            W,
            "Import Dependency Graph",
            f"{len(nodes)} modules · {len(edges_deduped)} imports",
        ),
    ]

    for src, dst in edges_deduped:
        if src in pos and dst in pos:
            sx, sy = pos[src]
            dx, dy = pos[dst]
            parts.append(
                _svg_edge(
                    sx + node_w // 2,
                    sy,
                    dx - node_w // 2,
                    dy,
                    color=_T["orange"],
                    marker="arr-orange",
                )
            )

    for n in nodes:
        if n not in pos:
            continue
        cx, cy = pos[n]
        x, y = cx - node_w // 2, cy - node_h // 2
        dep_count = sum(1 for s, _ in edges_deduped if s == n)
        parts.append(
            _svg_node_rect(
                x,
                y,
                node_w,
                node_h,
                n.split(".")[-1],
                sublabel=n if "." in n else "",
                node_type="module",
                accent=_T["orange"] if dep_count > 0 else "",
                badge=f"↑{dep_count}" if dep_count else "",
            )
        )

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "import_graph.svg")


@llm.tool(
    description=(
        "Render a pipeline / workflow diagram as SVG from a description of "
        "sequential processing stages. Accepts JSON, comma-separated, or newline-separated names."
    )
)
def svg_pipeline(
    steps: str,
    title: str = "Pipeline",
    output_path: str = "",
    direction: str = "horizontal",
) -> str:
    """Use when: Illustrate a multi-step processing pipeline, ML pipeline, ETL flow, or workflow.

    Triggers: pipeline diagram, workflow, stages, processing steps, flowchart, ML pipeline,
              data pipeline, workflow svg, visualise pipeline.
    Avoid when: The graph is not sequential — use svg_data_flow for arbitrary graphs.
    Inputs:
      steps (str, required): The pipeline stages. Accepted formats (pick whichever is easiest):
          SIMPLE  — comma-separated names: "Load Data, Train Model, Evaluate, Export"
          SIMPLE  — newline-separated names (one per line)
          RICH    — JSON array: '[{"name":"Load","desc":"read CSV","tags":["IO"],"color":"#58a6ff"},{"name":"Train"},{"name":"Export"}]'
          Each JSON step: name(required), desc(optional), tags(optional list), color(optional hex).
      title (str, optional): Diagram title (default "Pipeline").
      output_path (str, required-ish): Absolute path for the output SVG, e.g. "/tmp/pipeline.svg".
          Always specify this so you know where to find the file. Defaults to /tmp/pipeline.svg.
      direction (str, optional): "horizontal" (default) or "vertical".
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes SVG to disk.
    """
    step_list: list[dict[str, Any]] = _parse_steps_input(steps)
    if not step_list:
        return _safe_json(
            {
                "status": "error",
                "error": "No steps found. Pass comma-separated names or a JSON array.",
            }
        )

    step_list = step_list[:20]  # cap

    PALETTE = [
        _T["blue"],
        _T["green"],
        _T["orange"],
        _T["purple"],
        _T["teal"],
        _T["yellow"],
    ]
    NODE_W, NODE_H = 180, 70
    ARROW_LEN = 50
    PAD = 40
    TOP = 70

    horiz = direction.lower() != "vertical"

    if horiz:
        total_w = PAD + len(step_list) * (NODE_W + ARROW_LEN) - ARROW_LEN + PAD
        total_h = TOP + NODE_H + PAD * 2
        W, H = max(700, total_w), total_h
    else:
        total_h = TOP + len(step_list) * (NODE_H + ARROW_LEN) - ARROW_LEN + PAD * 2
        total_w = PAD * 2 + NODE_W
        W, H = max(340, total_w), max(500, total_h)

    parts = [
        _svg_header(W, H, title),
        _svg_defs(),
        _svg_title_bar(W, title, f"{len(step_list)} stages"),
    ]

    positions: list[tuple[int, int]] = []
    for i in range(len(step_list)):
        if horiz:
            x = PAD + i * (NODE_W + ARROW_LEN)
            y = TOP + PAD
        else:
            x = (W - NODE_W) // 2
            y = TOP + PAD + i * (NODE_H + ARROW_LEN)
        positions.append((x, y))

    # Arrows
    for i in range(len(step_list) - 1):
        x, y = positions[i]
        nx, ny = positions[i + 1]
        if horiz:
            ax1, ay1 = x + NODE_W, y + NODE_H // 2
            ax2, ay2 = nx, ny + NODE_H // 2
        else:
            ax1, ay1 = x + NODE_W // 2, y + NODE_H
            ax2, ay2 = nx + NODE_W // 2, ny
        parts.append(_svg_edge(ax1, ay1, ax2, ay2, color=_T["text_muted"]))

    # Step boxes
    for i, (step, (bx, by)) in enumerate(zip(step_list, positions)):
        name = step.get("name", f"Step {i + 1}")
        desc = step.get("desc", "")
        tags: list[str] = step.get("tags", [])
        color = step.get("color", PALETTE[i % len(PALETTE)])

        # Box
        parts.append(
            f'  <rect x="{bx}" y="{by}" width="{NODE_W}" height="{NODE_H}" rx="8" ry="8" '
            f'fill="{_T["surface"]}" stroke="{color}" stroke-width="1.5" filter="url(#shadow)"/>'
        )
        # Top accent
        parts.append(
            f'  <rect x="{bx}" y="{by}" width="{NODE_W}" height="4" rx="0" fill="{color}"/>'
        )
        # Step number badge
        parts.append(
            f'  <text x="{bx + 10}" y="{by + 16}" font-family="{_FONT}" font-size="9" '
            f'fill="{color}" font-weight="700">{i + 1:02d}</text>'
        )
        # Name
        name_y = by + (NODE_H // 2) - (8 if desc else 0)
        parts.append(
            f'  <text x="{bx + NODE_W // 2}" y="{name_y}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="{_FONT}" font-size="12" '
            f'font-weight="600" fill="{_T["text"]}">{_esc(_trunc(name, 20))}</text>'
        )
        if desc:
            parts.append(
                f'  <text x="{bx + NODE_W // 2}" y="{name_y + 16}" text-anchor="middle" '
                f'font-family="{_FONT_SANS}" font-size="9" fill="{_T["text_muted"]}">'
                f"{_esc(_trunc(desc, 26))}</text>"
            )
        # Tags
        tx = bx + 6
        for tag in tags[:4]:
            tw = len(tag) * 6 + 8
            parts.append(
                f'  <rect x="{tx}" y="{by + NODE_H - 16}" width="{tw}" height="12" rx="6" '
                f'fill="{_T["surface2"]}" stroke="{_T["border"]}" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{tx + tw // 2}" y="{by + NODE_H - 10}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="{_FONT}" font-size="8" '
                f'fill="{_T["text_muted"]}">{_esc(tag)}</text>'
            )
            tx += tw + 4

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "pipeline.svg")


@llm.tool(
    description=(
        "Render an arbitrary directed data-flow graph as SVG from a JSON "
        "description of nodes and edges."
    )
)
def svg_data_flow(
    graph_json: str,
    title: str = "Data Flow",
    output_path: str = "",
) -> str:
    """Use when: Visualise any custom data flow, system architecture, or processing graph.

    Triggers: data flow, architecture diagram, system diagram, node graph, directed graph,
              flow diagram, custom graph svg.
    Avoid when: You have a sequential pipeline — svg_pipeline is cleaner for that.
    Inputs:
      graph_json (str, required): JSON string with "nodes" and "edges" keys.
          EXACT FORMAT EXAMPLE (pass this as a single JSON string):
          '{"nodes":[{"id":"a","label":"Input","type":"input"},{"id":"b","label":"Process","type":"step"},{"id":"c","label":"Output","type":"output"}],"edges":[{"from":"a","to":"b"},{"from":"b","to":"c"}]}'
          node type: "input" | "output" | "step" | "function" | "class" | "default"
          edge label is optional.
      title (str, optional): Diagram title.
      output_path (str, required-ish): Absolute path for the output SVG, e.g. "/tmp/dataflow.svg".
          Always specify this so you know where to find the file. Defaults to /tmp/data_flow.svg.
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes SVG file.
    """
    try:
        g = json.loads(graph_json)
        raw_nodes: list[dict] = g["nodes"]
        raw_edges: list[dict] = g.get("edges", [])
    except Exception as exc:
        example = '{"nodes":[{"id":"a","label":"In","type":"input"},{"id":"b","label":"Out","type":"output"}],"edges":[{"from":"a","to":"b"}]}'
        return _safe_json(
            {
                "status": "error",
                "error": f"Invalid graph_json: {exc}. Example: {example}",
            }
        )

    raw_nodes = raw_nodes[:50]

    node_ids = [n["id"] for n in raw_nodes]
    edges: list[tuple[str, str]] = [(e["from"], e["to"]) for e in raw_edges]

    node_w, node_h = 160, 48
    pos = _layered_positions(node_ids, edges, node_w, node_h, 80, 20, 70, 40)

    max_x = max((x for x, _ in pos.values()), default=200) + node_w // 2 + 40
    max_y = max((y for _, y in pos.values()), default=200) + node_h // 2 + 40
    W, H = max(640, max_x), max(420, max_y)

    ACCENT_FOR_TYPE = {
        "input": _T["blue"],
        "output": _T["green"],
        "step": _T["orange"],
        "function": _T["blue"],
        "class": _T["green"],
        "default": "",
    }

    parts = [
        _svg_header(W, H, title),
        _svg_defs(),
        _svg_title_bar(W, title, f"{len(raw_nodes)} nodes · {len(raw_edges)} edges"),
    ]

    # Edges
    for e in raw_edges:
        src, dst = e.get("from", ""), e.get("to", "")
        if src in pos and dst in pos:
            sx, sy = pos[src]
            dx, dy = pos[dst]
            parts.append(
                _svg_edge(
                    sx + node_w // 2,
                    sy,
                    dx - node_w // 2,
                    dy,
                    label=e.get("label", ""),
                )
            )

    # Nodes
    node_map = {n["id"]: n for n in raw_nodes}
    for nid in node_ids:
        if nid not in pos:
            continue
        n = node_map[nid]
        cx, cy = pos[nid]
        x, y = cx - node_w // 2, cy - node_h // 2
        ntype = n.get("type", "default")
        accent = n.get("color", ACCENT_FOR_TYPE.get(ntype, ""))
        parts.append(
            _svg_node_rect(
                x,
                y,
                node_w,
                node_h,
                label=n.get("label", nid),
                sublabel=n.get("desc", ""),
                node_type=ntype,
                accent=accent,
            )
        )

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "data_flow.svg")


@llm.tool(
    description=(
        "Generate an SVG tree diagram of a directory's folder/file structure, "
        "with file-type colouring and file counts."
    )
)
def svg_directory_tree(
    path: str,
    output_path: str = "",
    max_depth: int = 3,
    show_files: bool = True,
) -> str:
    """Use when: Visualise the layout and structure of a project directory as a tree.

    Triggers: directory tree, folder structure, project layout, file tree, folder tree,
              visualise directory, tree diagram.
    Avoid when: You need import or function relationships.
    Inputs:
      path (str, required): Root directory to visualise.
      output_path (str, required-ish): Absolute path for the SVG, e.g. "/tmp/tree.svg".
          Always specify so you know where to find the file. Defaults to /tmp/directory_tree.svg.
      max_depth (int, optional): Maximum directory depth (default 3).
      show_files (bool, optional): Include individual files (default True).
    Returns: JSON {status, path, size_bytes} — report the path back to the user.
    Side effects: Writes SVG file.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {path}"})

    # Collect tree nodes BFS
    Node = dict
    all_nodes: list[Node] = []
    children_map: dict[str, list[str]] = {}
    queue: deque[tuple[Path, str, int]] = deque([(root, root.name, 0)])

    EXT_COLORS: dict[str, str] = {
        ".py": _T["blue"],
        ".js": _T["yellow"],
        ".ts": _T["blue"],
        ".json": _T["orange"],
        ".md": _T["text_muted"],
        ".yaml": _T["green"],
        ".yml": _T["green"],
        ".toml": _T["orange"],
        ".svg": _T["purple"],
        ".html": _T["red"],
        ".css": _T["teal"],
        ".sh": _T["green"],
    }

    while queue:
        dir_path, node_id, depth = queue.popleft()
        if len(all_nodes) > 200:
            break
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            continue

        dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")][:12]
        files = (
            [e for e in entries if e.is_file() and not e.name.startswith(".")]
            if show_files
            else []
        )

        child_ids: list[str] = []
        for d in dirs:
            cid = f"{node_id}/{d.name}"
            fc = (
                len([x for x in d.iterdir() if x.is_file()]) if depth < max_depth else 0
            )
            all_nodes.append(
                {"id": cid, "label": d.name, "type": "dir", "file_count": fc}
            )
            child_ids.append(cid)
            if depth < max_depth:
                queue.append((d, cid, depth + 1))
        for f in files[:10]:
            fid = f"{node_id}/{f.name}"
            ext = f.suffix.lower()
            all_nodes.append(
                {
                    "id": fid,
                    "label": f.name,
                    "type": "file",
                    "ext": ext,
                    "color": EXT_COLORS.get(ext, _T["text_muted"]),
                }
            )
            child_ids.append(fid)
        children_map[node_id] = child_ids

    root_id = root.name
    all_nodes.insert(
        0, {"id": root_id, "label": root.name, "type": "dir", "file_count": 0}
    )

    if not all_nodes:
        return _safe_json({"status": "error", "error": "Empty directory."})

    NODE_H, V_GAP, H_INDENT = 24, 6, 24
    counter = {"row": 0}
    pos = _tree_positions_recursive(
        root_id, children_map, 0, counter, NODE_H, V_GAP, H_INDENT
    )

    max_x = max((x for x, _ in pos.values()), default=200) + 240
    max_y = max((y for _, y in pos.values()), default=200) + NODE_H + 40
    W = max(500, max_x)
    H = max(400, max_y + 20)

    parts = [
        _svg_header(W, H, "Directory Tree"),
        _svg_defs(),
        _svg_title_bar(W, "Directory Tree", f"{root.name}  ·  depth {max_depth}"),
    ]

    # Tree edges
    def _draw_edges(nid: str) -> None:
        if nid not in pos:
            return
        px, py = pos[nid]
        for cid in children_map.get(nid, []):
            if cid in pos:
                cx, cy = pos[cid]
                parts.append(
                    f'  <line x1="{px + 8}" y1="{py + NODE_H // 2}" '
                    f'x2="{cx}" y2="{cy + NODE_H // 2}" '
                    f'stroke="{_T["border"]}" stroke-width="1"/>'
                )
                _draw_edges(cid)

    _draw_edges(root_id)

    # Nodes
    for n in all_nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        label = n["label"]
        ntype = n.get("type", "file")
        color = n.get("color", _T["blue"] if ntype == "dir" else _T["text_muted"])
        ext = n.get("ext", "")
        fc = n.get("file_count", 0)

        icon_text = "▶" if ntype == "dir" else "·"
        icon_col = _T["yellow"] if ntype == "dir" else color

        parts.append(
            f'  <text x="{x}" y="{y + 15}" font-family="{_FONT}" font-size="11" '
            f'fill="{icon_col}" font-weight="{"700" if ntype == "dir" else "400"}">'
            f"{icon_text} {_esc(_trunc(label, 40))}"
            f"{'  ' + _esc(f'({fc} files)') if fc and ntype == 'dir' else ''}"
            f"</text>"
        )
        if ext and ntype == "file":
            ex = x + max(140, len(label) * 7 + 20)
            parts.append(
                f'  <text x="{ex}" y="{y + 14}" font-family="{_FONT}" font-size="9" '
                f'fill="{color}" opacity="0.7">{_esc(ext)}</text>'
            )

    parts.append(_svg_footer())
    return _save_and_return("\n".join(parts), output_path, "directory_tree.svg")
