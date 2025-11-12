
# ─────────────────────────────────────────────────────────────────────────────
# NEW: tooling for automatic discovery (with de-dup + namespacing)
# ─────────────────────────────────────────────────────────────────────────────

# Decorator (opt-in per method). Use on methods to mark them as tools.
from typing import Any, Dict, List, Optional, get_type_hints
import inspect
import json
from core import ToolParameter


def as_tool(name: Optional[str] = None, desc: Optional[str] = None):
    """Mark an instance method as a tool. Optionally override name/description."""
    def _wrap(fn):
        setattr(fn, "_tool_exposed", True)
        if name is not None:
            setattr(fn, "_tool_name_override", name)
        if desc is not None:
            setattr(fn, "_tool_desc_override", desc)
        return fn
    return _wrap


# Helper: map Python annotations to our parameter type string (we keep "string" default)
def _ptype_from_annot(annot: Any) -> str:
    try:
        origin = getattr(annot, "__origin__", None)
        if origin is list or origin is tuple or origin is dict:
            return "string"  # keep schema simple; user passes JSON string
    except Exception:
        pass
    if annot in (int, float, bool):
        return "string"  # your ToolParameter expects "string"; keep consistent
    return "string"


def _tool_params_from_signature(func) -> List[ToolParameter]:
    sig = inspect.signature(func)
    hints: Dict[str, Any] = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass
    params: List[ToolParameter] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        # skip *args/**kwargs from auto surface (LLM interface prefers named args)
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annot = hints.get(name, str)
        ptype = _ptype_from_annot(annot)
        # Try to harvest a per-param description from docstring (simple; optional)
        pdesc = ""  # keep empty if not found
        params.append(ToolParameter(name, ptype, pdesc))
    return params


def _first_line_doc(obj) -> str:
    doc = (obj.__doc__ or "").strip()
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _iter_exposed_methods(instance):
    """
    Yield (tool_name, description, bound_method, params) for each exposed method.
    Exposure rules (in priority order):
      1) Methods decorated with @as_tool(...)
      2) Methods listed in class.__tools__ (str names or dict specs)
      3) (Optional) nothing else by default, to avoid exposing helpers accidentally

    Internal de-dup by function id so the same method is not yielded twice
    (e.g., if both decorated and also listed in __tools__).
    """
    yielded_fn_ids = set()

    # 1) decorated methods
    for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
        fn = getattr(method, "__func__", None)
        if not fn or not getattr(fn, "_tool_exposed", False):
            continue
        fid = id(fn)
        if fid in yielded_fn_ids:
            continue
        tool_name = getattr(fn, "_tool_name_override", name)
        desc = getattr(fn, "_tool_desc_override", _first_line_doc(fn)) or name
        params = _tool_params_from_signature(method)
        yielded_fn_ids.add(fid)
        yield tool_name, desc, method, params

    # 2) explicit class allow-list
    spec = getattr(type(instance), "__tools__", None)
    if spec:
        # __tools__ can be: List[str] OR Dict[str, Dict[str, Any]]
        if isinstance(spec, dict):
            items = spec.items()
        else:
            # treat list[str] as {name: {}}
            items = ((n, {}) for n in spec)

        for name, meta in items:
            if not hasattr(instance, name):
                continue
            method = getattr(instance, name)
            if not inspect.ismethod(method):
                continue
            fn = getattr(method, "__func__", None)
            if not fn:
                continue
            fid = id(fn)
            if fid in yielded_fn_ids:
                continue
            tool_name = meta.get("name", name) if isinstance(meta, dict) else name
            desc = (meta.get("desc") if isinstance(meta, dict) else None) or _first_line_doc(fn) or name
            params = _tool_params_from_signature(method)
            yielded_fn_ids.add(fid)
            yield tool_name, desc, method, params

