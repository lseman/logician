"""HookFn adapter and HeuristicHooks container."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .types import HOOKS

# ---------------------------------------------------------------------------
# HookFn – a callable that optionally forwards a ctx dict
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HookFn:
    """
    Thin wrapper around a user-supplied callable that handles the optional
    ``ctx`` keyword argument transparently.
    """

    fn: Callable[..., Any]
    wants_ctx: bool

    def __call__(self, *args: Any, ctx: dict[str, Any]) -> Any:
        if self.wants_ctx:
            return self.fn(*args, ctx=ctx)
        return self.fn(*args)


def _wrap_hook(fn: Callable[..., Any]) -> HookFn:
    """Inspect *fn* once and return a pre-configured :class:`HookFn`."""
    wants_ctx = False
    try:
        sig = inspect.signature(fn)
        wants_ctx = "ctx" in sig.parameters
    except Exception:
        wants_ctx = False
    return HookFn(fn=fn, wants_ctx=wants_ctx)


# ---------------------------------------------------------------------------
# HeuristicHooks – holds one optional HookFn per hook slot
# ---------------------------------------------------------------------------


@dataclass
class HeuristicHooks:
    """
    Container for the six standard hook slots.  Any slot left as ``None``
    causes the engine to fall back to its built-in default.
    """

    init_solution: HookFn | None = None
    propose_move: HookFn | None = None
    apply_move: HookFn | None = None
    accept_move: HookFn | None = None
    update_params: HookFn | None = None
    restart: HookFn | None = None

    def implemented(self) -> list[str]:
        """Return the names of all non-None hook slots."""
        return [name for name in HOOKS if getattr(self, name) is not None]
